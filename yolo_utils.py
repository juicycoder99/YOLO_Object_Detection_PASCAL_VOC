"""YOLO utilities: IoU, non-max suppression, mean Average Precision, cell decoding and drawing.

Adapted from the canonical YOLO-v1 reference implementation for the prediction layout used here:
each cell vector is [class(0:20), box1(conf,x,y,w,h), box2(conf,x,y,w,h)] (length 30).
Boxes are in midpoint format (x, y, w, h) normalised to [0, 1].
"""
import torch
from collections import Counter


def iou_midpoint(boxes_preds, boxes_labels):
    bp, bl = boxes_preds, boxes_labels
    px1, py1 = bp[..., 0:1] - bp[..., 2:3] / 2, bp[..., 1:2] - bp[..., 3:4] / 2
    px2, py2 = bp[..., 0:1] + bp[..., 2:3] / 2, bp[..., 1:2] + bp[..., 3:4] / 2
    lx1, ly1 = bl[..., 0:1] - bl[..., 2:3] / 2, bl[..., 1:2] - bl[..., 3:4] / 2
    lx2, ly2 = bl[..., 0:1] + bl[..., 2:3] / 2, bl[..., 1:2] + bl[..., 3:4] / 2
    x1, y1 = torch.max(px1, lx1), torch.max(py1, ly1)
    x2, y2 = torch.min(px2, lx2), torch.min(py2, ly2)
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_p = (px2 - px1).abs() * (py2 - py1).abs()
    area_l = (lx2 - lx1).abs() * (ly2 - ly1).abs()
    return inter / (area_p + area_l - inter + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold):
    """bboxes: list of [class, conf, x, y, w, h]."""
    bboxes = [b for b in bboxes if b[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    keep = []
    while bboxes:
        chosen = bboxes.pop(0)
        bboxes = [b for b in bboxes
                  if b[0] != chosen[0] or
                  iou_midpoint(torch.tensor(chosen[2:]), torch.tensor(b[2:])) < iou_threshold]
        keep.append(chosen)
    return keep


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20):
    """pred/true boxes: lists of [img_idx, class, conf, x, y, w, h]."""
    average_precisions = []
    for c in range(num_classes):
        detections = [d for d in pred_boxes if d[1] == c]
        ground_truths = [t for t in true_boxes if t[1] == c]
        amount_bboxes = Counter(gt[0] for gt in ground_truths)
        amount_bboxes = {k: torch.zeros(v) for k, v in amount_bboxes.items()}
        detections.sort(key=lambda x: x[2], reverse=True)
        TP, FP = torch.zeros(len(detections)), torch.zeros(len(detections))
        total_true = len(ground_truths)
        if total_true == 0:
            continue
        for di, det in enumerate(detections):
            gts = [g for g in ground_truths if g[0] == det[0]]
            best_iou, best_gt = 0, -1
            for gi, gt in enumerate(gts):
                iou = iou_midpoint(torch.tensor(det[3:]), torch.tensor(gt[3:])).item()
                if iou > best_iou:
                    best_iou, best_gt = iou, gi
            if best_iou > iou_threshold and amount_bboxes[det[0]][best_gt] == 0:
                TP[di] = 1; amount_bboxes[det[0]][best_gt] = 1
            else:
                FP[di] = 1
        TP_cum, FP_cum = torch.cumsum(TP, 0), torch.cumsum(FP, 0)
        recalls = TP_cum / (total_true + 1e-6)
        precisions = TP_cum / (TP_cum + FP_cum + 1e-6)
        precisions = torch.cat([torch.tensor([1.0]), precisions])
        recalls = torch.cat([torch.tensor([0.0]), recalls])
        average_precisions.append(torch.trapz(precisions, recalls).item())
    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0


def convert_cellboxes(predictions, S=7, C=20):
    """Convert (N,S,S,30) cell predictions to (N,S,S,6)=[class, conf, x, y, w, h] image-relative."""
    predictions = predictions.cpu()
    N = predictions.shape[0]
    predictions = predictions.reshape(N, S, S, C + 10)
    b1, b2 = predictions[..., C + 1:C + 5], predictions[..., C + 6:C + 10]
    scores = torch.cat([predictions[..., C].unsqueeze(0), predictions[..., C + 5].unsqueeze(0)], 0)
    best = scores.argmax(0).unsqueeze(-1)
    boxes = b1 * (1 - best) + best * b2
    cell_idx = torch.arange(S).repeat(N, S, 1).unsqueeze(-1)
    x = 1 / S * (boxes[..., 0:1] + cell_idx)
    y = 1 / S * (boxes[..., 1:2] + cell_idx.permute(0, 2, 1, 3))
    wh = 1 / S * boxes[..., 2:4] * S
    converted = torch.cat([x, y, wh], -1)
    pred_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_conf = torch.max(predictions[..., C], predictions[..., C + 5]).unsqueeze(-1)
    return torch.cat([pred_class, best_conf, converted], -1)


def cellboxes_to_boxes(out, S=7, C=20):
    converted = convert_cellboxes(out, S, C).reshape(out.shape[0], S * S, 6)
    converted[..., 0] = converted[..., 0].long()
    return [[ [x.item() for x in converted[ex, b, :]] for b in range(S * S)]
            for ex in range(out.shape[0])]
