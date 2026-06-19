"""Evaluation helpers for the YOLO Final Exam: gather boxes for mAP and draw sample detections."""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from yolo_utils import cellboxes_to_boxes, non_max_suppression
from yolo_data import VOC_CLASSES


def decode_targets(T, S=7, C=20):
    """Ground-truth grid (N,S,S,C+5) -> per-image list of [class, 1.0, x, y, w, h] (image-relative)."""
    out = []
    for n in range(T.shape[0]):
        boxes = []
        for i in range(S):
            for j in range(S):
                if T[n, i, j, C] == 1:
                    x = (j + T[n, i, j, C + 1].item()) / S
                    y = (i + T[n, i, j, C + 2].item()) / S
                    w, h = T[n, i, j, C + 3].item(), T[n, i, j, C + 4].item()
                    cls = int(T[n, i, j, :C].argmax().item())
                    boxes.append([cls, 1.0, x, y, w, h])
        out.append(boxes)
    return out


@torch.no_grad()
def get_bboxes(model, X, T, prep, device, batch=16, iou_threshold=0.5, threshold=0.4, S=7, C=20):
    model.eval()
    all_pred, all_true = [], []
    img_idx = 0
    for i in range(0, len(X), batch):
        out = model(prep(X[i:i+batch]))
        pred_boxes = cellboxes_to_boxes(out, S, C)
        true_boxes = decode_targets(T[i:i+batch], S, C)
        for b in range(len(pred_boxes)):
            nms = non_max_suppression(pred_boxes[b], iou_threshold, threshold)
            for box in nms:
                all_pred.append([img_idx] + box)
            for box in true_boxes[b]:
                all_true.append([img_idx] + box)
            img_idx += 1
    return all_pred, all_true


@torch.no_grad()
def draw_detections(model, X, prep, device, indices, threshold=0.4, iou_threshold=0.5, S=7, C=20):
    model.eval()
    fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 5))
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        img = X[idx]                              # uint8 CHW
        out = model(prep(X[idx:idx+1]))
        boxes = non_max_suppression(cellboxes_to_boxes(out, S, C)[0], iou_threshold, threshold)
        ax.imshow(img.permute(1, 2, 0).cpu().numpy())
        H = W = img.shape[1]
        for cls, conf, x, y, w, h in boxes:
            x1, y1 = (x - w / 2) * W, (y - h / 2) * H
            ax.add_patch(patches.Rectangle((x1, y1), w * W, h * H, fill=False,
                                           edgecolor="lime", linewidth=2))
            ax.text(x1, max(y1 - 4, 0), f"{VOC_CLASSES[int(cls)]} {conf:.2f}",
                    color="black", fontsize=8, backgroundcolor="lime")
        ax.axis("off")
    plt.tight_layout()
    return fig
