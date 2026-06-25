"""YOLO-v1 loss function (the core implementation).

Implements the multi-part YOLO loss: localisation loss for the box responsible for each object
(the predictor with the highest IoU), an object-confidence loss, a no-object confidence loss, and a
classification loss, weighted by lambda_coord and lambda_noobj as in the original paper.
Predictions and targets have shape (N, S, S, B*5 + C) with each box encoded as (conf, x, y, w, h).
"""
import torch
import torch.nn as nn


def _iou(boxes_a, boxes_b):
    """IoU between two sets of boxes given as (x, y, w, h) with centre coordinates."""
    ax1, ay1 = boxes_a[..., 0] - boxes_a[..., 2] / 2, boxes_a[..., 1] - boxes_a[..., 3] / 2
    ax2, ay2 = boxes_a[..., 0] + boxes_a[..., 2] / 2, boxes_a[..., 1] + boxes_a[..., 3] / 2
    bx1, by1 = boxes_b[..., 0] - boxes_b[..., 2] / 2, boxes_b[..., 1] - boxes_b[..., 3] / 2
    bx2, by2 = boxes_b[..., 0] + boxes_b[..., 2] / 2, boxes_b[..., 1] + boxes_b[..., 3] / 2
    inter_w = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp(0)
    inter_h = (torch.min(ay2, by2) - torch.max(ay1, by1)).clamp(0)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1).clamp(0) * (ay2 - ay1).clamp(0)
    area_b = (bx2 - bx1).clamp(0) * (by2 - by1).clamp(0)
    return inter / (area_a + area_b - inter + 1e-6)


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.lc, self.ln = lambda_coord, lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred, target):
        S, B, C = self.S, self.B, self.C
        N = pred.size(0)
        # split predictions: class scores, then B boxes of (conf, x, y, w, h)
        pred_cls = pred[..., :C]
        pred_boxes = pred[..., C:].reshape(N, S, S, B, 5)
        tgt_cls = target[..., :C]
        tgt_box = target[..., C:C + 5]              # single ground-truth box per cell
        obj = target[..., C].unsqueeze(-1)           # 1 if an object is in the cell

        # IoU of each predicted box with the ground-truth box -> responsible predictor
        gt_xywh = tgt_box[..., 1:5].unsqueeze(3)     # (N,S,S,1,4)
        ious = _iou(pred_boxes[..., 1:5], gt_xywh)   # (N,S,S,B)
        best = ious.argmax(-1, keepdim=True)         # (N,S,S,1)
        best_box = torch.gather(pred_boxes, 3, best.unsqueeze(-1).expand(N, S, S, 1, 5)).squeeze(3)

        objm = obj.squeeze(-1).bool()

        # --- localisation loss (only for cells with an object, only the responsible box) ---
        pxy, pwh, pconf = best_box[..., 1:3], best_box[..., 3:5], best_box[..., 0]
        txy, twh = tgt_box[..., 1:3], tgt_box[..., 3:5]
        coord_loss = self.mse(pxy[objm], txy[objm])
        coord_loss += self.mse(torch.sign(pwh[objm]) * torch.sqrt(pwh[objm].abs() + 1e-6),
                               torch.sqrt(twh[objm] + 1e-6))

        # --- object confidence loss (target = IoU of the responsible box) ---
        best_iou = torch.gather(ious, 3, best).squeeze(3)
        obj_loss = self.mse(pconf[objm], best_iou[objm].detach())

        # --- no-object confidence loss (both predictors in empty cells) ---
        noobj = ~objm
        noobj_loss = self.mse(pred_boxes[..., 0][noobj], torch.zeros_like(pred_boxes[..., 0][noobj]))

        # --- classification loss (cells with an object) ---
        cls_loss = self.mse(pred_cls[objm], tgt_cls[objm])

        total = self.lc * coord_loss + obj_loss + self.ln * noobj_loss + cls_loss
        return total / N
