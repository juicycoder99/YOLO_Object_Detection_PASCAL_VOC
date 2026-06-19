"""PASCAL VOC 2007 detection data for the YOLO Final Exam.

Each image is resized to 448x448 (uint8). Each ground-truth box is encoded into an S x S x (C+5)
target grid: C class one-hots followed by one box (conf, x, y, w, h), where x,y are relative to the
responsible cell and w,h are relative to the whole image (YOLO-v1 encoding). Cached to .pt.
"""
import os, torch, torchvision
import torchvision.transforms.functional as TF

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
C2I = {c: i for i, c in enumerate(VOC_CLASSES)}


def build_split(image_set, root="./data", size=448, S=7, C=20):
    cache = os.path.join(root, f"voc2007_yolo_{image_set}_{size}_{S}.pt")
    if os.path.exists(cache):
        d = torch.load(cache); return d["X"], d["T"]
    ds = torchvision.datasets.VOCDetection(root, year="2007", image_set=image_set, download=True)
    imgs, targets = [], []
    for img, ann in ds:
        W = float(ann["annotation"]["size"]["width"]); H = float(ann["annotation"]["size"]["height"])
        objs = ann["annotation"]["object"]
        objs = objs if isinstance(objs, list) else [objs]
        t = torch.zeros(S, S, C + 5)
        for o in objs:
            bb = o["bndbox"]
            cx = (float(bb["xmin"]) + float(bb["xmax"])) / 2 / W   # normalised centre
            cy = (float(bb["ymin"]) + float(bb["ymax"])) / 2 / H
            bw = (float(bb["xmax"]) - float(bb["xmin"])) / W        # normalised size
            bh = (float(bb["ymax"]) - float(bb["ymin"])) / H
            i, j = min(int(S * cy), S - 1), min(int(S * cx), S - 1)
            if t[i, j, C] == 0:                                     # one object per cell (YOLO-v1)
                t[i, j, C] = 1
                t[i, j, C + 1] = S * cx - j
                t[i, j, C + 2] = S * cy - i
                t[i, j, C + 3] = bw
                t[i, j, C + 4] = bh
                t[i, j, C2I[o["name"]]] = 1
        imgs.append(TF.pil_to_tensor(TF.resize(img, [size, size])))
        targets.append(t)
    X = torch.stack(imgs); T = torch.stack(targets)
    torch.save({"X": X, "T": T}, cache)
    return X, T


if __name__ == "__main__":
    for split in ["trainval", "test"]:
        X, T = build_split(split)
        print(f"{split}: X={tuple(X.shape)} ({X.dtype}), T={tuple(T.shape)}, "
              f"cells with objects={int((T[..., 20] == 1).sum())}")
