# YOLO Object Detection on PASCAL VOC 2007

A YOLO-v1 style object detector trained from scratch (head) on PASCAL VOC 2007, with a ResNet-50
backbone pretrained on ImageNet. The implementation centres on the **YOLO loss**
([`yolo_loss.py`](yolo_loss.py)); the full pipeline (target encoding, decoding, NMS, mAP) is also
implemented. The training notebook is [`yolo_object_detection.ipynb`](yolo_object_detection.ipynb).

## Method

- 7×7 grid, 2 boxes/cell, 20 classes → `S×S×30` predictions, sigmoid-activated so all outputs are in `[0,1]`.
- ResNet-50 backbone + a small convolutional head; images at 448×448.
- YOLO loss: localisation (responsible box by IoU), object-confidence, no-object confidence, and
  classification terms, with `λ_coord = 5`, `λ_noobj = 0.5`.
- Trained 40 epochs, Adam (lr 1e-4) with cosine schedule and gradient clipping, on an RTX 3080.

## Results

| Epoch | Train loss | Test loss | mAP@0.5 |
|------:|-----------:|----------:|--------:|
| 10 | 0.67 | 2.55 | 0.270 |
| 20 | 0.21 | 2.53 | 0.293 |
| 30 | 0.11 | 2.57 | 0.311 |
| 40 | 0.08 | 2.66 | **0.324** |

Final test loss **2.655**, final **mAP@0.5 = 0.324** on the VOC2007 test set. The detector localises
common objects (person, car, cat, dog, …) well; the usual YOLO-v1 limitations show on small or
clustered objects, since each grid cell commits to a single object. With more epochs and a finer
schedule the mAP would climb further toward the ~0.5 range quoted in the original paper; the figure
here is the honest result of 40 epochs on a single GPU.

## Code

| File | Description |
|------|-------------|
| `yolo_model.py` | YOLO-v1 detector (ResNet-50 backbone + head) |
| `yolo_loss.py` | The YOLO multi-part loss (core implementation) |
| `yolo_data.py` | VOC2007 → 448px images + 7×7 target grids |
| `yolo_utils.py` | IoU, non-max suppression, mAP, cell decoding |
| `yolo_eval.py` | mAP gathering and sample-detection drawing |
| `yolo_object_detection.ipynb` | Training, evaluation, sample detections |
| `PROJECT_BRIEF.pdf` | Project brief (goals, objectives, outcomes) |

## Running it

```bash
pip install torch torchvision numpy matplotlib   # CUDA build of torch for GPU
jupyter notebook yolo_object_detection.ipynb     # VOC2007 downloaded automatically
```
