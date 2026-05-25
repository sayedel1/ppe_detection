# ppe_detection
I developed a Personal Protective Equipment (PPE) Detection System using YOLOv8 for real-time worker safety monitoring.
# PPE Detection System — YOLOv8

Real-time Personal Protective Equipment detection using a custom-trained YOLOv8 model with a live OpenCV dashboard.

---

## What it detects

| Class | Description |
|-------|-------------|
| Cap | Safety cap |
| Gloves | Protective gloves |
| Googles | Safety goggles |
| White_C | White coat |
| No-Cap | Missing cap (violation) |
| Person | Person detected |

---

## Tech Stack

- **YOLOv8** — Object detection model
- **Roboflow** — Dataset labeling & export
- **Google Colab** — Model training (free Tesla T4 GPU)
- **OpenCV** — Real-time video processing & dashboard
- **Python** — Core language

---

## Model Results

Trained on 201 validation images across 6 classes:

| Class | mAP50 | Precision | Recall |
|-------|-------|-----------|--------|
| All | 0.697 | 0.636 | 0.724 |
| Cap | 0.710 | 0.658 | 0.822 |
| Gloves | 0.652 | 0.668 | 0.655 |
| Googles | 0.857 | 0.745 | 0.841 |
| No-Cap | 0.373 | 0.423 | 0.385 |
| Person | 0.808 | 0.574 | 0.863 |
| White_C | 0.782 | 0.747 | 0.778 |

> Training stopped at epoch 47/50 (EarlyStopping, patience=10).
> Total training time: ~20 minutes on free Colab T4 GPU.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# Webcam
python ppe.py

# Video file
python ppe.py --source ccp.mp4

# Image
python ppe.py --source image.jpg
```

---

## Project Structure

```
ppe/
├── ppe.py              # Main detection script
├── best.pt             # Trained YOLOv8 model weights
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## Dashboard

The system displays a real-time side panel showing:
- Number of workers with PPE
- Number of workers without PPE
- Total persons detected
- Live timestamp
- Overall status: ALL SAFE or ALERT

---

## Workflow

1. **Dataset** — Collected & labeled via Roboflow
2. **Training** — Fine-tuned YOLOv8n on Google Colab (free GPU)
3. **Export** — Downloaded `best.pt` weights
4. **Inference** — Real-time detection on video with OpenCV dashboard

---

## Notes

- Make sure `best.pt` is in the same folder as `ppe.py`
- For better No-Cap detection, more violation images are needed in the dataset
- Tested on Windows with OpenCV 4.5.5.64
-
