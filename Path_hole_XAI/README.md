# Roadwatch: Road Image Classification + Grad-CAM

Classify an uploaded road image as **NORMAL ROAD** or **POTHOLE ROAD** with an ImageNet-pretrained ResNet18, then show Grad-CAM regions that contributed to that prediction.

## Features

- Two classes only: `0 = NORMAL ROAD`, `1 = POTHOLE ROAD`.
- ResNet18 transfer learning with class-balanced cross-entropy.
- Validation-matched resize and ImageNet normalization.
- Prediction probabilities and confidence (confidence is not accuracy).
- Grad-CAM from `model.layer4[-1]`, overlaid at the original image size.
- Validation precision, recall, F1, confusion matrix, and pothole-specific metrics.

## Structure

```text
app.py
requirements.txt
README.md
models/pothole_resnet18.pth
scripts/train.py
src/config.py
src/inference.py
src/gradcam.py
src/__init__.py
dataset/train/{normal,pothole}/
dataset/val/{normal,pothole}/
dataset/test/{normal,pothole}/
outputs/.gitkeep
```

## Setup

Use Python 3.10 or newer. On Windows, install the [Microsoft Visual C++ 2015–2022 Redistributable (x64)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) if PyTorch reports a missing DLL. Windows Application Control may require an administrator to approve PyTorch DLLs.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset

Place real images in these folders. `ImageFolder` order is checked against the one mapping in `src/config.py`.

```text
dataset/
  train/{normal,pothole}/
  val/{normal,pothole}/
  test/{normal,pothole}/
```

The included images come from Passos, B. T., Cassaniga, M. J., Fernandes, A. M. R., Medeiros, K. B., & Comunello, E. (2020), [Cracks and Potholes in Road Images](https://data.mendeley.com/datasets/t576ydh9v8/4), Mendeley Data V4, DOI [10.17632/t576ydh9v8.4](https://doi.org/10.17632/t576ydh9v8.4), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The 2,235 `RAW` road images use their paired `POTHOLE` masks for labels and were split 80/20 with seed 42. Training stops if identical image bytes appear in both splits.

**Label limitation:** this source selected images containing cracks and/or potholes. `normal` therefore means *no pothole marked in the mask*, not guaranteed defect-free pavement. These negatives include crack/rough-road examples, but this dataset does not provide a separate verified clean-road set. Real patched, shadowed, wet, and clean-road images should be added to the `normal` class before using the model beyond a baseline.

## Training

```powershell
python scripts/train.py --epochs 10 --patience 3
```

Training uses one moderate class-weighted cross-entropy strategy with class-frequency weights raised to the `0.75` power, up to 10 epochs, batch size 8, learning rate `0.0001`, `num_workers=0`, CPU thread limits when running on CPU, and ImageNet-pretrained ResNet18 features. ResNet18 `layer3`, `layer4`, and the two-class head are fine-tuned with AdamW. Moderate lighting, texture, and perspective augmentation is applied only to training images. ReduceLROnPlateau scheduling and early stopping are used; the best checkpoint by validation balanced accuracy is saved to `models/pothole_resnet18.pth`. Training uses a fixed seed for reproducibility.

## Run

```powershell
python -m streamlit run app.py
```

Upload a JPG, PNG, or WebP road image. The app displays the original, the predicted class confidence, and a **Grad-CAM Explanation**. The highlighted regions contributed to the model's prediction; Grad-CAM is not a segmentation mask or exact pothole boundary.

## Evaluation

Results from the resumed best-balanced-accuracy checkpoint (`325` validation images and `318` independent test images):

| Metric | Result |
| --- | ---: |
| Split | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Pothole precision | Pothole recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 74.15% | 70.15% | 69.03% | 70.15% | 69.50% | 54.81% | 60.64% |
| Test | 75.47% | 73.07% | 62.87% | 73.07% | 63.91% | 31.58% | 69.77% |

```text
Confusion matrix (actual rows, predicted columns)
                 NORMAL  POTHOLE
Actual NORMAL      184      47
Actual POTHOLE      37      57

False positives: 47    False negatives: 37
```

The independent test confusion matrix is `[[210, 65], [13, 30]]`: 65 false positives and 13 false negatives. Full per-image reports are saved in `outputs/validation_predictions.csv` and `outputs/test_predictions.csv`; error examples are copied to `outputs/false_positives/` and `outputs/false_negatives/`.

Four held-out image spot checks (actual labels are derived from the source pothole masks):

| Case | Actual | Prediction | Confidence | Correct |
| --- | --- | --- | ---: | --- |
| Clear normal road | NORMAL | NORMAL | 86.7% | Yes |
| Rough/patchy normal road | NORMAL | POTHOLE | 82.1% | No |
| Clear pothole road | POTHOLE | POTHOLE | 84.0% | Yes |
| Difficult pothole road | POTHOLE | NORMAL | 76.5% | No |

This is a baseline, not production-ready: pothole recall and precision need improvement with more representative labeled normal-road examples and an independent test set. Validation confidence is not calibrated accuracy.