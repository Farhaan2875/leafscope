# LeafScope — Automated Plant Leaf Disease Diagnosis and Severity Classification

**CSE471 · Deep Learning for Computer Vision · Section K4P24MR**

An end-to-end pipeline that diagnoses crop leaf diseases from a photo, estimates how badly the leaf
is damaged, and shows *where* each network looked. It compares **ResNet-50** against
**MobileNetV2** on the real **PlantVillage** dataset (38 crop/disease classes, ~54,300 photos),
and ships with a hand-built local web interface instead of a stock "AI app" template.

| Project brief | Where it lives |
|---|---|
| Deep CNN disease classification | `models.py`, `train.py` |
| Data augmentation for real-world lighting | `data.py` (`train_transform`, `LIGHTING_VARIANTS`) |
| ResNet-50 vs MobileNetV2 | `train.py`, `evaluate.py` → `outputs/comparison.json` |
| Grad-CAM on lesions, not background | `gradcam.py`, `severity.attention_split` |
| Accuracy and F1 | `metrics.py` (plain NumPy), `evaluate.py` |
| Severity classification | `severity.py` (OpenCV) |
| PyTorch, free Colab T4 or laptop CPU | everything runs on either |

See **EXPLANATION.md** for a plain-language explanation of every part of the project and likely viva questions.

## Technology (syllabus only)

PyTorch + torchvision (pretrained ResNet-50 / MobileNetV2, fine-tuning, Grad-CAM hooks), OpenCV (image I/O,
colour spaces, resizing, normalisation, morphology, lighting perturbations), NumPy (metrics) and Matplotlib
(training curves, confusion matrices, comparison figures). The interface uses only Python's standard library
`http.server` plus plain HTML/CSS/JavaScript, with no Streamlit, Gradio, Flask or other extra packages.
No scikit-learn, TensorFlow or other libraries are used.

How this maps to the course syllabus (CSE471):

- **Unit I**: colour spaces, resizing, normalisation, loss functions, optimisers (`data.py`, `severity.py`, `train.py`).
- **Unit II**: ResNet, batch normalisation, dropout, data augmentation, learning-rate scheduling, early stopping.
- **Unit III**: transfer learning with ResNet and MobileNet, feature extraction, freezing/unfreezing layers,
  layer-wise learning rates (`models.py`, `train.py`).
- **Unit VI**: Grad-CAM interpretability (`gradcam.py`).
- **Practicals**: 1 (OpenCV preprocessing and Matplotlib), 5 (pretrained features on a new dataset),
  6 (fine-tuning MobileNet with freezing/unfreezing) and 10 (Grad-CAM on correct vs misclassified samples).

Severity uses classical OpenCV segmentation (colour thresholds, morphology), not a learned segmentation
network such as UNet (Unit V).

## Setup

Requires Python 3.9+ and `git` (git is only used to download the dataset).

```bash
cd leafscope
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

On a machine with an NVIDIA GPU, install the CUDA build of PyTorch from <https://pytorch.org/get-started/locally/>
first, then `pip install -r requirements.txt` for the rest.

## 1 · Get the data

```bash
python prepare_data.py --download
```

This sparse-clones only the `raw/color` folder of the public PlantVillage repository
(<https://github.com/spMohanty/PlantVillage-Dataset>, roughly 1 GB) into `data/`, then writes a
reproducible stratified **70 / 15 / 15** train/val/test split to `data/splits.json`.

Already have the dataset (for example from Kaggle)? Point to the folder that contains one sub-folder per class:

```bash
python prepare_data.py --data-dir "D:/datasets/PlantVillage/color"
```

**Laptop CPU?** Use a small subset so training finishes in a sensible time:

```bash
python prepare_data.py --download --max-per-class 150
```

## 2 · Train both models

```bash
python train.py --model resnet50
python train.py --model mobilenet_v2
```

Each run does two-phase transfer learning from ImageNet weights:

1. **Head only** (backbone frozen, 3 epochs) → 2. **Fine-tune** (backbone unfrozen, up to 12 epochs) with
layer-wise learning rates, a one-cycle LR schedule, label smoothing, and early stopping on validation macro-F1.

The best checkpoint goes to `outputs/checkpoints/<model>.pt` and curves to `outputs/figures/training_<model>.png`.

| Flag | Meaning |
|---|---|
| `--quick` | 1 head epoch + 2 fine-tune epochs at 128 px. Use it first to check everything works. |
| `--batch-size 32` | Lower it if you run out of GPU memory. |
| `--workers 0` | Use this if data loading hangs on Windows. |
| `--epochs-head N --epochs-ft N` | Training length. |

Rough guide: on a free Colab T4 the full dataset trains in the order of tens of minutes per model;
on a laptop CPU use `--max-per-class` and `--quick`. Exact times depend on your hardware, and the
evaluation step records the real numbers.

## 3 · Evaluate and compare

```bash
python evaluate.py
```

Writes, to `outputs/`: `comparison.json` / `comparison.csv` (accuracy, macro-F1, weighted-F1, parameters,
size, latency, training time), per-class precision/recall/F1 CSVs, confusion matrices, a
lighting-robustness test (dim, over-exposed, hard shadow, colour cast, applied with OpenCV), and a
**Grad-CAM study** that measures how much attention falls on lesions vs healthy tissue vs background,
plus a gallery of correct and misclassified leaves. Add `--robust-n 300 --cam-n 60` for a faster CPU run.

## 4 · Run the application

```bash
python app/server.py
```

Open <http://127.0.0.1:8000>.

- **The Bench**: drop, paste or click to add a leaf photo, or open a *specimen* — a real held-out test
  photograph from the drawer. Both examiners give a verdict with confidence, the OpenCV pass reports
  severity (lesion area ÷ leaf area, graded 0–4), and a draggable curtain reveals each network's Grad-CAM
  attention or the lesion map over the photo. For drawer specimens you can break the seal to compare with
  the dataset's own label.
- **Trials**: the ResNet-50 vs MobileNetV2 comparison, lighting robustness and figures from `evaluate.py`.

If no model is trained yet, the app still starts and runs the OpenCV severity measurement; the header lamp
shows which examiners are loaded. Use `--port 8080` if 8000 is taken.

### Google Colab

Upload the `leafscope` folder, then in a GPU runtime (Runtime → Change runtime type → T4):

```python
%cd leafscope
!pip install -q opencv-python-headless matplotlib
!python prepare_data.py --download
!python train.py --model resnet50
!python train.py --model mobilenet_v2
!python evaluate.py
```

Download `outputs/` and run `python app/server.py` locally, or just use the figures and tables.

## Honest limitations

- **Accuracy will look very high, and that overstates field performance.** PlantVillage photos were taken
  under controlled conditions, and the same leaf is often photographed several times. A random split can
  therefore put near-duplicates in both train and test (you can see this in the specimen drawer). Treat the
  reported numbers as performance on PlantVillage, not on a farmer's phone. The lighting tests are a partial
  check, not a substitute for testing on in-field photos.
- **Severity is an estimate, not a label.** PlantVillage has no severity annotations, so there is nothing to
  train or score a severity model against. `severity.py` segments the leaf and counts tissue whose colour departs
  from the leaf's own healthy green, then bins the percentage into five grades (<1, 1–10, 10–25, 25–50, >50 %).
  On healthy leaves it flags a few percent (veins, glare, texture), and **large grey/dead patches can be
  under-counted** because they resemble the background. Use it as a screening aid.
- **Classes are limited to the 38 in PlantVillage.** An unrelated plant or a crop outside the list will still
  be forced into one of them.
- The management notes are deliberately generic. Check treatment decisions with a local agricultural extension service.
- This submission was developed in an environment where PyTorch could not be installed, so the training and
  evaluation scripts have **not been run end to end** by the author. Dataset preparation, the OpenCV severity
  pipeline and the web server were run and tested on real PlantVillage images; please run `train.py --quick`
  first to confirm your setup before the full training.

## Troubleshooting

| Problem | Fix |
|---|---|
| `git failed` in `prepare_data.py` | Install git, or download PlantVillage manually and use `--data-dir`. |
| `data/splits.json not found` | Run `prepare_data.py` first. |
| CUDA out of memory | `--batch-size 32` (or 16). |
| Data loader hangs (Windows) | `--workers 0`. |
| App shows "No examiners trained yet" | Train both models; check `outputs/checkpoints/`. |
| Port already in use | `python app/server.py --port 8080`. |

## Layout

```
leafscope/
├── prepare_data.py   download + stratified split
├── data.py           dataset, augmentation, lighting perturbations
├── models.py         ResNet-50 / MobileNetV2, freezing, layer-wise LR
├── train.py          two-phase transfer learning
├── evaluate.py       metrics, comparison, robustness, Grad-CAM study
├── gradcam.py        Grad-CAM from scratch (hooks)
├── severity.py       OpenCV leaf segmentation + lesion/severity estimate
├── inference.py      checkpoint loading + single-image prediction
├── knowledge.py      names, pathogens, generic notes per class
├── metrics.py        accuracy / confusion matrix / P-R-F1 in NumPy
└── app/              server.py + static/ (index.html, style.css, app.js)
```

## Dataset citation

Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the
development of mobile disease diagnostics.* arXiv:1511.08060. Mohanty, S. P., Hughes, D. P., & Salathé, M.
(2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science.
