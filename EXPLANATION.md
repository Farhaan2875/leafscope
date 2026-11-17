# LeafScope: Project Explanation (for report and viva)

## 1. What the project does
Given a photo of a crop leaf, LeafScope does three things:
1. **Diagnoses** the crop and disease (38 classes, for example "Tomato - Early blight").
2. **Estimates severity**: what percentage of the leaf is damaged, graded 0 to 4.
3. **Explains itself** with Grad-CAM: a heat map of where the network looked.

Two networks, **ResNet-50** and **MobileNetV2**, are trained on the same data and compared.

## 2. Data
- **PlantVillage** (Hughes and Salathe, 2015): about 54,300 labelled leaf photos, 38 classes. We use the `raw/color` folder, which holds the original, non-augmented photos.
- **Split**: stratified 70% train, 15% validation, 15% test, with a fixed random seed (42), so the result is reproducible. Stratified means every class keeps the same proportions in each part.
- Validation is used to choose the best epoch. The **test set is used only once**, at the end.

## 3. Pipeline, file by file
| Step | File | What happens |
|---|---|---|
| Download and split | `prepare_data.py` | Sparse git clone of the dataset, writes `data/splits.json` |
| Load and augment | `data.py` | Dataset class, augmentations, lighting perturbations |
| Models | `models.py` | ResNet-50 / MobileNetV2 with a new classification head, freezing helpers, layer-wise learning rates |
| Training | `train.py` | Two-phase transfer learning, early stopping, saves the best checkpoint |
| Evaluation | `evaluate.py` | Accuracy, F1, confusion matrix, lighting test, Grad-CAM study |
| Grad-CAM | `gradcam.py` | Implemented from scratch with PyTorch hooks |
| Severity | `severity.py` | OpenCV leaf segmentation and lesion measurement |
| Metrics | `metrics.py` | Accuracy, precision, recall, F1 in plain NumPy |
| App | `app/server.py` + `app/static/` | Local web interface |

## 4. Key ideas, explained simply

### Transfer learning (Unit III)
Both networks start from **ImageNet weights**. Early layers already detect edges and textures, which also help on leaves. Training happens in two phases:
1. **Head only** (3 epochs): the backbone is frozen and only the new final layer is trained. Frozen BatchNorm layers stay in eval mode so their statistics do not drift.
2. **Fine-tune** (up to 12 epochs): everything is unfrozen with **layer-wise learning rates**. The block nearest the output gets the highest rate (0.0002) and each earlier block is multiplied by 0.6, because early layers hold general features that should change slowly.

### Training techniques (Unit II)
- **Loss**: cross-entropy with label smoothing 0.1 (stops over-confident predictions).
- **Optimiser**: AdamW (Adam with proper weight decay).
- **Schedule**: one-cycle learning rate (warm up, then decay).
- **Regularisation**: dropout 0.3 before the final layer, weight decay, data augmentation.
- **Early stopping**: stop if validation macro-F1 does not improve for 4 epochs; the best checkpoint is kept.

### Augmentation for real-world lighting
Random crop and zoom, horizontal and vertical flips, rotation up to 25 degrees, strong brightness / contrast / saturation jitter, and occasional blur. After training, `evaluate.py` also tests the model on test images that were made darker, over-exposed, half-shadowed or colour-cast with OpenCV.

### ResNet-50 vs MobileNetV2
- **ResNet-50**: about 25.6 million parameters. Uses **skip connections** (output = layers(x) + x), which let gradients flow in very deep networks and avoid the vanishing-gradient problem. Its building block is the bottleneck (1x1, 3x3, 1x1 convolutions).
- **MobileNetV2**: about 3.5 million parameters. Uses **depthwise separable convolutions** (a per-channel filter, then a 1x1 mix), which cost far less computation, and **inverted residuals** with linear bottlenecks. Built for phones.
- Expected trade-off: ResNet-50 is usually a bit more accurate; MobileNetV2 is much smaller and faster. The real numbers for your run are on the Trials tab.

### Metrics
- **Accuracy**: fraction correct. Can hide weak classes.
- **Precision / recall / F1** per class; **macro-F1** averages them equally over classes, so rare classes count as much as common ones. This is why training watches macro-F1.
- **Confusion matrix**: row = true class, column = predicted class. Off-diagonal cells show which diseases get mixed up.

### Grad-CAM (Unit VI)
1. Run the image forward and take the score of the predicted class.
2. Back-propagate that score to the **last convolutional block**.
3. Average the gradients over each feature map to get a weight per channel.
4. Weighted sum of the feature maps, ReLU, resize to the image: the heat map.

The code also measures how much of the heat falls on **lesions, healthy leaf and background**. A good model should look at lesions, not at the background.

### Severity (classical OpenCV, not deep learning)
PlantVillage has no severity labels, so a learned severity model cannot be trained or checked. Instead:
1. **Segment the leaf**: leaf tissue has colour (chroma) while soil, shadow and paper are nearly grey; Otsu thresholding plus morphology cleans the mask.
2. **Find the healthy-green reference** colour of that leaf in the **Lab** colour space.
3. **Mark lesion pixels**: tissue whose colour is far from that reference, or very dark, or less green.
4. **Severity % = lesion pixels / leaf pixels**, graded: below 1% Trace (0), 1-10% Mild (1), 10-25% Moderate (2), 25-50% Severe (3), above 50% Critical (4).

It is a screening estimate. Healthy leaves still show a few percent (veins, glare), and large grey dead patches can be under-counted.

## 5. The application
Run `python app/server.py` and open http://127.0.0.1:8000.
- **The Bench**: add a leaf photo or open a held-out test photo from the drawer. Both models give a verdict with confidence. A draggable curtain reveals the Grad-CAM or lesion map over the photo. Severity is shown on a ruler. For drawer photos you can break the seal to compare with the dataset label.
- **Trials**: the ResNet-50 vs MobileNetV2 table, lighting robustness bars and figures.

## 6. Limitations (say these confidently in the viva)
1. **PlantVillage is easy.** Photos are lab-style with plain backgrounds, and the same leaf appears several times, so a random split puts near-duplicates in train and test. High accuracy does not mean the model will work on a farmer's phone.
2. **Severity is an image-based estimate**, not a trained or pathologist-verified score.
3. **Only 38 classes**: any other plant is forced into one of them.
4. **Management advice is generic** on purpose.

**Improvements**: train on field images, split by leaf or plant instead of by photo, add expert severity labels, and use a segmentation network such as UNet (Unit V) for lesions.

## 7. Likely viva questions with short answers
- **Why transfer learning?** Small domain-specific data, and ImageNet features (edges, textures) transfer. It trains faster and generalises better than training from scratch.
- **Why freeze first, then unfreeze?** A new random head would send large gradients back and damage good pretrained weights. Train the head first, then fine-tune gently.
- **Why layer-wise learning rates?** Early layers hold general features and need small changes; later layers are task-specific.
- **Why macro-F1 instead of accuracy?** Classes are unequal in size; macro-F1 treats each class equally.
- **What does a skip connection do?** It adds the input to the block output, so gradients have a short path and very deep networks can train.
- **Why is MobileNetV2 lighter?** Depthwise separable convolutions need far fewer multiplications than normal convolutions.
- **What does Grad-CAM show?** The regions that increased the predicted class score, from gradients at the last convolutional layer.
- **Why is your accuracy so high?** The dataset is clean and has near-duplicates across splits; it overstates field performance.
- **Why OpenCV for severity?** There are no severity labels to train a network on.
- **Which model for a phone?** MobileNetV2, because of size and speed, if its accuracy is close to ResNet-50.
