# 🦷 ToothFairy SOTA Inferior Alveolar Nerve (IAN) Segmentation

State-of-the-Art Deep Learning Framework for automated, continuous 3D segmentation of the **Inferior Alveolar Canal / Nerve (IAN)** in CBCT volumes.

Target Benchmark: **MICCAI ToothFairy Challenge**
- **Champion Leaderboard Score**: **Dice: 0.7956 | HD95: 4.4906 mm**
- **Our Target / Design Goal**: **Dice > 0.8350 | HD95 < 2.8000 mm**

---

## 🌟 Core Breakthroughs & Architectural Highlights

1. **Topology-Preserving Compound Loss**:
   $$\mathcal{L}_{total} = 0.40 \mathcal{L}_{CE} + 0.30 \mathcal{L}_{Dice} + 0.20 \mathcal{L}_{clDice} + 0.10 \mathcal{L}_{Boundary}$$
   - **`clDice` (Centerline Dice)** via differentiable 3D soft skeletonization prevents topological breaks along the tubular nerve trunk.
   - **Boundary Loss** directly minimizes surface distance errors, slashing HD95.

2. **Bilateral 3-Class Anatomical Formulation**:
   - Class 0: Background
   - Class 1: Right IAN Canal
   - Class 2: Left IAN Canal
   - Enforces hemispheric priors, preventing midline cross-talk and ensuring exactly one continuous component per side.

3. **Geodesic Sparse-to-Dense Tracking**:
   - The 2023 champion discarded the 290 sparse scans (`gt_sparse`).
   - We utilize `gt_sparse` seed sequences as geometric anchors with Dijkstra minimal-path tracking on model probability cost maps to reconstruct continuous pseudo-dense labels, multiplying the effective training set size by $3\times$.

4. **3D Residual U-Net Backbone (`ResEncoderUNet3D`)**:
   - Deep residual paths with Instance Normalization and multi-scale Deep Supervision.
   - Optimized for RTX 4060 (8GB VRAM) via PyTorch AMP (FP16) and gradient accumulation.

5. **Anatomical Post-Processing**:
   - Hemispheric splitting $\rightarrow$ Single largest 3D connected component retention $\rightarrow$ Micro-gap morphological bridging $\rightarrow$ High-anatomical outlier suppression.

---

## 🚀 Execution Workflow

All scripts are configured to use the `toothfairy` Conda environment:
`/data/miniconda3/envs/toothfairy/bin/python`

### 1. Preprocess and Cache Dataset
Preprocesses all 153 dense scans and 290 sparse scans (intensity windowing `[-500, 2500]` HU, z-score normalization, and bilateral label generation):
```bash
/data/miniconda3/envs/toothfairy/bin/python scripts/01_prepare_data.py
```

### 2. Train Teacher Models (5-Fold Cross-Validation)
Trains the initial 3D Residual U-Net with Compound Topology Loss on the clean dense ground-truth cases:
```bash
# Train Fold 0
/data/miniconda3/envs/toothfairy/bin/python scripts/02_train_teacher.py --fold 0 --epochs 300

# Train other folds (1 to 4) for full 5-fold ensemble
/data/miniconda3/envs/toothfairy/bin/python scripts/02_train_teacher.py --fold 1 --epochs 300
```

### 3. Generate Geodesic Pseudo-Labels on 290 Sparse Scans
Traces shortest path geodesics through probability cost maps anchored on `gt_sparse` seed points:
```bash
/data/miniconda3/envs/toothfairy/bin/python scripts/03_generate_pseudo_labels.py
```

### 4. Retrain Final Student Model
Retrains the student model on the combined dataset (130 clean dense + 200+ geodesic pseudo-labeled cases) with Strong Data Augmentation:
```bash
/data/miniconda3/envs/toothfairy/bin/python scripts/04_train_student.py --epochs 350
```

### 5. Benchmark Evaluation & Test-Time Augmentation (TTA)
Evaluates the multi-model ensemble with 3D TTA (Axial, Coronal, Sagittal flips with bilateral class swap) and anatomical post-processing against the official ToothFairy leaderboard metrics:
```bash
/data/miniconda3/envs/toothfairy/bin/python scripts/05_evaluate_and_infer.py
```

---

## 📁 Repository Structure

```
.
├── configs/
│   └── default_config.py           # Paths, hyperparameters, loss weights
├── src/
│   ├── dataset/
│   │   ├── toothfairy_dataset.py   # 3D Patch Dataset with foreground oversampling
│   │   ├── transforms.py           # 3D spatial and intensity augmentations
│   │   └── geodesic_tracker.py     # Dijkstra minimal path tracking on gt_sparse
│   ├── models/
│   │   ├── resnet_unet3d.py        # 3D Residual U-Net with deep supervision
│   │   ├── soft_skeleton.py        # Differentiable 3D soft skeletonization
│   │   └── losses.py               # Compound CE + Dice + clDice + Boundary loss
│   ├── postprocessing/
│   │   └── anatomical_filter.py    # Hemispheric split, single CC, gap bridging
│   ├── training/
│   │   └── trainer.py              # PyTorch AMP trainer with sliding-window validation
│   └── evaluation/
│       └── metrics.py              # Official ToothFairy Dice & HD95 (mm) evaluators
├── scripts/
│   ├── 01_prepare_data.py          # Preprocessing & bilateral caching
│   ├── 02_train_teacher.py         # 5-fold Teacher training
│   ├── 03_generate_pseudo_labels.py# Geodesic pseudo-labeling
│   ├── 04_train_student.py         # Student retraining on combined data
│   └── 05_evaluate_and_infer.py    # Benchmark evaluation & TTA inference
└── README.md
```
