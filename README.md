# FFT-based Permeability Prediction (PISA-FFT)

This repository provides a **clean, modular** implementation of an FFT-based permeability
prediction model with **physics-aware patch embeddings**.

The code is structured as a small Python package:
- `permeability_fft/data.py`: dataset and dataloader utilities
- `permeability_fft/models.py`: model definitions
- `permeability_fft/utils.py`: helper functions
- `permeability_fft/train.py`: training / validation / testing script

> **Note:** This is an external, simplified version of the original research codebase.
> It is intended to help readers understand the architecture and basic training pipeline,
> rather than to exactly reproduce all experimental results.

---

## 1. Installation

Create a Python environment (Python 3.8+ recommended) and install the dependencies:

```bash
pip install -r requirements.txt
```

The most important packages are:
- `torch`, `torchvision`
- `albumentations`, `opencv-python`, `scikit-image`
- `numpy`, `pandas`, `tqdm`, `scikit-learn`

You may choose any compatible PyTorch + CUDA version for your hardware.

---

## 2. Data Format

The dataset is assumed to consist of:
- A folder of PNG images containing the porous media microstructure.
- A CSV file with two columns:
  1. Image ID (file name without extension)
  2. Permeability value (a positive scalar)

The `PermeabilityDataset` class will:
- Load each image as RGB,
- Resize and normalize it using Albumentations,
- Compute patch-level physics features (porosity & morphology descriptors)
  using Otsu thresholding and connected-component analysis,
- Apply a log-transform `log(1 + k)` to the permeability value.

---

## 3. Training

You can launch training via:

```bash
python -m permeability_fft.train \
  --image-dir /path/to/images \
  --csv-file /path/to/permeability.csv \
  --output-dir ./training_output/fft_external
```

Commonly used arguments:

- `--img-size`: input resolution (default: 224)
- `--patch-size`: patch size for physics features and patch embedding (default: 16)
- `--batch-size`: training batch size (default: 16)
- `--num-epochs`: number of training epochs (default: 80)
- `--embed-dim`, `--num-heads`, `--depth`: transformer encoder size
- `--lr`, `--weight-decay`: optimizer hyperparameters
- `--seed`: optional random seed (if omitted, the run is non-deterministic)

The training script will:
1. Randomly split the dataset into train / validation / test sets (60% / 20% / 20%),
2. Train the model and monitor validation R²,
3. Save the model with the best validation R² to `best_model.pth`,
4. Evaluate it on the held-out test set (MSE, RMSE, R², MedARE),
5. Save the training log and test metrics into the output directory.

---

## Dataset
DeePore dataset on Zenodo:  
https://zenodo.org/records/4297035

Eleven Sandstones micro-CT dataset:
https://github.com/LukasMosser/digital_rocks_data

Seven Fontainebleau sandstone specimens
https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-57
