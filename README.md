# Label Propagation

Code for the paper:

**Efficient Image Annotation via Semi-Supervised Object Segmentation with Label Propagation**
Vitalii Tutevych, Raphael Memmesheimer, Luca Eichler, Dmytro Pavlichenko, Fynn Schilke, Rodja Krudewig, Sven Behnke
https://arxiv.org/abs/2604.22992

![Examples](images/example.png)

## Dataset

The dataset used in the paper can be downloaded here:

**[Download dataset](https://TODO)**

Place the extracted contents under `datasets/` so each object class has its own subdirectory with `train/` and `valid/` splits.

## Setup

```bash
pip install -r requirements.txt
```

Place your YAML dataset recipes in `recipes/` and raw datasets in `datasets/`.
Set `THEIA_CACHE_DIR` to your local Theia model directory if not using the default (`~/.cache/theia/`).

## Usage

**Train labelers** (trains CLIP, Theia, and ViT Hopfield classifiers on representative samples):
```bash
python scripts/train_labelers.py
```

**Propagate labels** to a validation set and evaluate:
```bash
python scripts/validate_labelers.py
```

**Label raw datasets** using a detection model (requires `ultralytics`):
```bash
python scripts/segment_proposer.py --to_label_dir datasets/ --detector_dir /path/to/model
```
