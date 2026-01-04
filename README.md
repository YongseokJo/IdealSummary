# DeepSets regression example

This small project implements a DeepSets model for regression using PyTorch.

- Model: `src/deepset.py`
- Training example: `src/train.py` (synthetic dataset)

## Quick start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the training example:

```bash
python src/train.py --epochs 20
```

Files:

- [src/deepset.py](src/deepset.py) - DeepSet model implementation
- [src/train.py](src/train.py) - Training script and synthetic dataset
