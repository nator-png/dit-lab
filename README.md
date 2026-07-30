# DiT Lab

Small PyTorch experiments around Diffusion Transformers for image, audio, and video generation.

This repo now has two layers:

- `src/dit_lab/`: reusable, GitHub-ready code with configs, model modules, datasets, and a standard training loop.
- `experiments/`: original scratch scripts kept for reference and learning history.

## Project Layout

```text
.
├── src/dit_lab/          # Reusable package code
├── scripts/              # Command-line training entry points
├── experiments/          # Original exploratory scripts
├── notebooks/            # Jupyter notebooks
├── data/                 # Local datasets, ignored by git except placeholders
├── checkpoints/          # Model checkpoints, ignored by git
├── outputs/              # Generated samples, ignored by git
└── tests/                # Smoke tests
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

Install the PyTorch build that matches your CUDA version from the official PyTorch instructions if the default wheel is not right for your GPU.

## Data Format

For image training, put images in `data/images/` and create a captions file such as `data/captions.txt`:

```text
image_001.jpg a small robot standing under neon lights
image_002.jpg a mountain landscape at sunrise
```

## Train Image DiT

```bash
python scripts/train_image.py ^
  --image-dir data/images ^
  --captions data/captions.txt ^
  --epochs 10 ^
  --batch-size 32
```

Checkpoints are written to `checkpoints/`.

## Notes

- The included text encoder is intentionally simple: a deterministic hash tokenizer plus mean-pooled embedding table. It is useful for learning and smoke tests, but a real project should replace it with a stronger text encoder.
- The diffusion update is intentionally minimal and readable. It is a foundation for experimenting with beta schedules, DDPM/DDIM sampling, EMA weights, mixed precision, and better conditioning.
- The original files were preserved in `experiments/` and may still contain absolute local paths. Treat them as lab notes, not production entry points.
