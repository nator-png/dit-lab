# DiT Lab

PyTorch Diffusion Transformer workbench for image, audio, and video generation.

This repo now has two layers:

- `src/dit_lab/`: reusable package code with configs, model modules, datasets, and a standard training loop.
- `experiments/`: archived prototypes and early research sketches.

## Project Layout

```text
.
|-- src/dit_lab/          # Reusable package code
|-- scripts/              # Command-line training entry points
|-- experiments/          # Archived prototypes
|-- notebooks/            # Jupyter notebooks
|-- data/                 # Local datasets, ignored by git except placeholders
|-- checkpoints/          # Model checkpoints, ignored by git
|-- outputs/              # Generated samples, ignored by git
`-- tests/                # Smoke tests
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

## Development Roadmap

- Add CLIP or another pretrained text encoder for stronger prompt conditioning.
- Add DDPM/DDIM beta schedules, EMA checkpoints, and mixed precision training.
- Expand audio and video training scripts from the shared model components.
