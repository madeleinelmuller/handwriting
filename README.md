# handwriting

Synthesizing custom handwriting using neural networks, taking inspiration from sjvasquez/handwriting-synthesis on GitHub.

## Setup
1. Create a virtual environment (recommended) and install dependencies (the extra index is needed for the CPU-only PyTorch wheels):
   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```
2. The repository ships with the raw SVG pages and `text.txt` transcript under `Handwriting Data/`.

## Training the model
The training entrypoint renders the SVG pages to PNG, builds a character vocabulary from `text.txt`, and fits a conditional VAE that conditions on the transcript.

```bash
python -m src.train --epochs 20 --batch-size 4 --device cuda  # use --device cpu if no GPU
```

Key arguments:
- `--data-dir` and `--text-file`: paths to the dataset (default `Handwriting Data`).
- `--processed-dir`: cache for rendered PNGs (ignored by git).
- `--image-height/--image-width`: resize target (default 128x512).
- `--max-text-length`: truncate/pad transcript tokens (default 160 chars).
- `--save-every`: checkpoint frequency (default every 5 epochs).

Checkpoints and the vocabulary are written to `artifacts/`.

### Example run (CPU)
```bash
python -m src.train --epochs 3 --batch-size 2 --device cpu --save-every 1
```
This produces checkpoints such as `artifacts/handwriting_vae_epoch3.pt` and `artifacts/vocab.json`.

## Inference
Generate synthetic handwriting from arbitrary text using a trained checkpoint:
```bash
python -m src.inference \
  --checkpoint artifacts/handwriting_vae_epoch3.pt \
  --vocab artifacts/vocab.json \
  --texts "Hello world" "The quick brown fox"
```
Images are saved under `outputs/`, normalized to `[0, 1]` for viewing.

## Project structure
- `src/data_utils.py`: dataset utilities, SVG-to-PNG rendering, vocabulary handling.
- `src/model.py`: conditional VAE with text encoder, image encoder, and decoder.
- `src/train.py`: training loop, checkpointing, and CLI configuration.
- `src/inference.py`: load checkpoints and render new handwriting samples.
- `requirements.txt`: CPU-friendly dependencies (PyTorch, CairoSVG, etc.).

## Notes
- Training and inference default to GPU if available; otherwise fall back to CPU.
- SVG rendering runs once and is cached in `data/processed/`.
