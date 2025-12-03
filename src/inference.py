import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.data_utils import encode_text, load_vocab
from src.model import ConditionalVAE, ModelConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Generate handwriting samples from text")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint")
    parser.add_argument("--vocab", type=Path, required=True, help="Path to vocab.json")
    parser.add_argument("--texts", type=str, nargs="+", required=True, help="Texts to render")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-text-length", type=int, default=160)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> ConditionalVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ModelConfig(**checkpoint["config"])
    model = ConditionalVAE(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    vocab = load_vocab(args.vocab)
    stoi = vocab

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    text_tensors = [encode_text(text, stoi, args.max_text_length) for text in args.texts]
    batch = torch.stack(text_tensors).to(device)
    with torch.no_grad():
        images = model.generate(batch, device=device)
    for text, image in zip(args.texts, images):
        safe_text = text.replace(" ", "_")[:50]
        out_path = output_dir / f"sample_{safe_text}.png"
        save_image(image, out_path, normalize=True, value_range=(-1, 1))
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
