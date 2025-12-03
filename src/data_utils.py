import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cairosvg
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class Sample:
    text: str
    image_path: Path


def convert_svgs_to_png(data_dir: Path, output_dir: Path, scale: float = 1.0) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_paths = sorted(
        data_dir.glob("*.svg"), key=lambda p: int(re.search(r"(\d+)$", p.stem).group(1)) if re.search(r"(\d+)$", p.stem) else p.stem
    )
    png_paths: List[Path] = []
    for svg_path in svg_paths:
        png_path = output_dir / f"{svg_path.stem}.png"
        if not png_path.exists():
            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=scale)
        png_paths.append(png_path)
    return png_paths


def load_texts(text_path: Path) -> List[str]:
    with open(text_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def build_vocab(samples: List[str]) -> Dict[str, int]:
    chars = sorted({c for text in samples for c in text})
    stoi = {c: i + 2 for i, c in enumerate(chars)}  # reserve 0 for padding, 1 for unknown
    stoi["<unk>"] = 1
    stoi["<pad>"] = 0
    return stoi


def encode_text(text: str, stoi: Dict[str, int], max_length: int) -> torch.Tensor:
    encoded = [stoi.get(c, stoi["<unk>"]) for c in text[:max_length]]
    if len(encoded) < max_length:
        encoded += [stoi["<pad>"]] * (max_length - len(encoded))
    return torch.tensor(encoded, dtype=torch.long)


class HandwritingDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        text_file: Path,
        processed_dir: Path,
        image_size: Tuple[int, int] = (128, 512),
        max_text_length: int = 160,
        svg_scale: float = 1.0,
    ):
        self.data_dir = data_dir
        self.text_file = text_file
        self.processed_dir = processed_dir
        self.image_size = image_size
        self.max_text_length = max_text_length

        texts = load_texts(text_file)
        self.vocab = build_vocab(texts)
        png_paths = convert_svgs_to_png(data_dir, processed_dir, scale=svg_scale)
        if len(png_paths) != len(texts):
            raise ValueError("Mismatch between number of SVG pages and lines in text file")
        self.samples: List[Sample] = []
        for png_path, text in zip(png_paths, texts):
            self.samples.append(Sample(text=text, image_path=png_path))

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(self.image_size, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path)
        image = self.transform(image)
        text_tensor = encode_text(sample.text, self.vocab, self.max_text_length)
        return image, text_tensor

    def save_vocab(self, output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)


def load_vocab(vocab_path: Path) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_text(indices: torch.Tensor, itos: Dict[int, str]) -> str:
    chars = [itos.get(int(idx), "") for idx in indices if int(idx) != 0]
    return "".join(chars)


def prepare_mapping(vocab: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in vocab.items()}
