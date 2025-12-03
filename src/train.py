import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from src.data_utils import HandwritingDataset
from src.model import ConditionalVAE, ModelConfig, VAELoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional VAE for handwriting synthesis")
    parser.add_argument("--data-dir", type=Path, default=Path("Handwriting Data"), help="Directory with SVG pages")
    parser.add_argument("--text-file", type=Path, default=Path("Handwriting Data/text.txt"), help="Text file with page transcripts")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Where to cache rendered PNGs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-text-length", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--svg-scale", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def train():
    args = parse_args()
    dataset = HandwritingDataset(
        data_dir=args.data_dir,
        text_file=args.text_file,
        processed_dir=args.processed_dir,
        image_size=(args.image_height, args.image_width),
        max_text_length=args.max_text_length,
        svg_scale=args.svg_scale,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    config = ModelConfig(vocab_size=len(dataset.vocab), image_size=(args.image_height, args.image_width))
    model = ConditionalVAE(config).to(args.device)
    criterion = VAELoss(kl_weight=args.kl_weight)
    optimizer = Adam(model.parameters(), lr=args.lr)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_vocab(args.checkpoint_dir / "vocab.json")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        recon_running = 0.0
        kl_running = 0.0
        for images, texts in tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(args.device)
            texts = texts.to(args.device)

            optimizer.zero_grad()
            recon, mu, logvar = model(images, texts)
            loss, recon_loss, kl_loss = criterion(recon, images, mu, logvar)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            recon_running += recon_loss.item()
            kl_running += kl_loss.item()

        num_batches = len(dataloader)
        print(
            f"Epoch {epoch}: loss={epoch_loss/num_batches:.4f}, recon={recon_running/num_batches:.4f}, kl={kl_running/num_batches:.4f}"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = args.checkpoint_dir / f"handwriting_vae_epoch{epoch}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.__dict__,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train()
