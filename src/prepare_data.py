import argparse
import os
import shutil
import random
from pathlib import Path

def split_dataset(raw_dir, out_dir, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    random.seed(42)
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)

    for breed_dir in raw_dir.iterdir():
        if not breed_dir.is_dir():
            continue
        images = [f for f in breed_dir.glob("*.*") if f.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"]]
        if len(images) == 0:
            print(f"Warning: No valid images found for {breed_dir.name}, skipping")
            continue

        random.shuffle(images)
        n_total = len(images)
        n_train = max(int(train_frac * n_total), 1)
        n_val = max(int(val_frac * n_total), 1) if n_total > 2 else 0
        n_test = max(n_total - n_train - n_val, 1) if n_total - n_train - n_val > 0 else 0

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:]
        }

        for split, files in splits.items():
            if len(files) == 0:
                continue
            split_dir = out_dir / split / breed_dir.name
            split_dir.mkdir(parents=True, exist_ok=True)
            for file in files:
                shutil.copy(file, split_dir / file.name)
            print(f"{split} - {breed_dir.name}: {len(files)} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    args = parser.parse_args()

    split_dataset(args.raw_dir, args.out_dir, args.train_frac, args.val_frac, args.test_frac)
