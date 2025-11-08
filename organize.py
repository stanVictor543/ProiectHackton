# organize_images_mod3.py
# Moves files named 0001.jpg .. 1856.jpg from D:\hrp\images into train/val/test using modulo 3.
# Mapping: i % 3 == 0 -> train, == 1 -> val, == 2 -> test

import shutil
from pathlib import Path

# configuration
src_dir = Path(r"D:\hrp\images")         # source directory containing the images
train_dir = src_dir / "train"
val_dir   = src_dir / "val"
test_dir  = src_dir / "test"

start = 1
end = 1856       # inclusive

# safety options
dry_run = False      # If True: only print actions, don't move files
overwrite = False    # If True: overwrite existing files at destination

# create target directories if missing
for d in (train_dir, val_dir, test_dir):
    d.mkdir(parents=True, exist_ok=True)

def dest_for_index(i: int) -> Path:
    r = i % 3
    if r == 0:
        return train_dir
    elif r == 1:
        return val_dir
    else:  # r == 2
        return test_dir

moved = 0
skipped = 0
missing = 0
errors = 0

for i in range(start, end + 1):
    fname = f"{i:04d}.jpg"
    src = src_dir / fname
    if not src.exists():
        print(f"[MISSING] {fname} not found in {src_dir}")
        missing += 1
        continue

    dst_dir = dest_for_index(i)
    dst = dst_dir / fname

    if dst.exists():
        if overwrite:
            try:
                if not dry_run:
                    dst.unlink()
            except Exception as e:
                print(f"[ERROR] Couldn't remove existing {dst}: {e}")
                errors += 1
                continue
        else:
            print(f"[SKIP] Destination exists, skipping: {dst}")
            skipped += 1
            continue

    print(f"[MOVE] {src.name} -> {dst_dir}")
    if not dry_run:
        try:
            shutil.move(str(src), str(dst))
            moved += 1
        except Exception as e:
            print(f"[ERROR] Failed to move {src} -> {dst}: {e}")
            errors += 1

print("\nSummary:")
print(f"  moved  : {moved}")
print(f"  skipped: {skipped}")
print(f"  missing: {missing}")
print(f"  errors : {errors}")