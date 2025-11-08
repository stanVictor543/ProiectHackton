# organize_images_mod3_updated.py
# Moves files named im0001.jpg .. im2000.jpg from D:\hrp\images into train/val/test using modulo 3.
# Mapping: i % 3 == 0 -> train, == 1 -> val, == 2 -> test

import shutil
from pathlib import Path

# configuration
src_dir = Path(r"D:\archive (1)\images")       # source directory containing the images
train_dir = src_dir / "train"
val_dir   = src_dir / "val"
test_dir  = src_dir / "test"

start = 1
end = 2000       # <<< MODIFIED: Updated from 1856 to 2000

# safety options
dry_run = False       # If True: only print actions, don't move files
overwrite = False      # If True: overwrite existing files at destination

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

print(f"Starting process for images {start} to {end}...")

for i in range(start, end + 1):
    fname = f"im{i:04d}.jpg"  # <<< MODIFIED: Added 'im' prefix
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
                # If dry_run, we still print the "MOVE" message below
            except Exception as e:
                print(f"[ERROR] Couldn't remove existing {dst}: {e}")
                errors += 1
                continue
        else:
            print(f"[SKIP] Destination exists, skipping: {dst}")
            skipped += 1
            continue

    print(f"[MOVE] {src.name} -> {dst_dir.relative_to(src_dir)}")
    if not dry_run:
        try:
            shutil.move(str(src), str(dst))
            moved += 1
        except Exception as e:
            print(f"[ERROR] Failed to move {src} -> {dst}: {e}")
            errors += 1

print("\n--- Summary ---")
print(f"  Moved   : {moved}")
print(f"  Skipped : {skipped} (destination existed)")
print(f"  Missing : {missing} (source file not found)")
print(f"  Errors  : {errors}")
print("Done.")