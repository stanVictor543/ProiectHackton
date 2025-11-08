# organize_images_by_sequence.py
# Moves image files from a source directory into train/val/test subdirectories
# based on their sorted order, distributing them consecutively.
# (i % 3 == 0 -> train, == 1 -> val, == 2 -> test)

import shutil
from pathlib import Path

# --- Configuration ---
src_dir = Path(r"D:\archive\gender_dataset\men")  # source directory containing the images
train_dir = src_dir / "train"
val_dir = src_dir / "val"
test_dir = src_dir / "test"

# Define which files to consider "photos". Add/remove extensions as needed.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# --- Safety Options ---
dry_run = False     # If True: only print actions, don't move files
overwrite = False   # If True: overwrite existing files at destination

# --- End Configuration ---

# Create target directories if missing
for d in (train_dir, val_dir, test_dir):
    d.mkdir(parents=True, exist_ok=True)

def dest_for_index(i: int) -> Path:
    """Returns the correct destination directory (train/val/test) for a given index."""
    r = i % 3
    if r == 0:
        return train_dir
    elif r == 1:
        return val_dir
    else:  # r == 2
        return test_dir

moved = 0
skipped = 0
errors = 0

print(f"Scanning {src_dir} for images...")

# --- MODIFICATION: Get all files, filter by extension, and sort them ---
# This finds all files, filters for images, and sorts them alphabetically.
# This ensures "consecutive" files are processed in order.
all_files = [
    f for f in src_dir.iterdir()
    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
]
all_files.sort()  # Sort files alphabetically

if not all_files:
    print("No image files found to process. Exiting.")
    exit()

print(f"Starting process for {len(all_files)} images...")

# --- MODIFICATION: Loop over found files instead of a number range ---
for i, src in enumerate(all_files):
    # 'i' is the index (0, 1, 2, ...) used for the modulo split
    # 'src' is the Path object for the source file (e.g., D:\...\men\whatever_name.jpg)

    dst_dir = dest_for_index(i)
    dst = dst_dir / src.name  # Use the original file's name for the destination

    # Check if the destination file already exists
    if dst.exists():
        if overwrite:
            try:
                if not dry_run:
                    dst.unlink()  # Remove the existing file
                # If dry_run, we still print the "MOVE" message below
            except Exception as e:
                print(f"[ERROR] Couldn't remove existing {dst}: {e}")
                errors += 1
                continue
        else:
            print(f"[SKIP] Destination exists, skipping: {dst}")
            skipped += 1
            continue

    # Perform the move
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
print(f"  Errors  : {errors}")
print("Done.")