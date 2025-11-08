import sys
from pathlib import Path
from PIL import Image  # Note: ImageOps is no longer needed

def process_image(image_path):
    """
    Opens an image, creates ONE black and white (grayscale) version,
    and saves it in the same directory with a new name.
    """
    try:
        # Open the original image
        img = Image.open(image_path)
        
        # --- 1. Create Black and White (Grayscale) Version ---
        # 'L' mode stands for "Luminance"
        img_bw = img.convert('L')

        # --- Define new filename ---
        original_stem = image_path.stem
        original_suffix = image_path.suffix
        save_dir = image_path.parent  # This will be the 'oameni' folder

        # Define save path
        bw_path = save_dir / f"{original_stem}_bw{original_suffix}"

        # --- Save the new image ---
        img_bw.save(bw_path)
        
        print(f"    [SUCCESS] Saved B&W for: {image_path.name}")

    except Exception as e:
        print(f"    [ERROR] Failed to process {image_path.name}: {e}")

def main(base_folder_path):
    """
    Main function to find and process images ONLY in the 'oameni' subfolder
    within 'test', 'train', and 'value' folders.
    """
    base_path = Path(base_folder_path)
    
    if not base_path.is_dir():
        print(f"Error: Path '{base_folder_path}' is not a valid directory.")
        return

    # Primary folders
    primary_folders = ['test', 'train', 'value']
    
    # *** MODIFIED: Set target folder to 'oameni' ***
    target_subfolder = "oameni"
    
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    print(f"Starting image processing in: {base_path}\n")

    for folder_name in primary_folders:
        data_folder = base_path / folder_name
        
        if not data_folder.is_dir():
            print(f"Warning: Primary folder '{folder_name}' not found. Skipping.")
            continue

        print(f"--- Checking parent folder: {folder_name} ---")

        # Now, look for the 'oameni' folder inside the primary folder
        target_folder_path = data_folder / target_subfolder
        
        if not target_folder_path.is_dir():
            print(f"  '{target_subfolder}' folder not found in '{folder_name}'. Skipping.")
            print(f"--- Finished with {folder_name} ---\n")
            continue
        
        print(f"  Found '{target_subfolder}' folder. Processing images inside...")
        
        # Get a list of files from the target_folder_path
        files_to_process = [
            item for item in target_folder_path.iterdir() 
            if item.is_file() and item.suffix.lower() in supported_extensions
        ]
        
        if not files_to_process:
            print("    No images found to process.")
        else:
            for image_path in files_to_process:
                # Call the simplified function
                process_image(image_path)
        
        print(f"--- Finished processing '{target_subfolder}' in: {folder_name} ---\n")

    print("All folders processed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: You must provide the path to your main folder.")
        print("Usage: python process_images.py /path/to/your/folder")
    else:
        main(sys.argv[1])