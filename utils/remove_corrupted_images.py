import os
import argparse
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

def verify_and_remove_corrupted_images(directory, dry_run=False, max_workers=16):
    """
    Remove corrupted image files from a directory.
    
    Args:
        directory (str): Path to the directory containing images
        dry_run (bool): If True, only print what would be removed without actually removing
        max_workers (int): Number of threads for parallel processing
    """
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    
    # Get all image files
    all_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(IMG_EXTS)
    ]
    
    if not all_files:
        print(f"No image files found in {directory}")
        return
    
    print(f"Found {len(all_files)} image files. Checking for corruption...")
    
    def verify_image(img_path):
        try:
            with Image.open(img_path) as img:
                img.verify()
            return img_path, True
        except (IOError, OSError, Image.UnidentifiedImageError) as e:
            return img_path, False
    
    corrupted_files = []
    valid_files = []
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(verify_image, img_path): img_path 
                         for img_path in all_files}
        
        for future in as_completed(future_to_path):
            img_path, is_valid = future.result()
            if is_valid:
                valid_files.append(img_path)
            else:
                corrupted_files.append(img_path)
    
    # Report results
    print(f"\nScan complete:")
    print(f"  Valid images: {len(valid_files)}")
    print(f"  Corrupted images: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\nCorrupted files found:")
        for file_path in corrupted_files:
            print(f"  {file_path}")
        
        if dry_run:
            print(f"\nDRY RUN: Would remove {len(corrupted_files)} corrupted files")
        else:
            # Remove corrupted files
            removed_count = 0
            for file_path in corrupted_files:
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                    removed_count += 1
                except OSError as e:
                    print(f"Failed to remove {file_path}: {e}")
            
            print(f"\nRemoved {removed_count} corrupted files")
    else:
        print("\nNo corrupted files found!")

def main():
    parser = argparse.ArgumentParser(description='Remove corrupted image files from a directory')
    parser.add_argument('directory', help='Directory containing image files')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be removed without actually removing files')
    parser.add_argument('--max-workers', type=int, default=16,
                       help='Number of threads for parallel processing (default: 16)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return
    
    verify_and_remove_corrupted_images(args.directory, args.dry_run, args.max_workers)

if __name__ == "__main__":
    main()