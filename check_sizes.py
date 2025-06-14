import cv2
import os

def check_cropped_sizes():
    """Check dimensions of all cropped images"""
    cropped_dir = "IMAGES/cropped"
    sizes = {}
    for f in os.listdir(cropped_dir):
        if f.endswith(".jpg"):
            img_path = os.path.join(cropped_dir, f)
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
                sizes[f] = (width, height)
    
    # Print results
    print("\nCropped Image Dimensions:")
    for filename, (width, height) in sizes.items():
        print(f"{filename}: {width}x{height}")
    
    # Check if all images have the same size
    unique_sizes = set(sizes.values())
    if len(unique_sizes) == 1:
        print(f"\nAll images have the same size: {next(iter(unique_sizes))}")
    else:
        print("\nWARNING: Images have different sizes!")
        print("Unique sizes found:", unique_sizes)

if __name__ == "__main__":
    check_cropped_sizes() 