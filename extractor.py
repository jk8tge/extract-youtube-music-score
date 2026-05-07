import os
import cv2
import numpy as np
import argparse
import glob

def crop_image(image, band_position='bottom', y_value=0.3):
    """
    Crops the image to extract a horizontal band at the top or bottom.
    
    Args:
        image: The input image (numpy array).
        band_position: 'top' or 'bottom'.
        y_value: If <= 1.0, treated as a ratio of the image height.
                 If > 1.0, treated as the number of pixels.
    """
    height, width = image.shape[:2]
    
    if y_value <= 1.0:
        band_height = int(height * y_value)
    else:
        band_height = int(y_value)
        
    if band_position.lower() == 'bottom':
        y_start = height - band_height
        y_end = height
    elif band_position.lower() == 'top':
        y_start = 0
        y_end = band_height
    else:
        raise ValueError("band_position must be 'top' or 'bottom'")
        
    # Ensure bounds
    y_start = max(0, y_start)
    y_end = min(height, y_end)
    
    return image[y_start:y_end, 0:width]

def clean_score(image):
    """
    Cleans up the music score by removing yellow highlights, 
    converting to grayscale, and maximizing contrast.
    """
    # 1. Remove yellow highlights
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for yellow color in HSV
    # Hue for yellow is around 20-30. We use a slightly wider range.
    lower_yellow = np.array([61, 30, 200])
    upper_yellow = np.array([63, 60, 255])
    
    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Replace yellow pixels with white
    # (Since sheet music background is typically white)
    cleaned_image = image.copy()
    cleaned_image[mask > 0] = (255, 255, 255)
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)
    
    # 3. Contrast enhancement
    # Using Otsu's thresholding to get a clean binary image
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return thresholded

def process_directory(input_dir, output_dir, band_position, y_value):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Supported image extensions
    extensions = ['*.png', '*.jpg', '*.jpeg']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
        
    print(f"Found {len(image_paths)} images. Processing...")
    
    for idx, path in enumerate(sorted(image_paths)):
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to read {path}")
            continue
            
        cropped = crop_image(img, band_position, y_value)
        cleaned = clean_score(cropped)
        
        filename = os.path.basename(path)
        out_path = os.path.join(output_dir, filename)
        
        cv2.imwrite(out_path, cleaned)
        print(f"Processed: {filename}")
        
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract and clean music scores from screenshots.")
    parser.add_argument('--input', type=str, default='input', help="Input directory containing screenshots")
    parser.add_argument('--output', type=str, default='output', help="Output directory for processed scores")
    parser.add_argument('--band', type=str, choices=['top', 'bottom'], default='bottom', help="Position of the score ('top' or 'bottom')")
    parser.add_argument('--y_val', type=float, default=0.3, help="Height of the score band. If <= 1.0, it's a ratio. If > 1.0, it's pixels.")
    
    args = parser.parse_args()
    
    process_directory(args.input, args.output, args.band, args.y_val)
