import os
import time
import json
import logging
import numpy as np
import cv2
from logging.handlers import RotatingFileHandler
from google.cloud import vision
from google.api_core.exceptions import (GoogleAPIError, DeadlineExceeded, 
                                       ServiceUnavailable, InternalServerError,
                                       InvalidArgument)
try:
    from tqdm import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False

# HEIC support imports
try:
    from PIL import Image
    import pillow_heif
    pillow_heif.register_heif_opener()
    has_heic_support = True
except ImportError:
    has_heic_support = False

# Configuration
CONFIG = {
    "delay_between_calls": 0.5,
    "max_retries": 5,
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "log_file": "ocr_process.log",
    "output_file": "ocr_results.json",
    "supported_extensions": {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.heic', '.heif'},
    "cropped_dir": "cropped",
    "converted_dir": "converted",  # New directory for converted HEIC files
    "save_cropped": True,
    "save_converted": True,  # Whether to save converted HEIC files
    "enable_cropping": True,  # Re-enable cropping (set to True)
    "enable_heic_conversion": True,  # Set to False to disable HEIC conversion
    "converted_format": "jpg",  # Format to convert HEIC files to
    "converted_quality": 95,  # JPEG quality for converted files
    "standard_crop_size": (2000, 1200),  # Standard size for all cropped images (width, height)
    "crop_params": {
        "min_card_area": 5000,
        "max_aspect_ratio": 2.0,
        "min_aspect_ratio": 0.5,
        "border_size": 20,
        "adaptive_thresh_block": 21,  # Will be adjusted to odd number
        "adaptive_thresh_c": 5,
        "perspective_correction": True
    }
}

# Setup logging with rotation
log_handler = RotatingFileHandler(
    CONFIG["log_file"],
    maxBytes=5*1024*1024,  # 5MB
    backupCount=3
)
log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logging.getLogger().addHandler(log_handler)
logging.getLogger().setLevel(logging.INFO)

# Ensure adaptive_thresh_block is odd
CONFIG["crop_params"]["adaptive_thresh_block"] = CONFIG["crop_params"]["adaptive_thresh_block"] | 1

# Retryable exceptions
RETRYABLE_EXCEPTIONS = (DeadlineExceeded, ServiceUnavailable, InternalServerError)

def check_heic_support():
    """Check if HEIC support is available and warn if not"""
    if not has_heic_support:
        logging.warning("HEIC support not available. Install pillow-heif: pip install pillow-heif")
        print("Warning: HEIC support not available. To process HEIC files, install pillow-heif:")
        print("pip install pillow-heif")
        return False
    return True

def create_directories(image_dir):
    """Create necessary directories for converted and cropped images"""
    directories = {}
    
    # Create converted directory if HEIC conversion is enabled
    if CONFIG["enable_heic_conversion"] and CONFIG["save_converted"]:
        converted_dir = os.path.join(image_dir, CONFIG["converted_dir"])
        if not os.path.exists(converted_dir):
            os.makedirs(converted_dir)
            logging.info(f"Created converted directory: {converted_dir}")
        directories['converted'] = converted_dir
    
    # Create cropped directory if cropping is enabled
    if CONFIG["save_cropped"] and CONFIG["enable_cropping"]:
        cropped_dir = os.path.join(image_dir, CONFIG["cropped_dir"])
        if not os.path.exists(cropped_dir):
            os.makedirs(cropped_dir)
            logging.info(f"Created cropped directory: {cropped_dir}")
        directories['cropped'] = cropped_dir
    
    return directories

def is_heic_file(filename):
    """Check if file is a HEIC/HEIF file"""
    return os.path.splitext(filename)[1].lower() in {'.heic', '.heif'}

def convert_heic_to_standard(image_dir, filename):
    """Convert HEIC file to standard format (JPEG)"""
    if not CONFIG["enable_heic_conversion"] or not has_heic_support:
        return None
    
    try:
        file_path = os.path.join(image_dir, filename)
        
        # Open HEIC file
        with Image.open(file_path) as image:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate output filename
            base_name = os.path.splitext(filename)[0]
            converted_filename = f"{base_name}.{CONFIG['converted_format']}"
            
            if CONFIG["save_converted"]:
                converted_dir = os.path.join(image_dir, CONFIG["converted_dir"])
                converted_path = os.path.join(converted_dir, converted_filename)
                
                # Save converted file
                save_kwargs = {}
                if CONFIG["converted_format"].lower() in ['jpg', 'jpeg']:
                    save_kwargs['quality'] = CONFIG["converted_quality"]
                    save_kwargs['optimize'] = True
                
                # Use 'JPEG' as the format string for jpg/jpeg files
                save_format = 'JPEG' if CONFIG["converted_format"].lower() in ['jpg', 'jpeg'] else CONFIG["converted_format"].upper()
                image.save(converted_path, format=save_format, **save_kwargs)
                logging.info(f"Converted HEIC to {save_format}: {converted_path}")
                return converted_path
            else:
                # Return the PIL Image object for in-memory processing
                return image
                
    except Exception as e:
        logging.error(f"HEIC conversion failed for {filename}: {str(e)}")
        return None

def create_cropped_dir(image_dir):
    """Create directory for cropped images if enabled (legacy function for compatibility)"""
    if CONFIG["save_cropped"] and CONFIG["enable_cropping"]:
        cropped_dir = os.path.join(image_dir, CONFIG["cropped_dir"])
        if not os.path.exists(cropped_dir):
            os.makedirs(cropped_dir)
            logging.info(f"Created cropped directory: {cropped_dir}")
        return cropped_dir
    return None

def get_block_type_name(block_type_value):
    """Get human-readable name for block type"""
    try:
        # More robust way to handle block types
        from google.cloud.vision_v1.types import Block
        return Block.BlockType(block_type_value).name
    except (ValueError, ImportError):
        return f"UNKNOWN({block_type_value})"

def process_layout(document):
    """Process document layout into structured format, robust to missing fields"""
    if not document or not hasattr(document, 'pages'):
        return []
    layout_info = []
    for page in getattr(document, 'pages', []):
        for block in getattr(page, 'blocks', []):
            block_text = ""
            for paragraph in getattr(block, 'paragraphs', []):
                for word in getattr(paragraph, 'words', []):
                    if hasattr(word, 'symbols'):
                        word_text = ''.join([symbol.text for symbol in getattr(word, 'symbols', [])])
                        block_text += word_text + ' '
                block_text += '\n'
            block_text = block_text.strip()
            layout_info.append({
                "block_type": get_block_type_name(getattr(block, 'block_type', None)),
                "confidence": getattr(block, 'confidence', None),
                "bounding_box": [(v.x, v.y) for v in getattr(block, 'bounding_box', {}).vertices] if hasattr(block, 'bounding_box') and hasattr(block.bounding_box, 'vertices') else [],
                "text": block_text
            })
    return layout_info

def detect_and_crop_card(image_dir, filename, source_path=None):
    """Detects card contours and crops image to card boundaries
    
    Args:
        image_dir: Base directory for saving cropped images
        filename: Original filename for naming purposes
        source_path: Optional path to source image (for converted files)
    """
    if not CONFIG["enable_cropping"]:
        return None
        
    try:
        # Use source_path if provided, otherwise construct from image_dir and filename
        file_path = source_path or os.path.join(image_dir, filename)
        
        # Read image
        img = cv2.imread(file_path)
        if img is None:
            logging.error(f"Could not read image: {file_path}")
            return None
            
        orig_height, orig_width = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV,
            CONFIG["crop_params"]["adaptive_thresh_block"],
            CONFIG["crop_params"]["adaptive_thresh_c"]
        )
        
        # Morphological operations to close gaps
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours (version-proof method)
        contours_data = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]
        
        if not contours:
            logging.info(f"No contours found in {filename}")
            return None
            
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        card_contour = None
        for contour in contours:
            # Approximate contour polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Check if contour is quadrilateral
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                # Safety check for division by zero
                if h == 0:
                    continue
                    
                aspect_ratio = w / float(h)
                area = w * h
                
                # Validate size and aspect ratio
                if (area > CONFIG["crop_params"]["min_card_area"] and
                    CONFIG["crop_params"]["min_aspect_ratio"] < aspect_ratio < CONFIG["crop_params"]["max_aspect_ratio"]):
                    card_contour = approx
                    break
        
        if card_contour is None:
            logging.info(f"No valid card contour found in {filename}")
            return None
            
        # Apply perspective correction if enabled
        if CONFIG["crop_params"]["perspective_correction"]:
            try:
                # Order points clockwise: top-left, top-right, bottom-right, bottom-left
                rect = order_points(card_contour.reshape(4, 2))
                
                # Destination points for perspective transform
                width = max(np.linalg.norm(rect[0]-rect[1]), np.linalg.norm(rect[2]-rect[3]))
                height = max(np.linalg.norm(rect[1]-rect[2]), np.linalg.norm(rect[3]-rect[0]))
                
                # Safety check for zero dimensions
                if width <= 0 or height <= 0:
                    logging.warning(f"Invalid perspective transform dimensions for {filename}")
                    x, y, w, h = cv2.boundingRect(card_contour)
                    cropped = img[y:y+h, x:x+w]
                else:
                    dst = np.array([
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1],
                        [0, height-1]
                    ], dtype="float32")
                    
                    # Compute perspective transform matrix
                    M = cv2.getPerspectiveTransform(rect, dst)
                    
                    # Apply perspective transformation
                    warped = cv2.warpPerspective(img, M, (int(width), int(height)))
                    cropped = warped
            except Exception as e:
                logging.warning(f"Perspective correction failed: {e}. Using bounding box.")
                x, y, w, h = cv2.boundingRect(card_contour)
                cropped = img[y:y+h, x:x+w]
        else:
            # Get bounding rectangle without perspective correction
            x, y, w, h = cv2.boundingRect(card_contour)
            cropped = img[y:y+h, x:x+w]
        
        # Add padding
        border = CONFIG["crop_params"]["border_size"]
        cropped = cv2.copyMakeBorder(
            cropped, 
            border, border, border, border, 
            cv2.BORDER_CONSTANT, 
            value=[255, 255, 255]  # White padding
        )
        
        # Standardize the image size
        target_width, target_height = CONFIG["standard_crop_size"]
        cropped = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        # Save cropped image to subfolder
        if CONFIG["save_cropped"]:
            cropped_dir = os.path.join(image_dir, CONFIG["cropped_dir"])
            
            # Generate appropriate filename for cropped image
            if is_heic_file(filename):
                # For HEIC files, use the converted format extension
                base_name = os.path.splitext(filename)[0]
                cropped_filename = f"cropped_{base_name}.{CONFIG['converted_format']}"
            else:
                cropped_filename = f"cropped_{filename}"
            
            cropped_path = os.path.join(cropped_dir, cropped_filename)
            success = cv2.imwrite(cropped_path, cropped)
            if success:
                logging.info(f"Saved cropped image: {cropped_path}")
                return cropped_path
            else:
                logging.error(f"Failed to write cropped image: {cropped_path}")
                return None
        else:
            return cropped
        
    except Exception as e:
        logging.error(f"Card detection failed for {filename}: {e}")
        return None

def order_points(pts):
    """Arrange points in clockwise order starting from top-left"""
    # Sort by x-coordinate
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    
    # Split into left and right points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    
    # Sort left points by y-coordinate (top-left, bottom-left)
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most
    
    # Sort right points by y-coordinate (top-right, bottom-right)
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most
    
    return np.array([tl, tr, br, bl], dtype="float32")

def ocr_image(client, image_path):
    """Process single image with document_text_detection API (skip layout processing)"""
    # Check if file exists
    if not os.path.exists(image_path):
        logging.error(f"OCR image file not found: {image_path}")
        return None
    
    # Check file size
    try:
        file_size = os.path.getsize(image_path)
        if file_size > CONFIG["max_file_size"]:
            logging.warning(f"File {image_path} exceeds size limit ({file_size} bytes)")
            return None
    except OSError as e:
        logging.error(f"Error accessing {image_path}: {e}")
        return None
    
    for attempt in range(CONFIG["max_retries"]):
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            
            # Using document_text_detection
            response = client.document_text_detection(image=image)
            
            if response.error.message:
                logging.error(f'API error for {image_path}: {response.error.message}')
                return None
            
            # Extract only the raw text
            document = response.full_text_annotation
            text = document.text if document else ""
            layout = process_layout(document) if document else []
            return {
                "raw_text": text.strip(),
                "layout": layout
            }
            
        except RETRYABLE_EXCEPTIONS as api_err:
            wait_time = 2 ** attempt
            logging.warning(f"Retryable error ({api_err}) on {image_path}. Attempt {attempt+1}/{CONFIG['max_retries']}")
            time.sleep(wait_time)
        except InvalidArgument as e:
            logging.error(f"Invalid image for {image_path}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error processing {image_path}: {e}")
            return None
    return None

def get_next_cropped_number(cropped_dir):
    """Get the next available number for cropped files by finding the highest existing number"""
    existing_cropped = [f for f in os.listdir(cropped_dir) if f.startswith("cropped_")]
    if not existing_cropped:
        return 1
        
    # Extract numbers from filenames like "cropped_1.jpg", "cropped_2.jpg", etc.
    numbers = []
    for filename in existing_cropped:
        try:
            # Split on "cropped_" and take the first part before the extension
            num_str = filename.split("cropped_")[1].split(".")[0]
            numbers.append(int(num_str))
        except (IndexError, ValueError):
            # Skip files that don't match our naming pattern
            continue
    
    return max(numbers, default=0) + 1

def process_image(client, image_dir, filename):
    """Process single image with HEIC conversion, card detection and OCR"""
    file_path = os.path.join(image_dir, filename)
    converted_path = None
    cropped_path = None
    
    # For PNG files, attempt to crop in place, only delete if cropping is successful
    if filename.lower().endswith('.png'):
        cropped_path = detect_and_crop_card(image_dir, filename, file_path)
        if cropped_path:
            try:
                os.remove(file_path)
                logging.info(f"Cropped and deleted original PNG: {filename}")
            except Exception as e:
                logging.error(f"Failed to delete original PNG after cropping {filename}: {e}")
            ocr_path = cropped_path
        else:
            logging.warning(f"Cropping failed for PNG: {filename}. Original retained.")
            return {
                "id": filename,
                "error": "Cropping failed for PNG. Original retained."
            }
    # Step 1: Convert HEIC if necessary
    elif is_heic_file(filename):
        if not has_heic_support:
            logging.error(f"Cannot process HEIC file {filename}: pillow-heif not installed")
            return {
                "id": filename,
                "error": "HEIC support not available - install pillow-heif"
            }
        
        converted_path = convert_heic_to_standard(image_dir, filename)
        if not converted_path:
            logging.error(f"Failed to convert HEIC file: {filename}")
            return {
                "id": filename,
                "error": "HEIC conversion failed"
            }
        
        # Use converted file as source for further processing
        source_for_cropping = converted_path
        # Step 2: Crop if enabled
        if CONFIG["enable_cropping"]:
            cropped_path = detect_and_crop_card(image_dir, filename, source_for_cropping)
            if cropped_path:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logging.info(f"Deleted original image: {filename}")
                    if converted_path and os.path.exists(converted_path):
                        os.remove(converted_path)
                        logging.info(f"Deleted converted image: {os.path.basename(converted_path)}")
                except OSError as e:
                    logging.error(f"Failed to delete original image {filename}: {e}")
            else:
                logging.warning(f"Cropping failed or not needed: {filename}")
            ocr_path = cropped_path or source_for_cropping
        else:
            ocr_path = source_for_cropping
    else:
        # For non-HEIC (e.g. PNG or JPEG) files, skip conversion and attempt cropping.
        source_for_cropping = file_path
        if CONFIG["enable_cropping"]:
            cropped_path = detect_and_crop_card(image_dir, filename, source_for_cropping)
            if cropped_path:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logging.info(f"Deleted original non-HEIC image: {filename}")
                except OSError as e:
                    logging.error(f"Failed to delete original non-HEIC image {filename}: {e}")
            else:
                logging.warning(f"Cropping failed for non-HEIC image: {filename}. Original retained.")
            ocr_path = cropped_path or source_for_cropping
        else:
            ocr_path = source_for_cropping
    
    # Step 3: Perform OCR (using ocr_path, which is either cropped or original)
    result = ocr_image(client, ocr_path)
    
    # Prepare the output structure
    output = {}
    
    # Add cropping info with numbered filename (if cropped)
    if cropped_path and CONFIG["save_cropped"]:
        cropped_dir = os.path.join(image_dir, CONFIG["cropped_dir"])
        next_num = get_next_cropped_number(cropped_dir)
        ext = "." + CONFIG["converted_format"] if is_heic_file(filename) else os.path.splitext(filename)[1]
        new_cropped_name = f"cropped_{next_num}{ext}"
        new_cropped_path = os.path.join(cropped_dir, new_cropped_name)
        try:
            os.rename(cropped_path, new_cropped_path)
            output["cropped"] = new_cropped_name
            logging.info(f"Renamed cropped file to: {new_cropped_name}")
        except OSError as e:
            logging.error(f"Failed to rename cropped file: {e}")
            output["cropped"] = os.path.basename(cropped_path)
    
    if result:
        # Extract layout information with bounding boxes (or raw text if no layout)
        if result.get("layout"):
            output["content"] = {
                "blocks": [
                    {
                        "type": block["block_type"],
                        "text": block["text"],
                        "confidence": block["confidence"],
                        "position": {
                            "top_left": block["bounding_box"][0],
                            "top_right": block["bounding_box"][1],
                            "bottom_right": block["bounding_box"][2],
                            "bottom_left": block["bounding_box"][3]
                        }
                    }
                    for block in result["layout"]
                ]
            }
        else:
            output["content"] = {
                "blocks": [{
                    "type": "TEXT",
                    "text": result["raw_text"],
                    "confidence": 1.0,
                    "position": {
                        "top_left": [0, 0],
                        "top_right": [1, 0],
                        "bottom_right": [1, 1],
                        "bottom_left": [0, 1]
                    }
                }]
            }
    else:
        output["error"] = "OCR failed"
    
    return output

def process_directory(image_dir):
    """Process all images in directory with resume capability"""
    client = vision.ImageAnnotatorClient()
    results = {}
    
    # Check HEIC support if needed
    heic_files_present = any(is_heic_file(f) for f in os.listdir(image_dir) 
                            if os.path.isfile(os.path.join(image_dir, f)))
    if heic_files_present and not has_heic_support:
        print("Warning: HEIC files detected but pillow-heif is not installed.")
        print("Install it with: pip install pillow-heif")
    
    # Create necessary directories
    directories = create_directories(image_dir)
    
    # Load existing results
    if os.path.exists(CONFIG["output_file"]):
        try:
            with open(CONFIG["output_file"], 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load existing results: {e}. Starting fresh.")
    
    # Get all valid image files
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in CONFIG["supported_extensions"]
    ]
    
    # Filter out already processed files
    files_to_process = [f for f in image_files if f not in results]
    
    if not files_to_process:
        logging.info("All files already processed. Nothing to do.")
        return results
    
    # Count HEIC files for reporting
    heic_count = sum(1 for f in files_to_process if is_heic_file(f))
    if heic_count > 0:
        logging.info(f"Found {heic_count} HEIC files to process")
    
    # Setup progress display
    start_time = time.time()  # Always define start_time
    if has_tqdm:
        progress = tqdm(total=len(files_to_process), desc="Processing images")
    else:
        print(f"Processing {len(files_to_process)} images...")
    
    # Process images
    for i, filename in enumerate(files_to_process):
        logging.info(f"Processing ({i+1}/{len(files_to_process)}): {filename}")
        
        try:
            ocr_result = process_image(client, image_dir, filename)
        except Exception as e:
            logging.error(f"Processing failed for {filename}: {e}")
            ocr_result = {
                "id": filename,
                "error": str(e)
            }
        
        results[filename] = ocr_result
            
        # Update progress
        if has_tqdm:
            progress.update(1)
            progress.set_postfix(file=filename)
        elif (i+1) % 10 == 0 or (i+1) == len(files_to_process):
            elapsed = time.time() - start_time
            print(f"Processed {i+1}/{len(files_to_process)} files ({elapsed:.1f}s)")
        
        # Save after each file
        try:
            with open(CONFIG["output_file"], 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to save results: {e}")
            
        time.sleep(CONFIG["delay_between_calls"])
    
    if has_tqdm:
        progress.close()
    
    logging.info(f"Completed processing {len(files_to_process)} images")
    return results

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} /path/to/images_directory")
        sys.exit(1)
    
    images_directory = sys.argv[1]
    if not os.path.isdir(images_directory):
        print(f"Error: {images_directory} is not a valid directory")
        sys.exit(1)
    
    print("Starting OCR processing...")
    print(f"Logging to: {CONFIG['log_file']}")
    print(f"Results will be saved to: {CONFIG['output_file']}")
    
    # Check and report HEIC support
    if check_heic_support():
        print("HEIC support: AVAILABLE")
    else:
        print("HEIC support: NOT AVAILABLE (install pillow-heif for HEIC support)")
    
    if CONFIG["enable_heic_conversion"] and CONFIG["save_converted"]:
        print(f"HEIC conversion: ENABLED (saving to {CONFIG['converted_dir']}/ as {CONFIG['converted_format'].upper()})")
    elif CONFIG["enable_heic_conversion"]:
        print("HEIC conversion: ENABLED (in-memory processing)")
    else:
        print("HEIC conversion: DISABLED")
    
    if CONFIG["enable_cropping"]:
        print(f"Cropping: ENABLED")
        if CONFIG["save_cropped"]:
            print(f"Cropped images will be saved to: {os.path.join(images_directory, CONFIG['cropped_dir'])}")
    else:
        print("Cropping: DISABLED (processing original/converted images)")
    
    results = process_directory(images_directory)
    
    success_count = sum(1 for v in results.values() if v and "content" in v and v["content"].get("blocks"))
    error_count = sum(1 for v in results.values() if v and "error" in v)
    heic_processed = sum(1 for v in results.values() if v and v.get("was_heic", False))
    
    print(f"\nProcessing complete! Processed {len(results)} files.")
    print(f"Successfully OCR'd: {success_count} files")
    if heic_processed > 0:
        print(f"HEIC files processed: {heic_processed} files")
    print(f"Errors encountered: {error_count} files")
    print(f"Results saved to {CONFIG['output_file']}")