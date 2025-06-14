import cv2
import os
import json
from collections import defaultdict

def get_image_dimensions():
    """Get dimensions of all cropped images"""
    dimensions = {}
    # First get the mapping of original files to cropped files from results
    with open("ocr_results.json") as f:
        data = json.load(f)
    
    for filename, result in data.items():
        if "cropped" in result:
            cropped_path = os.path.join("IMAGES/cropped", result["cropped"])
            if os.path.exists(cropped_path):
                img = cv2.imread(cropped_path)
                if img is not None:
                    dimensions[filename] = {
                        "shape": img.shape,
                        "cropped_file": result["cropped"]
                    }
    return dimensions

def analyze_layout():
    """Analyze the current layout representation"""
    with open("ocr_results.json") as f:
        data = json.load(f)
    
    # Analyze block types and their relationships
    block_types = defaultdict(int)
    block_relationships = defaultdict(list)
    text_lengths = []
    
    for filename, result in data.items():
        if "content" in result and "blocks" in result["content"]:
            blocks = result["content"]["blocks"]
            for i, block in enumerate(blocks):
                block_types[block["type"]] += 1
                text_lengths.append(len(block["text"]))
                
                # Check spatial relationships
                if i > 0:
                    prev_block = blocks[i-1]
                    # Check if blocks are vertically aligned
                    if (abs(block["position"]["top_left"][0] - prev_block["position"]["top_left"][0]) < 50 and
                        block["position"]["top_left"][1] > prev_block["position"]["bottom_left"][1]):
                        block_relationships["vertical"].append((prev_block["text"], block["text"]))
                    # Check if blocks are horizontally aligned
                    elif (abs(block["position"]["top_left"][1] - prev_block["position"]["top_left"][1]) < 50 and
                          block["position"]["top_left"][0] > prev_block["position"]["top_right"][0]):
                        block_relationships["horizontal"].append((prev_block["text"], block["text"]))
    
    return {
        "dimensions": get_image_dimensions(),
        "block_types": dict(block_types),
        "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        "relationships": {k: len(v) for k, v in block_relationships.items()}
    }

if __name__ == "__main__":
    analysis = analyze_layout()
    print("\nCropped Image Dimensions:")
    for filename, info in analysis["dimensions"].items():
        height, width, channels = info["shape"]
        print(f"{filename} -> {info['cropped_file']}: {width}x{height}")
    
    print("\nBlock Types Distribution:")
    for block_type, count in analysis["block_types"].items():
        print(f"{block_type}: {count}")
    
    print(f"\nAverage text length per block: {analysis['avg_text_length']:.1f} characters")
    
    print("\nBlock Relationships:")
    for rel_type, count in analysis["relationships"].items():
        print(f"{rel_type}: {count} relationships")
    
    # Additional analysis
    total_images = len(analysis["dimensions"])
    print(f"\nTotal images with cropped versions: {total_images}")
    print(f"Total images processed: {len([f for f in os.listdir('IMAGES') if f.endswith('.heic')])}")
    print(f"Cropping success rate: {(total_images/len([f for f in os.listdir('IMAGES') if f.endswith('.heic')]))*100:.1f}%") 