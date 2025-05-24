import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize, to_tensor
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import numpy as np
from torchvision.models import vit_b_16
import torch.nn as nn
import os
import json
import argparse
import sys
from pathlib import Path

def detect_road_damage(image_path, output_dir="output"):
    """Detect road damage in an image and return structured results"""
    try:
        # ========== PATH SETUP ==========
        current_dir = Path(__file__).parent.absolute()
        models_dir = current_dir / "models"
        
        multiclass_model_path = models_dir / "multiclass_model.pth"
        yolo_model_path = models_dir / "BB_MODEL1.pt"
        output_image_path = Path(output_dir) / f"{Path(image_path).stem}_analyzed.jpg"
        os.makedirs(output_dir, exist_ok=True)

        # ========== MULTICLASS CLASSIFICATION MODEL ==========
        class MulticlassModel(nn.Module):
            def __init__(self):
                super(MulticlassModel, self).__init__()
                self.model = vit_b_16(weights=None)
                self.model.heads.head = nn.Linear(self.model.heads.head.in_features, 3)

            def forward(self, x):
                return self.model(x)

        # Check if models exist
        if not multiclass_model_path.exists() or not yolo_model_path.exists():
            print(f"WARNING: Models not found. Using mock data for testing.")
            print(f"Multiclass model exists: {multiclass_model_path.exists()}")
            print(f"YOLO model exists: {yolo_model_path.exists()}")
            
            # Create a mock output image
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return {
                    "success": False,
                    "error": f"Could not read image at {image_path}"
                }
                
            # Draw a mock bounding box
            height, width = img.shape[:2]
            cv2.rectangle(img, (int(width*0.3), int(height*0.3)), (int(width*0.7), int(height*0.7)), (0, 255, 0), 2)
            cv2.putText(img, "MOCK: Pothole", (int(width*0.3), int(height*0.3) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the output image
            output_image_path = str(Path(output_dir) / f"{Path(image_path).stem}_mock_analyzed.jpg")
            cv2.imwrite(output_image_path, img)
            
            # Return mock classification results
            return {
                "success": True,
                "message": "Classification completed with mock data (models not found)",
                "classification": {
                    "is_road": True,
                    "damage_type": "pothole",
                    "confidence": 0.95,
                    "severity": "high"
                },
                "detections": [
                    {
                        "class": "pothole",
                        "confidence": 0.95,
                        "bbox": [int(width*0.3), int(height*0.3), int(width*0.7), int(height*0.7)]
                    }
                ],
                "output_image": output_image_path
            }

        # Load models
        multiclass_model = MulticlassModel()
        multiclass_model.load_state_dict(torch.load(str(multiclass_model_path), map_location="cpu"))
        multiclass_model.eval()
        class_labels = ['normal_road', 'cracks', 'potholes']

        # ========== YOLOv8 MODEL ==========
        bb_model = YOLO(str(yolo_model_path))

        # ========== SKIP DEPTH MODEL (MiDaS) ==========
        print("Skipping MiDaS depth estimation model - using simpler approach")
        use_midas = False  # Always disable MiDaS to avoid errors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ========== IMAGE PREPROCESSING ==========
        image_pil = Image.open(image_path).convert("RGB")

        # ========== MULTICLASS CLASSIFICATION ==========
        cls_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        cls_input = cls_transform(image_pil).unsqueeze(0)
        with torch.no_grad():
            cls_output = multiclass_model(cls_input)
            predicted_class_idx = cls_output.argmax().item()
            predicted_class = class_labels[predicted_class_idx]
            class_confidence = torch.softmax(cls_output, dim=1)[0][predicted_class_idx].item()

        # ========== YOLO OBJECT DETECTION ==========
        results = bb_model(image_path)
        # Don't use the default plot() method as it adds colored boxes with labels
        # Instead, we'll draw our own green boxes without labels
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        # ========== LOAD IMAGE FOR VISUALIZATION ==========
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("Image loaded for visualization successfully")
        
        # ========== SEVERITY CALCULATION ==========
        def calculate_severity(bbox, image_shape):
            # Extract and validate bbox coordinates
            x_min, y_min, x_max, y_max = map(int, bbox)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image_shape[1], x_max)
            y_max = min(image_shape[0], y_max)

            # Calculate bbox area
            bbox_area = (x_max - x_min) * (y_max - y_min)
            
            # Calculate severity based on relative size
            image_area = image_shape[0] * image_shape[1]
            relative_size = bbox_area / image_area
            
            print(f"Damage relative size: {relative_size:.4f} of image area")
            
            if relative_size < 0.01:  # Less than 1% of image
                return "low"
            elif relative_size < 0.05:  # Between 1% and 5% of image
                return "medium"
            else:  # More than 5% of image
                return "high"

        # ========== PREPARE RESULTS ==========
        detections = []
        for i, box in enumerate(boxes):
            severity = calculate_severity(box, img.shape)
            # Convert plural class names to singular (e.g., 'potholes' to 'pothole')
            class_name = predicted_class.replace('s', '') if predicted_class.endswith('s') else predicted_class
            
            detections.append({
                "bbox": box.tolist(),
                "confidence": float(confidences[i]),
                "class": class_name,
                "severity": severity
            })

        # ========== DRAW ON IMAGE ==========
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            severity = detection["severity"]
            damage_type = detection["class"]
            
            # Always use green color for bounding boxes regardless of severity
            color = (0, 255, 0)  # Green color
            
            # Calculate the center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Calculate the dimensions of the image
            img_height, img_width = img.shape[:2]
            
            # Create a larger bounding box (40% of image size)
            box_size = min(img_width, img_height) // 3
            
            # Calculate new coordinates for the larger box
            new_x1 = max(0, center_x - box_size // 2)
            new_y1 = max(0, center_y - box_size // 2)
            new_x2 = min(img_width, center_x + box_size // 2)
            new_y2 = min(img_height, center_y + box_size // 2)
            
            # Draw bounding box with a thicker green line - no labels
            cv2.rectangle(img, (new_x1, new_y1), (new_x2, new_y2), color, 3)  # Green color, thicker line
            # No text labels

        # ========== SAVE IMAGE ==========
        cv2.imwrite(str(output_image_path), img)

        # Determine if it's a road based on classification
        is_road = predicted_class != "normal_road"

        # Return structured results
        return {
            "success": True,
            "is_road": is_road,
            "road_type": "damaged road" if is_road else "normal road",
            "damage_type": predicted_class.replace('s', ''),  # Convert 'potholes' to 'pothole', 'cracks' to 'crack'
            "confidence": float(class_confidence),
            "severity": "high" if predicted_class == "potholes" else "medium",
            "detections": detections,
            "output_image": str(output_image_path),
            "summary": f"{'Road' if is_road else 'Not a road'} with {len(detections)} damage instances detected"
        }
    
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def main():
    parser = argparse.ArgumentParser(description="Detect road damage in an image")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--output", default="output", help="Output directory for analyzed images")
    args = parser.parse_args()
    
    results = detect_road_damage(args.image, args.output)
    print(json.dumps(results, indent=2))
    return 0 if results["success"] else 1

if __name__ == "__main__":
    sys.exit(main())
