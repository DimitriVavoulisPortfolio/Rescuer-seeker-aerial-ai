import cv2
from ultralytics import YOLO
from pathlib import Path

# Get the directory of the current script
script_dir = Path(__file__).parent.absolute()

# Model name (adjust this to your model's filename)
MODEL_NAME = "best.pt"

# Path to the model file
model_path = script_dir / MODEL_NAME

# Path to the input image (adjust the filename as needed)
input_image_path = script_dir / "maxresdefault.jpg"

# Path for the output image
output_image_path = script_dir / "test_image_result.jpg"

def process_single_image():
    # Load the model
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Check if input image exists
    if not input_image_path.exists():
        print(f"Error: Input image not found at {input_image_path}")
        return

    # Run inference on the image
    try:
        results = model(str(input_image_path), conf=0.25, iou=0.45)[0]
        
        # Load the image for drawing
        img = cv2.imread(str(input_image_path))
        
        # Draw bounding boxes and labels on the image
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls = int(box.cls)
            label = f"{results.names[cls]} {conf:.2f}"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the result
        cv2.imwrite(str(output_image_path), img)
        print(f"Processed image saved to {output_image_path}")

    except Exception as e:
        print(f"Error during image processing: {e}")

if __name__ == "__main__":
    process_single_image()
