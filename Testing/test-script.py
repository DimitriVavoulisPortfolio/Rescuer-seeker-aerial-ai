import os
from ultralytics import YOLO
from pathlib import Path

# Get the directory of the current script
script_dir = Path(__file__).parent.absolute()

# Model name (use the latest saved model)
MODEL_NAME = "best.pt"  # Adjusted based on your error message

# Path to the model file
model_path = script_dir / MODEL_NAME

# Path to the test dataset
test_data_path = script_dir / "Natural Disater.v14-v2.yolov11" / "test"
test_images_path = test_data_path / "images"
test_labels_path = test_data_path / "labels"

# Output directory (same as script directory)
output_dir = script_dir / "test_results"

def check_dataset_structure():
    if not test_data_path.exists():
        print(f"Error: Test dataset folder not found at {test_data_path}")
        return False
    if not test_images_path.exists() or not test_labels_path.exists():
        print(f"Error: 'images' or 'labels' folder not found in {test_data_path}")
        return False
    if not any(test_images_path.glob('*')):
        print(f"Error: No images found in {test_images_path}")
        return False
    if not any(test_labels_path.glob('*')):
        print(f"Error: No label files found in {test_labels_path}")
        return False
    return True

def run_test():
    # Check dataset structure
    if not check_dataset_structure():
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Load the model
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create a temporary YAML file for dataset configuration
    yaml_content = f"""
path: {test_data_path}
train: images
val: images
test: images

nc: 3
names: ['House', 'Person', 'Vehicle']
    """
    yaml_file = script_dir / "temp_dataset.yaml"
    yaml_file.write_text(yaml_content)

    # Run validation on test data
    try:
        results = model.val(
            data=str(yaml_file),
            split='test',
            save=True,
            project=str(output_dir),
            name="test_predictions",
            conf=0.25,  # Confidence threshold
            iou=0.45    # NMS IoU threshold
        )
        print(f"Test completed. Results saved in {output_dir / 'test_predictions'}")
        
        # Print summary of results
        print("\nTest Results Summary:")
        print(f"mAP50: {results.box.map50:.3f}")
        print(f"mAP50-95: {results.box.map:.3f}")
        print(f"Precision: {results.box.p:.3f}")
        print(f"Recall: {results.box.r:.3f}")
        
    except Exception as e:
        print(f"Error during validation: {e}")
    finally:
        # Clean up temporary YAML file
        yaml_file.unlink(missing_ok=True)

if __name__ == "__main__":
    run_test()
