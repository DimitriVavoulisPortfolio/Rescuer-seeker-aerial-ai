import os
import yaml
import logging
import time
import torch
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def get_data_yaml(dataset_path):
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml not found in {dataset_path}")
    return yaml_path

def train_and_test():
    logging.info("Starting training and testing process for natural disaster detection")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        logging.warning("CUDA is not available. Using CPU. This may significantly slow down training.")
        device = 'cpu'

    try:
        # Get dataset path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, 'Natural Disater.v14-v2.yolov11')
        data_yaml = get_data_yaml(dataset_path)
        logging.info(f"Using data YAML file: {data_yaml}")

        # Initialize YOLOv11m-obb model
        model_path = os.path.join(current_dir, 'yolo11m.pt')
        model = YOLO(model_path)
        logging.info("YOLOv11m model initialized")

        # Train the model with optimized parameters
        results = model.train(
            data=data_yaml,
            epochs=200,
            imgsz=640,
            batch=8,
            patience=100,
            device=device,
            workers=8,
            project='natural_disaster_detection',
            name='yolov11m',
            exist_ok=True,
            pretrained=True,
            optimizer='SGD',
            lr0=0.0005,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=10.0,
            cls=0.3,
            dfl=1.5,
            label_smoothing=0.1,
            close_mosaic=10,
            amp=True,
            fraction=1.0,
            save=True,
            save_period=10,
            cache='disk',
            multi_scale=True,
            rect=False,
            resume=False,
            nbs=64,
            mixup=0.2,
            copy_paste=0.1,
            degrees=0,
            translate=0.1,
            scale=0.5,
            shear=0,
            perspective=0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.5,
            erasing=0.0,
            single_cls=False,
        )
        logging.info("Training complete!")

        # Save the final model
        model.save('natural_disaster_detection_yolov11m_obb_final.pt')
        logging.info("Final model saved as 'natural_disaster_detection_yolov11m_obb_final.pt'")

        # Validate the model
        val_results = model.val()
        logging.info(f"Validation results: {val_results}")

        # Test the model
        logging.info("Starting testing")
        test_results = model.val(split='test')

        # Log results
        logging.info("Test results:")
        logging.info(f"mAP50: {test_results.box.map50:.4f}")
        logging.info(f"mAP50-95: {test_results.box.map:.4f}")
        logging.info(f"Precision: {test_results.box.mp:.4f}")
        logging.info(f"Recall: {test_results.box.mr:.4f}")

        # Log per-class metrics
        for i, c in enumerate(model.names.values()):
            logging.info(f"Class {c}:")
            logging.info(f"  mAP50: {test_results.box.maps50[i]:.4f}")
            logging.info(f"  Precision: {test_results.box.p[i]:.4f}")
            logging.info(f"  Recall: {test_results.box.r[i]:.4f}")

    except Exception as e:
        logging.error(f"An error occurred during training or testing: {str(e)}")
        raise

if __name__ == '__main__':
    train_and_test()
