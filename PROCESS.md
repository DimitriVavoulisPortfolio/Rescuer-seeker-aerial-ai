# Development Process: Rescuer-seeker

## 1. Project Initialization and Data Preparation

- Started with a dataset of aerial imagery from disaster scenarios(https://universe.roboflow.com/deep-learning-dohjx/natural-disater).
DISCLAIMER: Dataset utilizes black and white images, when tested with regular colored images and video it showed better performance
- Implemented data preprocessing pipeline, including resizing to 640x640 pixels and augmentation techniques.

## 2. Model Selection and Configuration

- Chose YOLOv11m as the base model for its balance of speed and accuracy.
- Configured the model for three classes: Houses, People, and Vehicles.
- Initialized with pretrained weights and adapted the model architecture for the specific task.

## 3. Training Process and Optimization

- Implemented a training pipeline using Ultralytics framework.
- Utilized a NVIDIA GeForce RTX 2070 Super GPU for training.
- Key training parameters:
  - Batch size: 8
  - Initial learning rate: 0.0005
  - Optimizer: SGD with momentum (0.937) and weight decay (0.0005)
- Implemented learning rate scheduling with cosine annealing.
- Applied various data augmentation techniques including mosaic, mixup, and copy-paste.

## 4. Challenges and Solutions

1. **Python Version Compatibility**:
   - Challenge: Persistent warnings about Python version (3.9.13 installed, >=3.10 required).
   - Solution: While not resolved during this training run, future iterations should update the Python environment to 3.10 or higher.

2. **Disk Space Management**:
   - Challenge: Warnings about insufficient disk space for caching images.
   - Solution: Adjusted caching strategy to work within available disk space constraints.

3. **Dataset Inconsistencies**:
   - Challenge: Mismatch between box and segment counts in the dataset.
   - Solution: Focused on box detection, removing segments to ensure consistency, making it about what it's about, finding people and saving lives.

4. **Resource Optimization**:
   - Challenge: Balancing model performance with computational resources.
   - Solution: Carefully tuned batch size and image size to maximize GPU utilization without exceeding memory limits.

## 5. Training Progression

- Initiated training for 200 epochs, but stopped at 130 epochs due to satisfactory performance.
- Observed consistent improvement in model metrics throughout training:
  - mAP50 increased from 0.263 in epoch 1 to 0.695 by epoch 130.
  - Precision improved from 0.325 to 0.729.
  - Recall enhanced from 0.349 to 0.668.

## 6. Evaluation and Testing

- Implemented comprehensive evaluation on the test set (522 images).
- Achieved impressive final metrics:
  - mAP50: 0.702
  - mAP50-95: 0.459
  - Precision: 0.751
  - Recall: 0.586

## 7. Performance Analysis

- Conducted per-class performance analysis:
  - Houses showed high precision (0.776) but lower recall (0.536).
  - People detection was the most accurate with high mAP50 (0.784) and balanced precision and recall.
  - Vehicles had the lowest precision (0.662) but decent recall (0.554).

## 8. Iterative Improvement

- Regularly monitored training progress and adjusted parameters as needed.
- Implemented early stopping at epoch 130 due to satisfactory performance and to prevent overfitting.

## 9. Documentation and Repo Organization

- Created comprehensive README.md, MODEL.md, and PROCESS.md files.
- Organized repository structure for clarity and reproducibility.
- Documented code with clear comments for maintainability.

## Conclusion

The development of Rescuer-seeker demonstrates the successful adaptation of state-of-the-art object detection technology to the critical domain of disaster response. Through careful model selection, dataset preparation, and iterative optimization, we've created a tool with the potential to significantly enhance the efficiency and effectiveness of aerial search and rescue operations.

Key achievements:
- Successful training of a YOLOv11m model on a custom disaster scenario dataset.
- Achieving high accuracy in detecting crucial elements (houses, people, vehicles) in aerial imagery.
- Overcoming technical challenges related to data processing and resource management.
- Creating a model that balances accuracy with the speed necessary for real-time applications in critical situations.

This project showcases the potential of AI to make a meaningful impact in disaster response and humanitarian efforts, paving the way for more advanced and efficient search and rescue technologies.
