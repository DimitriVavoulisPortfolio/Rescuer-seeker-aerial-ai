# Rescuer-seeker: AI-Powered Aerial Search for Rapid Rescue Operations

## Model Overview

This project implements an advanced YOLOv11m object detection model specifically designed for aerial search and rescue operations. The model is optimized to detect and localize three critical elements in disaster scenarios: Houses, People, and Vehicles.

## Model Architecture

This model utilized YOLOv11m. YOLOv11m is a state-of-the-art object detection architecture known for its excellent balance of speed and accuracy, making it ideal for real-time aerial search operations.

Key features of our implementation:
- Single-stage detector for efficient inference
- Customized for three classes: Houses, People, and Vehicles
- Optimized for aerial imagery specifics

Model specifications:
- Parameters: Approximately 20,055,321
- FLOPs: Around 68.2 billion per forward pass
- Input size: 640x640 pixels

## Dataset and Preprocessing

DISCLAIMER: The dataset utilizes black and white images with a blurry condition for people, with regular colored images and video it showed much better performance

The model was trained on a custom dataset of aerial imagery from disaster scenarios:
- Training set: 7,308 images
- Validation set: 1,044 images
- Test set: 522 images

Data preprocessing steps:
1. Image resizing to 640x640 pixels
2. Augmentation techniques including flipping, scaling, and mosaic augmentation
3. Splitting into train, validation, and test sets

## Training Process

The model was trained for 130 epochs with the following key parameters:
- Learning rate: Starting at 0.0005 with cosine annealing
- Batch size: 8
- Optimizer: SGD with momentum (0.937) and weight decay (0.0005)
- Data augmentation: Mosaic, mixup, and copy-paste

Training was performed on a NVIDIA GeForce RTX 2070 Super GPU, taking approximately 30 hours.

## Model Performance

Our YOLOv11m model achieved impressive results on the test set:

- mAP50 (mean Average Precision at IoU=50): 0.702
- mAP50-95: 0.459
- Precision: 0.751
- Recall: 0.586

### Per-Class Performance

Performance across the three target classes:

1. Houses: mAP50 = 0.672, Precision = 0.776, Recall = 0.536
2. People: mAP50 = 0.784, Precision = 0.813, Recall = 0.670
3. Vehicles: mAP50 = 0.650, Precision = 0.662, Recall = 0.554

## Practical Applications

This model demonstrates significant potential for several real-world applications in disaster response and search and rescue operations:

1. **Rapid Area Assessment**: Quickly scan large areas to identify inhabitable structures, stranded individuals, and usable vehicles.
2. **Resource Allocation**: Help emergency responders prioritize areas for rescue efforts based on the density of detected people and accessible structures.
3. **Dynamic Mission Planning**: Provide real-time intelligence to guide rescue drones or manned aircraft to areas of highest need.
4. **Damage Estimation**: Assess the extent of damage to infrastructure by analyzing the condition and distribution of detected houses and vehicles.
5. **Survivor Location Tracking**: Monitor movement of detected individuals over time to guide ground rescue teams.

## Future Improvements

To further enhance the model for real-world deployment:

1. Expand the dataset with more diverse disaster scenarios and environmental conditions.
2. Implement multi-spectral imaging to improve detection in low-visibility conditions.
3. Develop a more lightweight version for edge deployment on drones with limited computational resources.
4. Integrate with geo-mapping tools for precise location reporting of detected objects.
5. Implement temporal analysis to track movement and changes over time during ongoing rescue operations.

## Conclusion

The Rescuer-seeker model demonstrates the powerful potential of applying advanced deep learning techniques to critical search and rescue operations. With its ability to rapidly and accurately detect key elements in aerial imagery, this technology could significantly enhance the speed and efficiency of disaster response efforts, potentially saving many lives in the process.
