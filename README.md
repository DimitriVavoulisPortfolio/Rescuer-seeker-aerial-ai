# Rescuer-seeker: AI-Powered Aerial Search for Rapid Rescue Operations


Uploading output_video.mp4…


[![Watch the video](https://img.youtube.com/vi/XaHIW4iETJg/maxresdefault.jpg)](https://www.youtube.com/embed/XaHIW4iETJg)

## Project Overview

Rescuer-seeker is a cutting-edge application designed to revolutionize aerial search and rescue operations in disaster scenarios with live aerial drone detection. Utilizing the YOLOv11m architecture, this project aims to rapidly detect and localize critical elements - Houses, People, and Vehicles - from aerial imagery, providing invaluable real-time intelligence to rescue teams, and ultimately, save lives.

### Key Features

- YOLOv11m model optimized for aerial disaster imagery analysis
- Detects three critical classes: Houses, People, and Vehicles
- High-speed inference suitable for real-time drone operations
- Trained on a diverse dataset of disaster scenarios
- Comprehensive testing and evaluation framework

## Model Performance

Precision
- **People**: 0.813
- **Houses**: 0.776
- **Vehicles**: 0.662

## Project Structure

1. `Rescuer-seeker App v1.0`: The entire application
2. `Testing`: Testing folder with test dataset, scripts, and outputs including logs
3. `Training`: Training folder with training dataset as well as outputs including metrics and logging
4. `Dataset`: https://universe.roboflow.com/deep-learning-dohjx/natural-disater

## Quick Start Guide

1. Clone the repository:
   ```
   git clone https://github.com/DimitriVavoulisPortfolio/Rescuer-seeker-aerial-ai
   cd Rescuer-seeker-aerial-ai
   ```

2. Install dependencies:
   ```
   pip install ultralytics opencv-python torch torchvision torchaudio numpy scikit-learn pyyaml
   ```

3. To test the model on the test dataset:
   ```
   python test-script.py
   ```

4. For live video detection:
   ```
   python live-video-detection.py
   ```

Note: Ensure you have the trained model file (best.pt) in the same directory as the scripts.

## Documentation

- [MODEL.md](MODEL.md): Detailed information about the model architecture, training process, and performance metrics.
- [PROCESS.md](PROCESS.md): Comprehensive overview of the development process, challenges faced, and solutions implemented.

## Applications

1. Rapid area assessment in disaster zones
2. Resource allocation for emergency responders
3. Dynamic mission planning for rescue operations
4. Damage estimation for infrastructure
5. Survivor location tracking

## Future Work

- Implement multi-spectral imaging for improved detection in various conditions
- Develop a lightweight model version for edge deployment on drones
- Integrate with geo-mapping tools for precise location reporting
- Implement temporal analysis for tracking changes over time during ongoing operations

## License

This project is licensed under the Apache 2.0 license - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please open an issue in this repository or contact [Dimitri Vavoulis](mailto:dimitrivavoulis3@gmail.com).

## Acknowledgments

- Ultralytics for the YOLO implementation
