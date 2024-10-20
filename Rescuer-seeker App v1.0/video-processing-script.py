import cv2
from ultralytics import YOLO
from pathlib import Path

# Get the directory of the current script
script_dir = Path(__file__).parent.absolute()

# Model name (adjust this to your model's filename)
MODEL_NAME = "best.pt"

# Path to the model file
model_path = script_dir / MODEL_NAME

# Path to the input video (adjust the filename as needed)
input_video_path = script_dir / "Franklin County Arkansas Flooding – Drone Footage – WeatherNation.mp4"

# Path for the output video
output_video_path = script_dir / "output_video.mp4"

def process_video():
    # Load the model
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Check if input video exists
    if not input_video_path.exists():
        print(f"Error: Input video not found at {input_video_path}")
        return

    # Open the video file
    video = cv2.VideoCapture(str(input_video_path))
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Process the video
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Run inference on the frame
        results = model(frame, conf=0.10, iou=0.45)[0]

        # Draw bounding boxes and labels on the frame
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls = int(box.cls)
            label = f"{results.names[cls]} {conf:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(frame)

        # Update progress
        frame_count += 1
        if frame_count % 30 == 0:  # Update every 30 frames
            print(f"Processed {frame_count}/{total_frames} frames")

    # Release video objects
    video.release()
    out.release()

    print(f"Processed video saved to {output_video_path}")

if __name__ == "__main__":
    process_video()
