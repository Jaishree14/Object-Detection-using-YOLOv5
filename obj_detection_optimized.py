import cv2
import torch

# Load YOLOv5 model (assuming you have YOLOv5 cloned as mentioned previously)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the image from your computer
image_path = 'img3.jpg'  # Path to the uploaded image
frame = cv2.imread(image_path)

# Define the coordinates for the "bin" area (adjust these coordinates as needed)
BIN_X, BIN_Y, BIN_WIDTH, BIN_HEIGHT = 100, 100, 400, 300  # Adjust based on your setup

# Crop the frame to only include the bin area
bin_frame = frame[BIN_Y:BIN_Y+BIN_HEIGHT, BIN_X:BIN_X+BIN_WIDTH]

# Perform object detection within the bin area with an increased confidence threshold
results = model(bin_frame)
results = results.pandas().xyxy[0]  # Convert results to Pandas dataframe for easier handling
filtered_results = results[results['confidence'] > 0.7]  # Increase confidence threshold to 0.7

# Only keep detections that are fully within the bin area
for _, row in filtered_results.iterrows():
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    
    # Ensure the bounding box is within bin boundaries
    if x1 >= 0 and y1 >= 0 and x2 <= BIN_WIDTH and y2 <= BIN_HEIGHT:
        label = f"{row['name']} {row['confidence']:.2f}"
        cv2.rectangle(bin_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(bin_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Place the bin_frame back into the main frame for display
frame[BIN_Y:BIN_Y+BIN_HEIGHT, BIN_X:BIN_X+BIN_WIDTH] = bin_frame

# Draw a rectangle around the bin area in the main frame (optional)
cv2.rectangle(frame, (BIN_X, BIN_Y), (BIN_X + BIN_WIDTH, BIN_Y + BIN_HEIGHT), (255, 0, 0), 2)
cv2.putText(frame, "Detection Zone", (BIN_X, BIN_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Display the optimized output
cv2.imshow("Optimized Object Detection in Bin", frame)

# Press 'q' to quit
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
