import cv2
import torch

# Load YOLOv5 model (assuming you have YOLOv5 cloned as mentioned previously)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the image from your computer
image_path = 'img1.jpg'  # Replace with the path to your image file
frame = cv2.imread(image_path)

# Define the coordinates for the "bin" area
# These coordinates will define the specific area of the frame where the bin is located
# Adjust these based on your specific bin setup
BIN_X, BIN_Y, BIN_WIDTH, BIN_HEIGHT = 100, 100, 400, 300  # Example values, adjust as needed

# Crop the frame to only include the bin area
bin_frame = frame[BIN_Y:BIN_Y+BIN_HEIGHT, BIN_X:BIN_X+BIN_WIDTH]

# Perform object detection within the bin area
results = model(bin_frame)

# Draw bounding boxes and labels on the bin area frame
for *box, conf, cls in results.xyxy[0]:
    x1, y1, x2, y2 = map(int, box)
    label = f"{model.names[int(cls)]} {conf:.2f}"
    cv2.rectangle(bin_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(bin_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Place the bin_frame back into the main frame for display
frame[BIN_Y:BIN_Y+BIN_HEIGHT, BIN_X:BIN_X+BIN_WIDTH] = bin_frame

# Draw a rectangle around the bin area in the main frame (optional)
cv2.rectangle(frame, (BIN_X, BIN_Y), (BIN_X + BIN_WIDTH, BIN_Y + BIN_HEIGHT), (255, 0, 0), 2)
cv2.putText(frame, "Detection Zone", (BIN_X, BIN_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Display the output
cv2.imshow("Object Detection in Bin", frame)

# Press 'q' to quit
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
