import cv2
from ultralytics import YOLO

# --- Game Configuration ---
AIM_COLOR = (0, 0, 255)      # Red for a "hit"
DEFAULT_COLOR = (0, 255, 0)  # Green for a standard detection
FONT = cv2.FONT_HERSHEY_SIMPLEX
CROSSHAIR_COLOR = (255, 255, 255) # White crosshair

# --- Global variable for mouse position ---
mouse_pos = (0, 0)

def mouse_event_handler(event, x, y, flags, param):
    """Updates the global mouse position."""
    global mouse_pos
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)

def draw_crosshair(img, center, size=15, color=(255, 255, 255), thickness=1):
    """Draws a crosshair on the image."""
    x, y = center
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)

# --- Load the YOLOv8 model ---
# The model will be downloaded automatically the first time you run it.
# 'yolov8n.pt' is the smallest and fastest model, perfect for real-time.
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')

# Force model to use CPU to avoid CUDA compatibility issues with MX330
import torch
device = 'cpu'
model.to(device)
print(f"Model loaded successfully on device: {device}")

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

window_name = "First Person Game (YOLOv8)"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_event_handler)

print("Game started! Aim with the mouse. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # --- Perform detection ---
    # The '0' class in the COCO dataset is 'person'.
    # stream=True is more efficient for video feeds.
    # Force inference on CPU to avoid CUDA compatibility issues
    results = model(frame, classes=0, stream=True, device='cpu')

    hit_detected_this_frame = False

    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            
            box_color = DEFAULT_COLOR
            # Check if the mouse cursor is inside this box
            if x1 < mouse_pos[0] < x2 and y1 < mouse_pos[1] < y2:
                box_color = AIM_COLOR
                hit_detected_this_frame = True

            # Draw the box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), FONT, 0.7, box_color, 2)

    if hit_detected_this_frame:
        print("HIT!")
        cv2.putText(frame, "HIT!", (mouse_pos[0] + 20, mouse_pos[1]), FONT, 0.8, AIM_COLOR, 2)

    draw_crosshair(frame, mouse_pos, color=CROSSHAIR_COLOR)
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Game exited.")