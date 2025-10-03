import cv2
import numpy as np
import random
import time
from ultralytics import YOLO

# --- Game Configuration ---
AIM_COLOR = (0, 0, 255)      # Red for a "hit"
DEFAULT_COLOR = (0, 255, 0)  # Green for a standard detection
FONT = cv2.FONT_HERSHEY_SIMPLEX
CROSSHAIR_COLOR = (255, 255, 255) # White crosshair
SPLASH_COLOR = (255, 255, 0)  # Cyan for muzzle flash
BLOOD_COLOR = (0, 0, 139)     # Dark red for blood

# --- Global variables ---
mouse_pos = (0, 0)
shooting = False
splash_effects = []
blood_effects = []
hit_effects = []
score = 0

def mouse_event_handler(event, x, y, flags, param):
    """Updates the global mouse position and handles shooting."""
    global mouse_pos, shooting, splash_effects
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        shooting = True
        # Add muzzle flash effect
        splash_effects.append({
            'pos': (x, y),
            'time': time.time(),
            'radius': 5,
            'max_radius': 30,
            'duration': 0.3
        })

def draw_crosshair(img, center, size=15, color=(255, 255, 255), thickness=1):
    """Draws a crosshair on the image."""
    x, y = center
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)
    # Add a small circle in the center for better aiming
    cv2.circle(img, center, 2, color, -1)

def draw_splash_effects(img, current_time):
    """Draws muzzle flash effects."""
    global splash_effects
    active_effects = []
    
    for effect in splash_effects:
        elapsed = current_time - effect['time']
        if elapsed < effect['duration']:
            # Calculate effect progress (0 to 1)
            progress = elapsed / effect['duration']
            
            # Create expanding circle with fading effect
            radius = int(effect['radius'] + (effect['max_radius'] - effect['radius']) * progress)
            alpha = 1.0 - progress
            
            # Draw multiple circles for better effect
            for i in range(3):
                circle_radius = radius - i * 3
                if circle_radius > 0:
                    intensity = int(255 * alpha * (1 - i * 0.3))
                    color = (intensity, intensity, 0)  # Yellow to white
                    cv2.circle(img, effect['pos'], circle_radius, color, 2)
            
            # Add some random sparks
            for _ in range(8):
                angle = random.uniform(0, 2 * np.pi)
                spark_dist = random.uniform(radius * 0.5, radius * 1.2)
                spark_x = int(effect['pos'][0] + spark_dist * np.cos(angle))
                spark_y = int(effect['pos'][1] + spark_dist * np.sin(angle))
                
                if 0 <= spark_x < img.shape[1] and 0 <= spark_y < img.shape[0]:
                    spark_intensity = int(255 * alpha * random.uniform(0.5, 1.0))
                    cv2.circle(img, (spark_x, spark_y), 1, (spark_intensity, spark_intensity, 0), -1)
            
            active_effects.append(effect)
    
    splash_effects = active_effects

def draw_blood_effects(img, current_time):
    """Draws blood splatter effects."""
    global blood_effects
    active_effects = []
    
    for effect in blood_effects:
        elapsed = current_time - effect['time']
        if elapsed < effect['duration']:
            progress = elapsed / effect['duration']
            alpha = 1.0 - progress
            
            # Draw blood splatters
            for splatter in effect['splatters']:
                intensity = int(139 * alpha)  # Dark red
                cv2.circle(img, splatter['pos'], splatter['size'], (0, 0, intensity), -1)
            
            active_effects.append(effect)
    
    blood_effects = active_effects

def draw_hit_effects(img, current_time):
    """Draws hit text effects."""
    global hit_effects
    active_effects = []
    
    for effect in hit_effects:
        elapsed = current_time - effect['time']
        if elapsed < effect['duration']:
            progress = elapsed / effect['duration']
            
            # Text moves up and fades
            y_offset = int(progress * 30)
            alpha = 1.0 - progress
            
            text_pos = (effect['pos'][0], effect['pos'][1] - y_offset)
            
            # Draw text with outline for better visibility
            text_color = (0, int(255 * alpha), 0) if effect['type'] == 'HIT' else (0, 0, int(255 * alpha))
            
            # Outline
            cv2.putText(img, effect['text'], (text_pos[0]-1, text_pos[1]-1), FONT, 1.2, (0, 0, 0), 3)
            cv2.putText(img, effect['text'], (text_pos[0]+1, text_pos[1]+1), FONT, 1.2, (0, 0, 0), 3)
            
            # Main text
            cv2.putText(img, effect['text'], text_pos, FONT, 1.2, text_color, 2)
            
            active_effects.append(effect)
    
    hit_effects = active_effects

def create_blood_effect(pos):
    """Creates a blood splatter effect at the given position."""
    splatters = []
    for _ in range(random.randint(5, 12)):
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(10, 25)
        splatter_x = int(pos[0] + distance * np.cos(angle))
        splatter_y = int(pos[1] + distance * np.sin(angle))
        size = random.randint(2, 6)
        splatters.append({'pos': (splatter_x, splatter_y), 'size': size})
    
    return {
        'splatters': splatters,
        'time': time.time(),
        'duration': 1.5
    }

def draw_ui(img, score, fps=0):
    """Draws game UI elements."""
    # Background for UI
    cv2.rectangle(img, (10, 10), (300, 80), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (300, 80), (255, 255, 255), 2)
    
    # Score
    cv2.putText(img, f"Score: {score}", (20, 35), FONT, 0.7, (0, 255, 0), 2)
    
    # Instructions
    cv2.putText(img, "Left Click to Shoot", (20, 55), FONT, 0.5, (255, 255, 255), 1)
    
    # FPS counter
    if fps > 0:
        cv2.putText(img, f"FPS: {fps:.1f}", (200, 35), FONT, 0.5, (255, 255, 0), 1)

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

print("Game started! Aim with the mouse and LEFT CLICK to shoot. Press 'q' to quit.")

# FPS calculation
fps_counter = 0
fps_timer = time.time()
current_fps = 0

while True:
    frame_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    current_time = time.time()

    # --- Perform detection ---
    # The '0' class in the COCO dataset is 'person'.
    # stream=True is more efficient for video feeds.
    # Force inference on CPU to avoid CUDA compatibility issues
    results = model(frame, classes=0, stream=True, device='cpu')

    hit_detected_this_frame = False
    target_hit = False

    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            
            box_color = DEFAULT_COLOR
            is_targeted = x1 < mouse_pos[0] < x2 and y1 < mouse_pos[1] < y2
            
            if is_targeted:
                box_color = AIM_COLOR
                hit_detected_this_frame = True
                
                # Check if shooting and hitting target
                if shooting:
                    target_hit = True
                    score += 10
                    
                    # Create blood effect
                    blood_center = (mouse_pos[0], mouse_pos[1])
                    blood_effects.append(create_blood_effect(blood_center))
                    
                    # Create hit text effect
                    hit_effects.append({
                        'text': f'+{10}',
                        'pos': (mouse_pos[0] + 30, mouse_pos[1] - 20),
                        'time': current_time,
                        'duration': 1.0,
                        'type': 'SCORE'
                    })

            # Draw the box and label with enhanced graphics
            thickness = 3 if is_targeted else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
            
            # Enhanced label background
            label_bg_color = (0, 0, 0) if not is_targeted else (0, 0, 128)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + 80, y1), label_bg_color, -1)
            cv2.putText(frame, "TARGET", (x1 + 5, y1 - 8), FONT, 0.5, box_color, 1)

    # Handle shooting effects
    if shooting:
        if target_hit:
            # Add hit effect
            hit_effects.append({
                'text': 'HIT!',
                'pos': (mouse_pos[0] + 20, mouse_pos[1]),
                'time': current_time,
                'duration': 0.8,
                'type': 'HIT'
            })
            print(f"HIT! Score: {score}")
        else:
            # Add miss effect  
            hit_effects.append({
                'text': 'MISS!',
                'pos': (mouse_pos[0] + 20, mouse_pos[1]),
                'time': current_time,
                'duration': 0.8,
                'type': 'MISS'
            })
        
        shooting = False  # Reset shooting flag

    # Draw all visual effects
    draw_splash_effects(frame, current_time)
    draw_blood_effects(frame, current_time)
    draw_hit_effects(frame, current_time)
    
    # Draw enhanced crosshair
    crosshair_color = (0, 255, 255) if hit_detected_this_frame else CROSSHAIR_COLOR
    draw_crosshair(frame, mouse_pos, size=20, color=crosshair_color, thickness=2)
    
    # Draw UI
    draw_ui(frame, score, current_fps)

    # Calculate FPS
    fps_counter += 1
    if current_time - fps_timer >= 1.0:
        current_fps = fps_counter
        fps_counter = 0
        fps_timer = current_time

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Game exited. Final Score: {score}")