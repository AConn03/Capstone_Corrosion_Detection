import os
import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- CONFIGURATION ---
# Assuming the model folder is in the same directory as this script
MODEL_PATH = "./rust_and_crack_model_final_2026-03-31" 

# Optimized for GPU usage
GRID_SIZE = 2               
DEEP_SEARCH_MODE = True    
MIN_CONFIDENCE = 80         
WEIGHT_FACTOR = 0.1
MAX_IMAGE_SIZE = 480        

# --- MODEL INITIALIZATION ---
print(f"Loading the fine-tuned detector from: {MODEL_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# GTX 1050 Ti (Pascal) does not have Tensor Cores, so float32 is safer and faster
dtype = torch.float32 

if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

try:
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    except Exception:
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
    model.to(device, dtype=dtype)
    model.eval()

    label_dict = {k.lower(): v for k, v in model.config.label2id.items()}
    rust_idx = label_dict.get('rust', 1)
    crack_idx = label_dict.get('crack', 2)
    clean_idx = label_dict.get('clean', 0)

    print(f"✅ Model loaded successfully on {str(device).upper()}!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure you downloaded the 'rust_and_crack_model_final' folder and placed it next to this script.")
    exit(1)

# --- HIGH SPEED OPENCV INFERENCE FUNCTION ---
def analyze_frame_cv2(frame_bgr, grid_size, deep_search):
    """Pure OpenCV/NumPy pipeline for maximum video FPS."""
    height, width = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    grids_to_run = []
    if deep_search:
        g = grid_size
        while g > 0:
            grids_to_run.append(g)
            g -= 2
    else:
        grids_to_run = [grid_size]

    crops = []
    crop_metadata = []

    for current_grid in grids_to_run:
        step_x = width // current_grid
        step_y = height // current_grid

        for row in range(current_grid):
            for col in range(current_grid):
                left = col * step_x
                top = row * step_y
                right = width if col == current_grid - 1 else (col + 1) * step_x
                bottom = height if row == current_grid - 1 else (row + 1) * step_y

                cropped_img = frame_rgb[top:bottom, left:right]
                crops.append(cropped_img)
                crop_metadata.append((current_grid, left, top, right, bottom))

    # --- MINI-BATCHING FOR 4GB VRAM (GTX 1050 Ti) ---
    batch_size = 16
    all_probabilities = []

    with torch.inference_mode():
        for i in range(0, len(crops), batch_size):
            chunk = crops[i:i + batch_size]
            inputs = processor(images=chunk, return_tensors="pt").to(device, dtype=dtype)
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            all_probabilities.append(probs)

    probabilities = torch.cat(all_probabilities, dim=0)

    base_step_x = width // grid_size
    base_step_y = height // grid_size
    base_grid_scores = {}

    for row in range(grid_size):
        for col in range(grid_size):
            left = col * base_step_x
            top = row * base_step_y
            right = width if col == grid_size - 1 else (col + 1) * base_step_x
            bottom = height if row == grid_size - 1 else (row + 1) * base_step_y

            cx = left + (right - left) / 2
            cy = top + (bottom - top) / 2

            base_grid_scores[(row, col)] = {
                'rust': 0.0, 'crack': 0.0, 'clean': 0.0,
                'total_weight': 0.0, 'coords': (left, top, right, bottom)
            }

            for i, (g_size, c_left, c_top, c_right, c_bottom) in enumerate(crop_metadata):
                if c_left <= cx <= c_right and c_top <= cy <= c_bottom:
                    weight = 1.0 + ((grid_size - g_size) * WEIGHT_FACTOR)
                    base_grid_scores[(row, col)]['rust'] += (probabilities[i][rust_idx].item() * 100) * weight
                    base_grid_scores[(row, col)]['crack'] += (probabilities[i][crack_idx].item() * 100) * weight
                    base_grid_scores[(row, col)]['clean'] += (probabilities[i][clean_idx].item() * 100) * weight
                    base_grid_scores[(row, col)]['total_weight'] += weight

    line_width = max(1, width // 500)

    for (row, col), data in base_grid_scores.items():
        total_w = data['total_weight']
        avg_rust = data['rust'] / total_w
        avg_crack = data['crack'] / total_w
        left, top, right, bottom = data['coords']

        combined_defect_conf = avg_rust + avg_crack

        if combined_defect_conf >= MIN_CONFIDENCE:
            r_val = int((avg_rust / combined_defect_conf) * 255)
            b_val = int((avg_crack / combined_defect_conf) * 255)

            color_bgr = (b_val, 0, r_val)
            alpha = min(130, int((combined_defect_conf / 100.0) * 180)) / 255.0

            roi = frame_bgr[top:bottom, left:right]
            overlay_rect = np.full(roi.shape, color_bgr, dtype=np.uint8)
            cv2.addWeighted(overlay_rect, alpha, roi, 1 - alpha, 0, roi)
            cv2.rectangle(frame_bgr, (left, top), (right, bottom), color_bgr, line_width)

    return frame_bgr

# --- NATIVE WEBCAM STREAMING ---
def start_local_webcam():
    print("🔥 Warming up PyTorch... (Please wait)")
    dummy_cv2 = np.zeros((480, 640, 3), dtype=np.uint8)
    analyze_frame_cv2(dummy_cv2, GRID_SIZE, DEEP_SEARCH_MODE)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("\n📷 Connecting to Local Webcam...")
    # '0' is usually the default built-in webcam. Change to '1' or '2' if you have external USB cameras attached.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam. Please check your camera permissions or connections.")
        return

    print("🚀 Live feed connected! Press 'q' on your keyboard to exit.")
    
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Resize frame aggressively to maintain fast inference
        h, w = frame.shape[:2]
        if max(w, h) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(w, h)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        if frame_count == 0:
            print("⏳ Processing first frame... The video window will appear in just a moment.")

        # Run Inference
        analyzed_frame = analyze_frame_cv2(frame, GRID_SIZE, DEEP_SEARCH_MODE)

        # Display the resulting frame in a window
        cv2.imshow('Live AI Inspection (Press "q" to quit)', analyzed_frame)

        frame_count += 1

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate average streaming FPS
    elapsed = time.time() - start_time
    if frame_count > 0:
        print(f"\n📊 Session Complete: Processed {frame_count} frames at {frame_count/elapsed:.1f} FPS.")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_local_webcam()