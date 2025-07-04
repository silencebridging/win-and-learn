import gradio as gr
import numpy as np
import mediapipe as mp
import joblib
from sklearn.preprocessing import LabelEncoder
import cv2
import time
from datetime import datetime
import os

# Load model and labels
try:
    model = joblib.load('mlp_tsl_static.pkl')
    le = LabelEncoder()
    # Fit the encoder with the same labels used during training
    le.fit([chr(i) for i in range(ord('A'), ord('Z') + 1)])
except FileNotFoundError:
    print("Error: Model file 'mlp_tsl_static.pkl' not found.")
    print("Please make sure the model file is in the same directory as the script.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Normalize landmarks function
def normalize_landmarks(landmarks):
    coords = np.array(landmarks).reshape(-1, 3).astype(np.float32)
    # Add a small epsilon to avoid division by zero
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    norm_coords = (coords - coords_min) / (coords_max - coords_min + 1e-6)
    return norm_coords.flatten().reshape(1, -1)

# Save to file function
def save_output_to_file(text):
    os.makedirs('sound', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'sound/output_{timestamp}.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Output saved to {path}")

# --- Gradio State Management ---
def initial_state():
    return {
        "prev_letter": "",
        "letter_hold_start": None,
        "last_seen_time": time.time(),
        "word": "",
        "sentence": ""
    }

# --- The Core Processing Function ---
def process_image(image, state):
    if image is None:
        return None, "", state

    frame = cv2.flip(image, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    current_letter = ""
    current_time = time.time()
    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        state['last_seen_time'] = current_time

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            try:
                X = normalize_landmarks(landmarks)
                pred_index = model.predict(X)[0]
                current_letter = le.inverse_transform([pred_index])[0]

                if current_letter == state['prev_letter']:
                    if state['letter_hold_start'] is None:
                        state['letter_hold_start'] = current_time
                    # Add letter to word after holding for 0.8 seconds
                    if current_time - state['letter_hold_start'] >= 0.8:
                        if not state['word'] or state['word'][-1] != current_letter:
                            state['word'] += current_letter
                        # Reset timer to prevent rapid-fire additions
                        state['letter_hold_start'] = None 
                else:
                    state['letter_hold_start'] = None

                state['prev_letter'] = current_letter

            except Exception as e:
                print(f"Prediction error: {e}")
    
    # Logic for handling no hand detection
    time_since_last = current_time - state['last_seen_time']
    # Add a space if hand is gone for 2 seconds
    if time_since_last >= 2 and state['word'] and not state['word'].endswith(" "):
        state['word'] += " "
        state['last_seen_time'] = current_time # Reset timer to prevent multiple spaces
    # Form a sentence if hand is gone for 5 seconds
    if time_since_last >= 5 and state['word'].strip():
        state['sentence'] += state['word'].strip() + " "
        state['word'] = ""
        state['last_seen_time'] = current_time # Reset timer

    display_text = f"Word: {state['word']}\nSentence: {state['sentence']}"
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return processed_frame, display_text, state

# --- UI Control Functions ---
def clear_text():
    return "", initial_state()

def save_text(state):
    full_sentence = (state['sentence'] + state['word']).strip()
    message = ""
    if full_sentence:
        save_output_to_file(full_sentence)
        message = f"Saved: '{full_sentence}'"
    else:
        message = "Nothing to save."
    
    return message, "", initial_state()

# --- Build the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="TSL - Bridging Silence") as demo:
    app_state = gr.State(value=initial_state())
    
    gr.Markdown("# TSL - Bridging Silence")
    gr.Markdown("Show a hand sign to the camera. Hold the sign to form a word. A space is added when the hand is removed for 2 seconds. A sentence is formed after 5 seconds of no hand.")

    with gr.Row():
        video_in = gr.Image(sources="webcam", streaming=True, height=480, width=640, label="Webcam Feed")
        with gr.Column():
            video_out = gr.Image(height=480, width=640, label="Processed Feed")
            text_out = gr.Textbox(label="Output", lines=4, interactive=False)
            save_status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        clear_btn = gr.Button("Clear Text")
        save_btn = gr.Button("Save & Reset")

    # Connect components to functions
    video_in.stream(process_image, inputs=[video_in, app_state], outputs=[video_out, text_out, app_state])
    clear_btn.click(clear_text, inputs=[], outputs=[text_out, app_state])
    save_btn.click(save_text, inputs=[app_state], outputs=[save_status, text_out, app_state])

if __name__ == "__main__":
    demo.launch()
