
import sys
import argparse
import time
import csv
import os
import cv2
import numpy as np
import subprocess
from tflite_runtime.interpreter import Interpreter
import threading

# ------------------------------------------------------------- #
# ------------------------- PARAMETERS ------------------------ #
# ------------------------------------------------------------- #

num_inferences = 1001
image_size = 128

# ------------------------------------------------------------- #
# --------------------------- GLOBALS ------------------------- #
# ------------------------------------------------------------- #

labels = ['Paper', 'Screen', 'Wonder']
time_start_preprocess = 0
time_start_inference = 0
time_end_inference = 0
inference_fnames = ['#', 'start_preprocess', 'start_inference', 'end_inference']

# ------------------------------------------------------------- #
# ----------------------- ARGUMENT PARSING -------------------- #
# ------------------------------------------------------------- #

ap = argparse.ArgumentParser()
ap.add_argument("model_name", type=str, help="Name of the model (without .tflite extension)")
args = vars(ap.parse_args())

model_name = args["model_name"]
time_csv_file = f"Orin_{model_name}_time.csv"
power_csv_file = f"Orin_{model_name}_power.csv"

# ------------------------------------------------------------- #
# --------------------- IMAGE LOADING UTILS ------------------- #
# ------------------------------------------------------------- #

def load_image(path):
    # Load the image from the specified path
    image = cv2.imread(path)
    if image is None:
        print("Error: Image not found.")
        sys.exit(1)
    return image

def write_inference_timestamp(i):
    global time_start_preprocess, time_start_inference, time_end_inference

    inference_writer.writerow({
        inference_fnames[0]: i,
        inference_fnames[1]: time_start_preprocess,
        inference_fnames[2]: time_start_inference,
        inference_fnames[3]: time_end_inference
    })

    print(f'Inference {i} timestamps saved to CSV: Preprocess={time_start_preprocess:.2f}, Inference Start={time_start_inference:.2f}, Inference End={time_end_inference:.2f}')

# ------------------------------------------------------------- #
# -------------------- POWER MONITORING ----------------------- #
# ------------------------------------------------------------- #

def monitor_power():
    # Redirects the full output of `tegrastats` to a text file
    with open(power_csv_file, 'w') as power_file:
        with subprocess.Popen(['tegrastats'], stdout=power_file, text=True) as proc:
            try:
                proc.wait()  # Keep the process running until the main program stops
            except KeyboardInterrupt:
                proc.terminate()

# ------------------------------------------------------------- #
# ------------------------ MODEL LOADING ---------------------- #
# ------------------------------------------------------------- #

print("Loading Model")
interpreter = Interpreter(f"{model_name}.tflite")
print("Model Loaded")
interpreter.allocate_tensors()

input1_details = interpreter.get_input_details()
output1_details = interpreter.get_output_details()

# Start power monitoring in a separate thread
power_thread = threading.Thread(target=monitor_power, daemon=True)
power_thread.start()

# ------------------------------------------------------------- #
# ------------------------ CLASSIFICATION --------------------- #
# ------------------------------------------------------------- #

def classify(image):
    global time_start_preprocess, time_start_inference, time_end_inference

    time_start_preprocess = time.time()

    # Preprocess image: Resize and normalize
    image_bin = cv2.resize(image, (image_size, image_size))
    bin_input_img = np.asarray(image_bin, dtype=np.float32) / 255.0

    # Expand dimensions to add batch size
    bin_input_img = np.expand_dims(bin_input_img, axis=0)  # Add batch dimension

    if input1_details[0]['dtype'] == np.int8:
        bin_input_img = (bin_input_img * 127.5 - 1).astype(np.int8)

    # Set the tensor for the model
    interpreter.set_tensor(input1_details[0]['index'], bin_input_img)

    time_start_inference = time.time()
    interpreter.invoke()
    preds = interpreter.get_tensor(output1_details[0]['index'])
    pred_label = labels[np.argmax(preds)]
    pred_conf = np.amax(preds)

    time_end_inference = time.time()

    return pred_label, pred_conf


# ------------------------------------------------------------- #
# --------------------------- MAIN ---------------------------- #
# ------------------------------------------------------------- #

# Load the static image
image_path = 'cool-cars-1.jpg'
img_input = load_image(image_path)

# Open CSV for writing inference times
with open(time_csv_file, 'w') as inference_file:
    inference_writer = csv.DictWriter(inference_file, fieldnames=inference_fnames)
    inference_writer.writeheader()

    for i in range(num_inferences):
        # Classify the loaded static image
        pred_label, pred_conf = classify(img_input)

        print(f'Inference {i}: [Label={pred_label}, Confidence={pred_conf:.4f}]')
        write_inference_timestamp(i)
