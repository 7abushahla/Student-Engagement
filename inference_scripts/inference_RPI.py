#!/usr/bin/env python
# coding: utf-8

# In[14]:


import sys
import argparse
import time
import csv
import os
import cv2
import numpy as np
import subprocess
from tflite_runtime.interpreter import Interpreter

# ------------------------------------------------------------- #
# ------------------------- PARAMETERS ------------------------ #
# ------------------------------------------------------------- #

num_inferences = 1000
num_warmup_runs = 10  # Number of warmup runs
image_size = 128
model_path = 'big_model_int8.tflite'
image_path = 'cool-cars-1.jpg'
labels = ['Paper', 'Screen', 'Wonder']


# ------------------------------------------------------------- #
# --------------------- IMAGE LOADING UTILS ------------------- #
# ------------------------------------------------------------- #

def load_image(path, target_size):
    """Load and preprocess the image."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Error: Image not found.")
    image_resized = cv2.resize(image, (target_size, target_size))
    return np.asarray(image_resized, dtype=np.float32) / 255.0


# ------------------------------------------------------------- #
# ------------------------ MODEL LOADING ---------------------- #
# ------------------------------------------------------------- #


interpreter = Interpreter(model_path=model_path, num_threads=1)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



# Display input details including quantization parameters
print("Input Details with Quantization Parameters:")
for input_detail in input_details:
    print(f"  Name: {input_detail['name']}")
    print(f"  Shape: {input_detail['shape']}")
    print(f"  Data Type: {input_detail['dtype']}")
    print(f"  Quantization Parameters: {input_detail['quantization']}")
    print(f"  Quantization Scale: {input_detail['quantization_parameters']['scales']}")
    print(f"  Quantization Zero Points: {input_detail['quantization_parameters']['zero_points']}")
    print()

# Display output details including quantization parameters
print("\nOutput Details with Quantization Parameters:")
for output_detail in output_details:
    print(f"  Name: {output_detail['name']}")
    print(f"  Shape: {output_detail['shape']}")
    print(f"  Data Type: {output_detail['dtype']}")
    print(f"  Quantization Parameters: {output_detail['quantization']}")
    print(f"  Quantization Scale: {output_detail['quantization_parameters']['scales']}")
    print(f"  Quantization Zero Points: {output_detail['quantization_parameters']['zero_points']}")
    print()


# ------------------------------------------------------------- #
# ------------------------- INFERENCE ------------------------- #
# ------------------------------------------------------------- #

# Assuming load_image and load_tflite_model are defined elsewhere.

import numpy as np
import time

# Assuming load_image and load_tflite_model are defined elsewhere.

def preprocess_input(image, input_details):
    """Preprocess and quantize the input image for TFLite model."""
    input_data = np.expand_dims(image, axis=0)  # Add batch dimension
    if input_details[0]['dtype'] == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
    return input_data

def run_inference(interpreter, input_data, input_details, output_details):
    """Run inference and return predictions."""
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if output_details[0]['dtype'] == np.int8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
    return output_data

# ------------------------------------------------------------- #
# --------------------------- MAIN ---------------------------- #
# ------------------------------------------------------------- #

# Load the static image and preprocess it
print("Loading image...")
image = load_image(image_path, image_size)

# Load the TFLite model
# print("Loading TFLite model...")
# interpreter, input_details, output_details = load_tflite_model(model_path)

# Preprocess the input image for the model
print("Preprocessing image for TFLite model...")
input_data = preprocess_input(image, input_details)

# # Perform warmup inferences
# print(f"Running {num_warmup_runs} warmup inferences...")
# for _ in range(num_warmup_runs):
#     run_inference(interpreter, input_data, input_details, output_details)

# Perform actual inferences
print("Running actual inferences...")
inference_times = []
fps_values = []
for i in range(num_inferences):
    start_time = time.time()
    preds = run_inference(interpreter, input_data, input_details, output_details)
    inference_time = time.time() - start_time
    inference_times.append(inference_time)
    fps_values.append(1 / inference_time)

    # Get label and confidence
    pred_label = labels[np.argmax(preds)]
    pred_conf = np.amax(preds)
    print(f'Inference {i + 1}: [Label={pred_label}, Confidence={pred_conf:.4f}, Time={inference_time:.4f}s]')

# Calculate average inference time and standard deviation
avg_inference_time = np.mean(inference_times)
std_inference_time = np.std(inference_times)

# Calculate FPS and its standard deviation
avg_fps = np.mean(fps_values)
std_fps = np.std(fps_values)

print(f"\nAverage Inference Time: {avg_inference_time:.4f} seconds")
print(f"Standard Deviation of Inference Time: {std_inference_time:.4f} seconds")
print(f"Average FPS: {avg_fps:.4f}")
print(f"Standard Deviation of FPS: {std_fps:.4f}")

# Export inference times and FPS to CSV
csv_filename = "inference_results.csv"
print(f"Exporting results to {csv_filename}...")
try:
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Inference", "Time (s)", "FPS"])
        # Write each inference's results
        for i, (time, fps) in enumerate(zip(inference_times, fps_values), start=1):
            writer.writerow([i, time, fps])
    print(f"Results successfully saved to {csv_filename}.")
except Exception as e:
    print(f"Error saving results to CSV: {e}")