import sys
import argparse
import time
import csv
import os
import cv2
import numpy as np
import threading
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

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
ap.add_argument("model_name", type=str, help="Name of the TensorRT model (with .trt extension)")
args = vars(ap.parse_args())

model_name = args["model_name"]
time_csv_file = f"Orin_trt_{model_name}_time.csv"
power_csv_file = f"Orin_trt_{model_name}_power.csv"

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

def load_engine(engine_file_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, "rb") as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine, context):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for tensor_name in engine:
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        shape = context.get_tensor_shape(tensor_name)
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append((host_mem, device_mem, tensor_name))
        else:
            outputs.append((host_mem, device_mem, tensor_name))
    return inputs, outputs, bindings, stream

print("Loading TensorRT Model...")
engine = load_engine(model_name)
context = engine.create_execution_context()
print("Model Loaded")

# Automatically determine input and output tensor names
input_names = [name for name in engine if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
output_names = [name for name in engine if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
if len(input_names) != 1 or len(output_names) != 1:
    print("Error: Model must have exactly one input and one output tensor.")
    sys.exit(1)
INPUT_NAME = input_names[0]
OUTPUT_NAME = output_names[0]

inputs, outputs, bindings, stream = allocate_buffers(engine, context)

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
    bin_input_img = np.expand_dims(bin_input_img, axis=0)  # Add batch dimension

    # Copy input to memory
    np.copyto(inputs[0][0], bin_input_img.ravel())

    # Set input and output tensor addresses
    context.set_tensor_address(INPUT_NAME, int(inputs[0][1]))
    context.set_tensor_address(OUTPUT_NAME, int(outputs[0][1]))

    time_start_inference = time.time()
    context.execute_async_v3(stream_handle=stream.handle)
    time_end_inference = time.time()

    # Retrieve output
    cuda.memcpy_dtoh(outputs[0][0], outputs[0][1])
    preds = outputs[0][0]
    pred_label = labels[np.argmax(preds)]
    pred_conf = np.amax(preds)

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
