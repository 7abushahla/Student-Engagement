
import os
import csv
import time
import argparse
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

# Define label map
label_map = {0: "Paper", 1: "Screen", 2: "Wander"}

# Labels and CSV header
inference_fnames = ['#', 'start_preprocess', 'start_inference', 'end_inference']

def preprocess_image(image_file, interpreter):
    """Preprocess the fixed test image to match the model's input size."""
    global time_start_preprocess
    time_start_preprocess = time.time()  # Record preprocessing start time

    size = common.input_size(interpreter)
    image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)
    return image

def run_inference(interpreter, image, num_runs=1000):
    """Run inference on the image data multiple times and measure times."""
    global time_start_inference, time_end_inference
    results = []

    for i in range(num_runs):  # Repeat inference for performance testing
        # Record inference start time
        time_start_inference = time.time()

        # Set input and run inference
        common.set_input(interpreter, image)
        interpreter.invoke()

        # Get the top prediction
        classes = classify.get_classes(interpreter, top_k=1)
        label = label_map.get(classes[0].id, "Unknown")
        score = classes[0].score

        # Print result of the prediction
        print(f"Inference {i}: Label = {label}, Score = {score:.5f}")

        # Record inference end time
        time_end_inference = time.time()

        # Append inference timing data
        results.append([
            i, 
            time_start_preprocess, 
            time_start_inference, 
            time_end_inference
        ])

    return results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inference on Coral TPU with a specified model.")
    parser.add_argument("model_name", type=str, help="Name of the model (without .tflite extension)")
    args = parser.parse_args()

    model_name = args.model_name
    model_file = f"{model_name}.tflite"
    image_file = "cool-cars-1.jpg"  # Replace with the exact test image name
    output_file = f"{model_name}_time.csv"

    try:
        # Initialize the interpreter
        interpreter = edgetpu.make_interpreter(model_file)
        interpreter.allocate_tensors()

        # Preprocess the fixed image
        image = preprocess_image(image_file, interpreter)

        # Run inference and collect results
        results = run_inference(interpreter, image)
        
        # Save results to CSV
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(inference_fnames)  # Write CSV header
            writer.writerows(results)         # Write inference timing data

        print(f"Inference completed. Results saved to {output_file}")
    except FileNotFoundError:
        print(f"Model file {model_file} or the test image {image_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
