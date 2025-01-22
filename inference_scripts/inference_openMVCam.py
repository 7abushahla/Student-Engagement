import sensor
import time
import ml
import gc
import os
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((128, 128))
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)
print(os.uname())
gc.collect()
mem_before_loading = gc.mem_free()
print(f"Free memory before loading model: {mem_before_loading / 1024:.2f} KB")
simple_model = "small_model_int8.tflite"
try:
	model = ml.Model(simple_model, load_to_fb=True)
	mem_after_loading = gc.mem_free()
	print(f"Free memory after loading model: {mem_after_loading / 1024:.2f} KB")
except MemoryError:
	print("Error: Not enough memory to load the model. Consider using a smaller model or reducing input size.")
	raise SystemExit
except OSError:
	print("Error: Could not find the model file. Please check the file path and ensure the model is on the device.")
	raise SystemExit
try:
	labels = [line.rstrip("\n") for line in open("labels.txt")]
except OSError:
	print("Error: Labels file not found.")
	raise SystemExit
clock = time.clock()
inference_times = []
fps_values = []
num_inferences = 1000
for i in range(1, num_inferences + 1):
	clock.tick()
	gc.collect()
	img = sensor.snapshot()
	gc.collect()
	try:
		start_time = time.ticks_ms()
		predictions = model.predict([img])[0].flatten().tolist()
		end_time = time.ticks_ms()
		inference_time = (end_time - start_time) / 1000.0
		fps = clock.fps()
		inference_times.append(inference_time)
		fps_values.append(1/inference_time)
		sorted_list = sorted(zip(labels, predictions), key=lambda x: x[1], reverse=True)
		label = sorted_list[0][0]
		confidence = sorted_list[0][1]
		print(f"Inference {i}: [Label={label}, Confidence={confidence:.4f}, Time={inference_time:.4f}s, FPS={fps:.2f}]")
	except MemoryError:
		print("Error: Not enough memory to run inference.")
	finally:
		img = None
		gc.collect()
def calculate_mean(data):
	return sum(data) / len(data)
def calculate_std_dev(data, mean):
	return (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
average_inference_time = calculate_mean(inference_times)
std_dev_inference_time = calculate_std_dev(inference_times, average_inference_time)
average_fps = calculate_mean(fps_values)
std_dev_fps = calculate_std_dev(fps_values, average_fps)
print(f"Average Inference Time: {average_inference_time:.4f}s")
print(f"Standard Deviation of Inference Time: {std_dev_inference_time:.4f}s")
print(f"Average FPS: {average_fps:.4f}")
print(f"Standard Deviation of FPS: {std_dev_fps:.4f}")
try:
	with open("inference_results.txt", "w") as f:
		f.write("Time(s)	FPS\n")
		for time, fps in zip(inference_times, fps_values):
			f.write(f"{time:.4f}	{fps:.2f}\n")
	print("Inference results saved to inference_results.txt")
	with open("inference_results.txt", "r") as f:
		lines = f.readlines()
		print("First few lines of the file:")
		print("".join(lines[:5]))
except OSError as e:
	print(f"Error writing to TXT file: {e}")