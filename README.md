# Real-Time Student Engagement Monitoring on Edge Devices

This repository contains code and resources for the paper: [**Real-Time Student Engagement Monitoring on Edge Devices: Deep Learning Meets Efficiency and Privacy**](https://ieeexplore.ieee.org/document/11016436)

**Authors**: Hamza A. Abushahla, Lodan Elmugamer, Rana Gharaibeh, Ali Reza Sajun, Imran A. Zualkernan

---

## Table of Contents

1. [Hardware Platforms](#1-hardware-platforms)
   1. [Low-Power Microcontrollers](#low-power-microcontrollers)
   2. [Mid-Range Edge Devices](#mid-range-edge-devices)
   3. [High-End Edge Accelerators](#high-end-edge-accelerators)
2. [Dataset](#2-dataset)
   1. [The Student Engagement Dataset](#the-student-engagement-dataset)
   3. [Data Preprocessing](#data-preprocessing)
3. [Models and Training](#3-models-and-training)
   1. [Model Architectures](#model-architectures)
      1. [Original Model](#original-model)
      2. [Lightweight Model](#lightweight-model)
   2. [Model Training Details](#model-training-details)
4. [Quantization and Deployment](#4-quantization-and-deployment)
   1. [Quantization](#1-quantization)
   2. [Deployment](#2-deployment)
5. [Running Experiments and Taking Measurements](#5-running-experiments-and-taking-measurements)
   1. [Inference Time](#inference-time)
   2. [Power Consumption](#power-consumption)
6. [Citation and Reaching Out](#6-citation-and-reaching-out)
   1. [Citation](#citation)
   2. [Contact](#contact)

---

## 1. Hardware Platforms

This study evaluates the deployment of deep learning models across a diverse range of hardware platforms, categorized into three levels based on their specifications and capabilities:

### Low-Power Microcontrollers
- **Sony Spresense**
- **OpenMV Cam H7**
- **OpenMV Cam H7 Plus**

### Mid-Range Edge Devices
- **Raspberry Pi Zero 2 W**
- **Google Coral Dev Board**
- **Raspberry Pi 5**

### High-End Edge Accelerators
- **NVIDIA Jetson Orin Nano**
- **NVIDIA Jetson AGX Orin**

Each category offers a different balance of performance, power consumption, and cost, allowing for a comprehensive analysis of trade-offs in real-time scenarios.

---

## 2. Dataset

### The Student Engagement Dataset

The [Student Engagement Dataset](https://ieeexplore.ieee.org/document/9607578) employed in this study is a frame-level dataset comprising approximately **19,000 frames** across three distinct behavior classes: **Looking at Screen**, **Looking at Paper**, and **Wandering**

A folder with **sample images**—one from each class—is available in the [`dataset/`](dataset/) directory.

### Data Preprocessing

The dataset underwent several preprocessing steps to prepare it for model training, implemented within a Jupyter notebook:

- **Resizing and Normalization**
  - Images were resized to **128x128 pixels**.
  - Pixel intensity values were normalized to the range [0, 255].

- **Augmentation Techniques**
  - **Gaussian Noise**: Applied to add randomness and improve generalization.
  - **Color Channel Changes**: Adjusted hue, saturation, and brightness to enhance robustness.
  - **Random Cropping**: Introduced variability in spatial dimensions.

These preprocessing steps are integrated into the [`notebooks/train_and_quantize.ipynb`](notebooks/train_and_quantize.ipynb) Jupyter notebook, ensuring consistent augmentation across training and evaluation.

---

## 3. Models and Training

We developed custom deep learning models based on MobileNet architectures. Below are the details of the models and their training processes:

### Model Architectures

#### Original Model

- **Backbone**: MobileNetV2 pretrained on ImageNet.
- **Additional Layers**:
  - Global Average Pooling 2D
  - Dense Layer with 128 neurons and ReLU activation
  - Batch Normalization
  - Output Layer with 3 neurons and Softmax activation

#### Lightweight Model

- **Backbone**: MobileNetV1 pretrained on ImageNet with `alpha=0.25` for compactness.
- **Additional Layers**: Same layers in the original model.

Both models are fully fine-tuned on the Student Engagement Dataset, and the training scripts, along with configurations, are available in the [`models/`](models/) directory.

### Model Training Details

**General Training Hyperparameters**

| Parameter             | Value                                     |
| --------------------- | ----------------------------------------- |
| Optimizer             | Stochastic Gradient Descent (SGD)         |
| Loss Function         | Categorical Cross-Entropy                 |
| Learning Rate         | 0.01                                      |
| Batch Size            | 16                                        |
| Number of Epochs      | 100                                       |
| Dropout Rate          | 0.6                                       |
| L2-Regularization     | 0.01                                      |

**Early Stopping Parameters**

| Parameter               | Value               |
| ----------------------- | ------------------- |
| Early Stopping Metric   | Validation Loss     |
| Early Stopping Patience | 5 epochs            |

**Training Process**

- **K-Fold Cross-Validation**: 5-fold cross-validation approach to ensure robustness and generalization.
- **Training Script**: Available at [`notebooks/train_and_quantize.ipynb`](notebooks/train_and_quantize.ipynb).

---

## 4. Quantization and Deployment

### 1. Quantization

After training both models, the best-validated models were saved and subsequently fine-tuned using Quantization-Aware Training (QAT) for 10 epochs with the TensorFlow Model Optimization Toolkit (TFMOT). Post fine-tuning, the models were fully quantized to INT8, including weights, activations, inputs, and outputs. The fully quantized models were saved in TensorFlow Lite (.tflite) format to facilitate deployment on hardware devices.

These steps are performed in the [`notebooks/train_and_quantize.ipynb`](notebooks/train_and_quantize.ipynb) Jupyter notebook.

**Model Variants:**

- **Big Model (MobileNetV2-FP32)**
  - **File Available At**: [`models/big_model.tflite`](models/big_model.tflite)

- **Big Model INT8 (MobileNetV2-INT8)**
  - **File Available At**: [`models/big_model_int8.tflite`](models/big_model_int8.tflite)

- **Small Model (MobileNetV1-FP32)**
  - **File Available At**: [`models/small_model.tflite`](models/small_model.tflite)
    
- **Small Model INT8 (MobileNetV1-INT8)**
  - **File Available At**: [`models/small_model_int8.tflite`](models/small_model_int8.tflite)

The INT8 versions (`big_model_int8.tflite`, `small_model_int8.tflite`) are the fully quantized models optimized for deployment on edge devices.


### 2. Deployment

Below are the detailed deployment processes for each hardware platform:

#### Sony Spresense

Deployment on Sony Spresense involves converting the `.tflite` model to a byte array and integrating it into embedded C code using TensorFlow Lite for Microcontrollers (TFLM). Detailed steps are provided in our [Sony Spresense TFLite Guide](https://github.com/7abushahla/Sony-Spresense-TFLite-Guide).

1. **Model Conversion**
   - Convert the `.tflite` model to a C-style header (`.h`) file.
   - **File Available At**: [`models/small_model_int8.h`](models/small_model_int8.h)

2. **Embedded C Integration**
   - Integrate the `.h` file into an embedded C codebase using TFLM.
   - The embedded C code is an Arduino sketch (`.ino` file).
   - **File Available At**: [`models/inference_sony.ino`](models/inference_sony.ino)

3. **Flashing the Device**
   - Flash the compiled code onto the Sony Spresense device's memory using the Arduino IDE.

#### OpenMV Cam H7 and H7 Plus

Deployment on OpenMV Cam H7 and H7 Plus utilizes MicroPython to run inference programs through the OpenMV IDE.

1. **Inference Script**
   - **Inference Script Available At**: [`inference_scripts/inference_openmv.py`](inference_scripts/inference_openmv.py)
   - This script loads the `.tflite` model and performs inference.

2. **Running Inference**
   - Use the OpenMV IDE to upload and execute the script on the device.

#### Google Coral Dev Board

Deployment on the Google Coral Dev Board involves compiling the `.tflite` model for the Edge TPU and performing inference using the PyCoral API.

1. **Model Compilation**
   - Use the Edge TPU Compiler to convert the `.tflite` model into a format compatible with the Edge TPU.

2. **Inference Script**
   - **Inference Script Available At**: [`inference_scripts/inference_coral.py`](inference_scripts/inference_coral.py)
   - This script utilizes the PyCoral API to run inference on the compiled model.

3. **Running Inference**
   - Execute the script on the Coral Dev Board to perform real-time inference.

#### Linux-Based Devices (Raspberry Pi, NVIDIA Edge Accelerators)

For devices running a Linux OS with Python support, including Raspberry Pi and NVIDIA edge accelerators, the TFLite interpreter is used to run inference directly using Python scripts.

1. **Inference Scripts**
   - **Raspberry Pi Inference Script Available At**: [`inference_scripts/inference_raspberry_pi.py`](inference_scripts/inference_raspberry_pi.py)
   - **NVIDIA Jetson Inference Script Available At**: [`inference_scripts/inference_jetson.py`](inference_scripts/inference_jetson.py)

2. **Running Inference with TFLite**
   - Use the respective Python scripts to load the `.tflite` model and perform inference.

3. **NVIDIA Jetson Orin Nano and AGX Orin Optimization**
   - For optimized performance on NVIDIA Jetson devices, the models were converted to the ONNX format and then to TensorRT format.
   - **Files Available At**:
     - **ONNX Files**:
       - [`models/big_model.onnx`](models/big_model.onnx)
       - [`models/big_model_int8.onnx`](models/big_model_int8.onnx)
       - [`models/small_model.onnx`](models/small_model.onnx)
       - [`models/small_model_int8.onnx`](models/small_model_int8.onnx)
     - **TensorRT Files**:
       - [`models/big_model.trt`](models/big_model.trt)
       - [`models/small_model.trt`](models/small_model.trt)

   - These optimized models leverage GPU acceleration for faster inference.

   - **TensorRT Inference Scripts Available At**:
     - [`inference_scripts/trt_inference_jetson.py`](inference_scripts/trt_inference_jetson.py)
     - [`inference_scripts/trt10_singleinf.py`](inference_scripts/trt10_singleinf.py)

---

## 5. Running Experiments and Taking Measurements

This section provides detailed instructions on how to run experiments and take measurements for inference time and power consumption.

### Inference Time

Inference time is measured using the provided inference scripts in the previous section. Each script runs the model for **1,000 inferences** and reports the mean and standard deviation of the inference times in **milliseconds (ms)**. Additionally, Frames Per Second (FPS) is calculated as the inverse of the mean inference time (FPS = 1 / time).

The results are saved in the [`results/inference_time/`](results/inference_time/) directory as a csv file for each model varient on all edge devices.

### Power Consumption
Power consumption measurements are performed using the **[Yocto-Amp](https://www.yoctopuce.com/EN/products/usb-electrical-sensors/yocto-amp)** current sensor. Below are the step-by-step instructions:

1. **Hardware Setup**
- Yocto-Amp Setup:
  - Cut the power adapter of the target edge device (e.g., Raspberry Pi).
  - Connect the Yocto-Amp in series between the power adapter and the edge device.
  - Connect the Yocto-Amp's USB to your computer.
  
2. **Software Configuration**
- **VirtualHub**
  - Download and install the [VirtualHub](https://www.yoctopuce.com/EN/virtualhub.php) software.
  - Open VirtualHub then go to `localhost:4444` and then configure the following settings:
    - **Frequency:** Set the sampling frequency to 100/s
    - **Data Logging:** Enable data logging to record current measurements.

3. **Recording Data**
- Start the measurement in VirtualHub.
- Run the 1000 inferences on the edge device
- Stop the measurement after completion.
- Download the recorded current readings as a CSV file.

4. **Calculating Power Consumption**
- Multiply the average current by the input voltage to obtain the average power consumption in mW.
- The recorded CSV files are saved in the [`results/power_consumption/`](results/power_consumption/) directory.

#### Resources
- [Yocto-Amp User Manual](https://www.yoctopuce.com/EN/products/yocto-amp/doc/YAMPMK01.usermanual.html)

---
## 6. Citation and Reaching Out

### Citation
If you use this repository or its contents in your work, please cite our paper:
```bibtex
@INPROCEEDINGS{Abushahla2025,
  author    = {Abushahla, Hamza A. and Gharaibeh, Rana and Elmugamer, Lodan and Reza Sajun, Ali and Zualkernan, Imran A.},
  title     = {Real-Time Student Engagement Monitoring on Edge Devices: Deep Learning Meets Efficiency and Privacy},
  booktitle = {Proceedings of the 2025 IEEE Global Engineering Education Conference (EDUCON)},
  year      = {2025},
  pages     = {1--9},
  doi       = {10.1109/EDUCON62633.2025.11016436}
}

```

### Contact
If you have any questions, please feel free to reach out to me through email ([b00090279@alumni.aus.edu](mailto:b00090279@alumni.aus.edu)) or by connecting with me on [LinkedIn](https://www.linkedin.com/in/hamza-abushahla/).





