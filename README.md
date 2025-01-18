# Real-Time Student Engagement Monitoring on Edge Devices

This repository contains code and resources for the paper: [**Real-Time Student Engagement Monitoring on Edge Devices: Deep Learning Meets Efficiency and Privacy**](url)

**Authors**: Hamza A. Abushahla, Lodan Elmugamer, Rana Gharaibeh, Ali Reza Sajun, Imran A. Zualkernan

### What’s Included:
-
-

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
4. [Quantization and Deployment](#4-quantization-and-deployment)
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

A folder with **sample images**—one from each class—is available in the [`data/sample_images/`](data/sample_images/) directory.

### Data Preprocessing

The dataset underwent several preprocessing steps to prepare it for model training, implemented within a Jupyter notebook:

- **Resizing and Normalization**
  - Images were resized to **128x128 pixels**.
  - Pixel intensity values were normalized to the range [0, 255].

- **Augmentation Techniques**
  - **Gaussian Noise**: Applied to add randomness and improve generalization.
  - **Color Channel Changes**: Adjusted hue, saturation, and brightness to enhance robustness.
  - **Random Cropping**: Introduced variability in spatial dimensions.

These preprocessing steps are integrated into the [`notebooks/data_preprocessing.ipynb`](notebooks/data_preprocessing.ipynb) Jupyter notebook, ensuring consistent augmentation across training and evaluation.

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

Both models are fully fine-tuned on the Student Engagement Dataset, and the training scripts, along with configurations, are available in the [`src/models/`](src/models/) directory.

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
- **Training Script**: Available in the [`src/models/custom_model.py`](src/models/custom_model.py).

---

## 4. Quantization and Deployment

---

## 5. Running Experiments and Taking Measurements

This section provides detailed instructions on how to run experiments and take measurements for inference time and power consumption.

### Inference Time

...

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
- The recorded CSV file is saved in the experiments/ directory.

#### Resources
- [Yocto-Amp User Manual](https://www.yoctopuce.com/EN/products/yocto-amp/doc/YAMPMK01.usermanual.html)

---
## 6. Citation and Reaching Out

### Citation
If you use this repository or its contents in your work, please cite our paper:
```bibtex

```

### Contact
If you have any questions, please feel free to reach out to me through email ([b00090279@alumni.aus.edu](mailto:b00090279@alumni.aus.edu)) or by connecting with me on [LinkedIn](https://www.linkedin.com/in/hamza-abushahla/).





