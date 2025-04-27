# NoMoreServiceDogs
- Made for the AMD Pervasive AI contest 2024 - Being aware of their surroundings has always been a challenge for visually impared people, in consequence, they have been provided with service dogs to help them. This uses ai to solve this problem of independancy allowing visually impared people to navigate without any external help.
# Project Overview

## External Resources and Licenses
- This project makes use of the ResNet50 implementation from PyTorch and Torchvision (BSD 3-Clause License) ([Pytorch ResNet50 implementation]())
- This project makes use of the Faster R-CNN with ResNet-50 implementation from PyTorch and Torchvision (BSD 3-Clause License) ([Pytorch Faster R-CNN ResNet50 implementation]())
- This project utilizes HWMonitor (freely available software by CPUID) for hardware monitoring purposes ([HWMonitor's website]())
- The cover image of this project was made with StableDiffusionWeb (CreativeML Open RAIL-M License) ([StableDiffusionWeb website]())
- This project is using the "Daily objects around the world" dataset (CC0: Public Domain License) ([Daily Objects Dataset kaggle page]())

> **Note**: The app and scripts are in a zip file attached to the "code" section of this hackster project.

## Initial Project Presentation
This project aims to solve a main problem encountered by visually impaired individuals: their lack of independence. Without external help (including service dogs or human assistance), it is very hard for these individuals to navigate the world.

To solve this problem, I implemented:
1. An image classification model fine-tuned on the Daily Object Dataset ([Daily Objects Dataset kaggle page]()). I chose ResNet-50 which has proven its capabilities many times.
2. Text-to-speech functionality to make the user aware of their surroundings.

The advantage of using AMD AI technology is that it makes this technology portable and usable daily for visually impaired individuals thanks to fast execution times provided by the hardware. This contributes to making visually impaired people more independent as no internet access is required since everything is executed locally.

## Project Adjustments/Upgrades
I realized that while the project could tell users about their general area, it couldn't identify specific objects within these areas. Therefore, I implemented:
- A Faster R-CNN with ResNet-50 (pretrained on the COCOV1 dataset) to recognize individual objects within the scene.
- After exporting to ONNX, quantizing it, and running it on the NPU, I achieved satisfying execution times.

## Project Execution

### Model Finetuning
**Goal**: Fine-tune the chosen image classification model (ResNet50) for the project.

**Changes made**:
- Modified the classifier layer from a single linear layer to two linear layers with batch normalization and dropout in between to prevent overfitting.
- Fine-tuned on the "Places" part of the Daily Places Dataset (excluding "Objects" and "Abstract" parts).

**Data Preprocessing**:
- Steps to extract images from the dataset are visible in `environnementModel\dataset.py`.

**Training**:
- Used data augmentation to help the model generalize better.
- Optimal hyperparameters: learning rate of ~1e-5 with SGD (momentum and weight decay).
- Achieved accuracy: 80-90% on the Daily Places Dataset.

### Model Quantization
**ResNet-50 for Image Classification**:
- Initially considered Quantization Aware Training but found Post Training Quantization with ONNX to be nearly as effective with less time.
- Quantized using Vitis AI ONNX Quantizer's static quantization with recommended CNN configuration for IPU deployment.
- File: `environnementModel\ONNXVistisAIQuantization.py`

**ResNet-50-FasterRCNN for Object Detection**:
- Faced challenges during preprocessing and quantization.
- Had to skip symbolic shape during preprocessing and deviate from recommended CNN configuration.
- Result: Model doesn't run entirely on IPU but maintains good execution times and accuracy.
- File: `objectsModel\VistisAIQuantization.py`

### Deployment
1. Created a simple UI with PyQt5.
2. Loaded two ONNX inference sessions on the IPU.
3. Implemented functionality where:
   - At button press, a frame is captured by the camera and given to models for prediction.
   - Current general place is displayed on UI.
   - Detected objects are communicated via Text-to-Speech (if enabled).

## Final Results

### Execution Times
Tests on final application (Finetuned ResNet50 + FasterRCNN-ResNet50) using AMD Ryzen 9 7940HS:
- PTQ+IPU: 0.63 seconds
- FP32+CPU: 1.17 seconds

### Power Consumption
Tests on final application using HWMonitor ([HWMonitor's website]()):
- Model Running on IPU: 33.52W
- Model Running on CPU: 42.16W

## Usage Instructions

### Installation
1. Activate the AMD Ryzen Software v1.1 conda environment
2. Run `pip install -r "requirements.txt"` (file in root folder)

> **Note**: All files must be executed from the project root directory (`No More Service Dogs`) for relative paths to work.

### Scripts Overview
- Training script: `environnementModel\train.py`
  > **Note**: Dataset folder has been emptied to save space (only label names kept)
- Quantization script (finetuned ResNet50): `environnementModel\ONNXVistisAIQuantization.py`
- Inference script (finetuned ResNet50 on IPU): `environnementModel\inference.py`
- Unused PyTorch QAT script: `environnementModel\unused`
- FasterRCNN-ResNet50 quantization script: `objectsModel\VistisAIQuantization.py`
- FasterRCNN-ResNet50 inference script: `objectsModel\ONNX_IPU_Inference.py`

### Using the Final App
1. Run `FinalApp.py` to launch the UI
2. Press "Run detection" button to run detection
3. Use tickbox to enable TTS (output will also print to console)

> **Requirements**:
> - Camera must be plugged in to use this file
> - For inference on saved images, use `environnementModel\inference.py` and `objectsModel\ONNX_IPU_Inference.py`

> **Note**: The app and scripts are in a zip file attached to the "code" section of this hackster project.
