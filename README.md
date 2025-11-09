# Chest X-Ray Medical Diagnosis with Deep Learning

![Project Banner](images/xray-header-image.png) <!-- Replace with actual banner image if available -->

This repository contains a deep learning project for classifying chest X-ray images to diagnose 14 different pathological conditions using the [ChestX-ray8 dataset](https://arxiv.org/abs/1705.02315). The project leverages transfer learning with a pre-trained DenseNet121 model and includes a Gradio app for user-friendly X-ray image classification.

## Features
- **Data Preprocessing**: Processes and prepares a subset of the ChestX-ray8 dataset (~1000 images) for training, validation, and testing.
- **Class Imbalance Handling**: Implements weighted loss functions to address class imbalance across pathologies.
- **Transfer Learning**: Utilizes DenseNet121 for efficient and accurate classification.
- **Model Evaluation**: Computes AUC and ROC curves to measure diagnostic performance.
- **Interpretability**: Visualizes model focus areas using GradCAM heatmaps.
- **Gradio App**: Provides an interactive interface for users to upload X-ray images and receive classification results.

## Dataset
The project uses a subset of the ChestX-ray8 dataset, containing:
- **Training**: 875 images
- **Validation**: 109 images
- **Testing**: 420 images

Labels include 14 pathologies, such as Cardiomegaly, Effusion, and Pneumonia, with annotations for 5 pathologies by radiologist consensus.
   ```bash
   git clone https://github.com/your-username/chest-xray-diagnosis.git
   cd chest-xray-diagnosis
