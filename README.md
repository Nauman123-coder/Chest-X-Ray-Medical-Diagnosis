# Chest X-Ray Medical Diagnosis with Deep Learning

![Chest X-Ray Analysis](images/xray-header-image.png)

## Overview

This project implements a state-of-the-art deep learning model for automated chest X-ray diagnosis using convolutional neural networks. The system can simultaneously detect 14 different pathological conditions from frontal-view chest X-ray images, providing binary classification predictions (positive/negative) for each condition.

## Clinical Significance

### Why This Model Matters

Chest X-rays are one of the most common diagnostic imaging procedures worldwide, with over 2 billion performed annually. However, the interpretation of chest X-rays requires significant expertise and time from radiologists. This AI-powered diagnostic system addresses several critical healthcare challenges:

1. **Radiologist Shortage**: Many regions face a severe shortage of trained radiologists, leading to delayed diagnoses and increased healthcare costs
2. **Time Efficiency**: Automated screening can help prioritize urgent cases and reduce the workload on radiologists
3. **Second Opinion**: The model can serve as a diagnostic aid, providing a second opinion to validate or flag potential findings
4. **Accessibility**: Enables high-quality diagnostic screening in underserved areas with limited access to specialist radiologists
5. **Consistency**: Provides consistent interpretations across different healthcare facilities and reduces inter-observer variability

## Dataset

### ChestX-ray8 Dataset

- **Source**: NIH Clinical Center
- **Total Images**: 108,948 frontal-view chest X-ray images
- **Unique Patients**: 32,717 patients
- **Image Resolution**: 320x320 pixels (resized and normalized)
- **Format**: Grayscale X-ray images converted to 3-channel format

### Pathological Conditions (14 Classes)

The model is trained to detect the following conditions:

1. **Atelectasis** - Partial lung collapse
2. **Cardiomegaly** - Enlarged heart
3. **Consolidation** - Fluid/tissue filling air spaces
4. **Edema** - Fluid accumulation in lungs
5. **Effusion** - Fluid around lungs
6. **Emphysema** - Damaged air sacs
7. **Fibrosis** - Lung tissue scarring
8. **Hernia** - Organ displacement
9. **Infiltration** - Substance accumulation in lungs
10. **Mass** - Abnormal tissue growth
11. **Nodule** - Small rounded growth
12. **Pleural Thickening** - Thickened lung lining
13. **Pneumonia** - Lung infection/inflammation
14. **Pneumothorax** - Collapsed lung

### Data Split

- **Training Set**: 875 images (87.5%)
- **Validation Set**: 109 images (10.9%)
- **Test Set**: 420 images

**Note**: Split performed at patient level to prevent data leakage - ensuring no patient appears in multiple datasets.

## Data Preprocessing

### Handling Class Imbalance

Medical datasets often exhibit severe class imbalance. Our dataset shows:

- **Most Imbalanced**: Hernia (~0.2% positive cases)
- **Least Imbalanced**: Infiltration (~17.5% positive cases)

**Solution Implemented**: Weighted loss function that assigns class-specific weights to balance positive and negative contributions:
```
w_pos = freq_negative
w_neg = freq_positive
```

This ensures equal contribution from both classes during training.

### Image Normalization

- **Training Data**: Batch-wise normalization (zero mean, unit standard deviation)
- **Validation/Test Data**: Normalized using training set statistics to prevent data leakage
- **Color Conversion**: Grayscale images converted to 3-channel format for compatibility with pre-trained models

## Model Architecture

### Base Model: DenseNet121

The model uses **DenseNet121** (Dense Convolutional Network) as the feature extractor:

- **Pre-trained Weights**: ImageNet weights adapted for medical imaging
- **Architecture Advantage**: Dense connections between layers enable feature reuse and gradient flow
- **Input Shape**: (320, 320, 3)

### Custom Layers

1. **Global Average Pooling 2D**
   - Reduces spatial dimensions while preserving feature information
   - Prevents overfitting by reducing parameters

2. **Output Dense Layer**
   - 14 neurons (one per pathology)
   - Sigmoid activation for multi-label binary classification
   - Enables simultaneous prediction of multiple conditions

### Transfer Learning

- Leverages knowledge from ImageNet pre-training
- Fine-tuned on chest X-ray dataset
- Significantly reduces training time and improves performance

## Training Configuration

### Loss Function
**Weighted Binary Cross-Entropy**:
```
L = -Σ [w_pos × y × log(ŷ + ε) + w_neg × (1-y) × log(1-ŷ + ε)]
```
where ε = 1e-7 prevents numerical errors

### Optimizer
- **Adam Optimizer**: Adaptive learning rate for efficient convergence

### Training Process

The model was trained on the full dataset using:
- **Batch Size**: 32
- **Hardware**: GPU-accelerated training
- **Callbacks**:
  - `ModelCheckpoint`: Save best model based on validation loss
  - `ReduceLROnPlateau`: Adaptive learning rate decay
  - `EarlyStopping`: Prevent overfitting with patience monitoring
  - `TensorBoard`: Real-time training visualization

### Data Augmentation
- Random horizontal flipping
- Batch-wise normalization
- Shuffling after each epoch

## Model Evaluation

### Primary Metric: AUROC (Area Under ROC Curve)

The model is evaluated using the **ROC-AUC** score for each pathology:

- **ROC Curve**: Plots True Positive Rate vs False Positive Rate at various thresholds
- **AUROC Value**: Measures overall classification performance (0.5 = random, 1.0 = perfect)
- **Interpretation**: Higher AUROC indicates better diagnostic accuracy

### Performance Highlights

The model achieves competitive performance comparable to published research on the ChestX-ray8 dataset, including comparison with radiologist-level performance documented in the CheXNet and ChexNeXt papers.

**Top Performing Classes** (by AUROC):
1. Cardiomegaly
2. Emphysema
3. Mass
4. Edema

## Model Interpretability: GradCAM Visualization

### What is GradCAM?

**Gradient-weighted Class Activation Mapping (GradCAM)** provides visual explanations of model predictions:

- **Purpose**: Shows which regions of the X-ray influenced the model's decision
- **Technique**: Extracts gradients flowing into the final convolutional layer
- **Output**: Heatmap overlay highlighting relevant anatomical regions

### Clinical Value

- **Model Validation**: Radiologists can verify the model is focusing on clinically relevant areas
- **Trust Building**: Transparent decision-making increases clinician confidence
- **Educational Tool**: Helps train medical students by highlighting diagnostic features
- **Debugging**: Identifies if model is learning spurious correlations

### Example Visualizations

The notebook includes GradCAM visualizations for multiple test cases, showing model attention for the top-performing pathologies across different patient X-rays.

## Project Structure
```
.
├── app.py                              # Gradio web application for inference
├── chest_x_ray_diagnosis.ipynb         # Complete implementation notebook
├── util.py                             # Utility functions (ROC, GradCAM)
├── images/
│   └── xray-header-image.png          # Header visualization
├── models/
│   ├── chest_xray_model.keras         # Trained model weights
│   └── nih/
│       ├── densenet.hdf5              # Pre-trained DenseNet121
│       └── pretrained_model.h5        # Fine-tuned model
└── README.md
```

**Note**: Dataset excluded from repository due to size constraints

## Requirements
```python
tensorflow >= 2.x
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
```

## Usage

### Running the Notebook

1. Open `chest_x_ray_diagnosis.ipynb` in Jupyter Notebook or JupyterLab
2. Execute cells sequentially to:
   - Load and preprocess the dataset
   - Check for data leakage
   - Address class imbalance
   - Build and compile the model
   - Load pre-trained weights
   - Generate predictions on test set
   - Evaluate using ROC-AUC metrics
   - Visualize results with GradCAM

### Model Inference

Use `app.py` to run the Gradio web interface for real-time predictions on new chest X-ray images.

## Key Technical Implementations

### 1. Data Leakage Prevention
Custom `check_for_leakage()` function ensures no patient overlap between train/validation/test sets at the patient ID level.

### 2. Weighted Loss Computation
`get_weighted_loss()` function implements class-specific weighting to handle severe class imbalance in medical datasets.

### 3. Custom Data Generators
- Training generator: Batch normalization with augmentation
- Validation/Test generators: Normalized using training statistics

### 4. Multi-Label Classification
Sigmoid activation enables independent probability predictions for each of the 14 pathologies.

## Research References

This implementation is inspired by state-of-the-art research:

- **CheXNet**: Radiologist-Level Pneumonia Detection on Chest X-Rays
- **CheXpert**: A Large Chest Radiograph Dataset with Uncertainty Labels
- **ChexNeXt**: Automated Chest X-ray Interpretation

## Clinical Disclaimer

This model is designed for research and educational purposes. It should be used as a diagnostic aid tool only and not as a replacement for professional medical judgment. All predictions should be reviewed and validated by qualified healthcare professionals.

## Future Improvements

- Multi-view integration (lateral and AP views)
- Uncertainty quantification for predictions
- Attention mechanisms for better interpretability
- Integration with electronic health records (EHR)
- Real-time deployment optimization

---

**Developed as part of AI for Medical Diagnosis specialization**
