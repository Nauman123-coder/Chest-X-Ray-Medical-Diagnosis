import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from PIL import Image
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import os

# Disable TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Define labels
labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']

def compute_class_freqs(labels_array):
    """Compute positive and negative frequencies for each class."""
    positive_frequencies = np.mean(labels_array, axis=0)
    negative_frequencies = 1 - positive_frequencies
    return positive_frequencies, negative_frequencies

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """Return weighted loss function given negative weights and positive weights."""
    def weighted_loss(y_true, y_pred):
        loss = 0.0
        for i in range(len(pos_weights)):
            loss += K.mean(-(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon) +
                             neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon)))
        return loss
    return weighted_loss

# Class frequencies from training data (pre-computed)
freq_pos = np.array([0.02, 0.013, 0.128, 0.002, 0.175, 0.045, 0.054, 0.106, 
                     0.038, 0.021, 0.01, 0.014, 0.016, 0.033])
freq_neg = 1 - freq_pos
pos_weights = freq_neg
neg_weights = freq_pos

class ChestXrayDiagnosis:
    def __init__(self, model_path):
        """Initialize the model and load weights."""
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model."""
        try:
            # Load model with custom objects
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={'weighted_loss': get_weighted_loss(pos_weights, neg_weights)}
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # If loading fails, create model architecture and load weights
            self.create_model_architecture()
    
    def create_model_architecture(self):
        """Create model architecture if direct loading fails."""
        try:
            # Create the base pre-trained model
            base_model = DenseNet121(weights=None, include_top=False)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(len(labels), activation="sigmoid")(x)
            
            self.model = Model(inputs=base_model.input, outputs=predictions)
            self.model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))
            
            # Try to load weights if they exist
            weights_path = self.model_path.replace('.keras', '_weights.h5')
            if os.path.exists(weights_path):
                self.model.load_weights(weights_path)
                print("Model architecture created and weights loaded!")
            else:
                print("Model architecture created, but no weights file found.")
        except Exception as e:
            print(f"Error creating model: {e}")
    
    def preprocess_image(self, image):
        """Preprocess the input image for model prediction."""
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        
        # Resize to 320x320
        image = cv2.resize(image, (320, 320))
        
        # Convert to 3 channels (RGB) by repeating grayscale
        image = np.stack([image] * 3, axis=-1)
        
        # Normalize the image (same as training)
        image = image.astype(np.float32)
        image = (image - np.mean(image)) / (np.std(image) + 1e-7)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image):
        """Make prediction on the input image."""
        if self.model is None:
            return None, "Model not loaded properly"
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            return predictions[0], None
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def compute_gradcam(self, image, cls, layer_name='bn'):
        """Compute GradCAM for a specific class."""
        try:
            grad_model = tf.keras.models.Model(
                [self.model.inputs],
                [self.model.get_layer(layer_name).output, self.model.output]
            )

            processed_image = self.preprocess_image(image)

            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(processed_image)
                loss = predictions[:, cls]

            grads = tape.gradient(loss, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_output = conv_output[0].numpy()
            pooled_grads = pooled_grads.numpy()

            for i in range(pooled_grads.shape[-1]):
                conv_output[:, :, i] *= pooled_grads[i]

            cam = np.mean(conv_output, axis=-1)
            cam = np.maximum(cam, 0)
            cam = cam / (np.max(cam) + 1e-8)

            cam = cv2.resize(cam, (320, 320), interpolation=cv2.INTER_LINEAR)

            return cam
            
        except Exception as e:
            print(f"GradCAM error: {e}")
            return None
    
    def create_visualization(self, image, predictions):
        """Create visualization with original image and GradCAM heatmaps."""
        try:
            selected_labels = ['Cardiomegaly', 'Mass', 'Pneumothorax', 'Edema']
            
            # Create figure
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            # Convert image for display
            if isinstance(image, Image.Image):
                display_image = np.array(image.convert('L'))
            else:
                display_image = image
                if len(display_image.shape) == 3:
                    display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2GRAY)
            
            display_image = cv2.resize(display_image, (320, 320))
            
            # Original image
            axes[0].imshow(display_image, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # GradCAM for selected labels
            j = 1
            for label in selected_labels:
                i = labels.index(label)
                heatmap = self.compute_gradcam(image, i)
                
                if heatmap is not None:
                    axes[j].imshow(display_image, cmap='gray')
                    axes[j].imshow(heatmap, cmap='jet', alpha=min(0.5, predictions[i]))
                else:
                    # If GradCAM fails, just show the original image
                    axes[j].imshow(display_image, cmap='gray')
                
                axes[j].set_title(f'{label}: p={predictions[i]:.3f}')
                axes[j].axis('off')
                j += 1
            
            plt.tight_layout()
            
            # Convert plot to image
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            
            # Convert to PIL Image
            result_image = Image.open(buf)
            plt.close(fig)
            
            return result_image
            
        except Exception as e:
            print(f"Visualization error: {e}")
            return None

# Initialize the diagnosis system
MODEL_PATH = r'E:\AI in Medical Diagnosis\Model Building\Final Model\models\chest_xray_model.keras'
diagnosis_system = ChestXrayDiagnosis(MODEL_PATH)

def diagnose_xray(image):
    """Main function for Gradio interface."""
    if image is None:
        return None, "Please upload an X-ray image."
    
    # Make prediction
    predictions, error = diagnosis_system.predict(image)
    
    if error:
        return None, error
    
    if predictions is None:
        return None, "Failed to make predictions."
    
    # Create visualization
    result_image = diagnosis_system.create_visualization(image, predictions)
    
    if result_image is None:
        return None, "Failed to create visualization."
    
    # Create prediction summary
    sorted_indices = np.argsort(predictions)[::-1][:5]
    prediction_text = "Prediction Summary\nTop 5 Predictions:\n"
    for rank, idx in enumerate(sorted_indices, 1):
        label = labels[idx]
        prob = predictions[idx]
        prediction_text += f"{rank}. {label}: {prob:.3f} ({prob*100:.1f}%)\n"
    
    return result_image, prediction_text

# Create Gradio interface
def create_interface():
    """Create and configure Gradio interface."""
    
    title = "🩻 Chest X-Ray Medical Diagnosis with Deep Learning"
    
    
    # Create interface
    interface = gr.Interface(
        fn=diagnose_xray,
        inputs=gr.Image(type="pil", label="Upload Chest X-Ray Image"),
        outputs=[
            gr.Image(type="pil", label="Analysis Results with GradCAM"),
            gr.Textbox(label="Prediction Summary", lines=8)
        ],
        title=title,
        theme="huggingface",
        allow_flagging="never"
    )
    
    return interface

# Launch the application
if __name__ == "__main__":
    # Create and launch interface
    interface = create_interface()
    
    # Launch with specific settings
    interface.launch(
        share=True,  # Creates a public link
        server_name="0.0.0.0",  # Allows access from other devices on network
        server_port=7860,  # Specific port
        show_error=True,  # Show detailed error messages
        debug=True  # Enable debug mode
    )