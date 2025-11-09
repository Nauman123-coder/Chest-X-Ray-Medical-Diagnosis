import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity
import os

random.seed(a=None, version=2)

set_verbosity(INFO)


def get_mean_std_per_batch(image_dir, df, H=320, W=320):
    sample_data = []                                     # List to store image pixel data
    for img in df.sample(100)["Image"].values:           # Randomly select 100 image file names from the DataFrame
        image_path = os.path.join(image_dir, img)        # Build full file path to the image
        img_array = np.array(image.load_img(image_path, target_size=(H, W)))# Load the image, resize to (H, W), and convert to a NumPy array
        sample_data.append(img_array)                    # Append the image array to the sample_data list
    sample_data = np.array(sample_data)                  # Convert list of arrays to a single 4D NumPy array of shape (100, H, W, 3)
    mean = np.mean(sample_data, axis=(0, 1, 2, 3))       # Compute the overall mean of all pixel values (across all dimensions)
    std = np.std(sample_data, axis=(0, 1, 2, 3), ddof=1) # Compute the sample std of all pixel values
    return mean, std                                     # Return the calculated mean and standard deviation


def load_image(img, image_dir, df, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    mean, std = get_mean_std_per_batch(image_dir, df, H=H, W=W)
    img_path = os.path.join(image_dir, img)
    x = image.load_img(img_path, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x


# def grad_cam(input_model, image, cls, layer_name, H=320, W=320):
#     """GradCAM method for visualizing input saliency."""
#     y_c = input_model.output[0, cls]
#     conv_output = input_model.get_layer(layer_name).output
#     grads = K.gradients(y_c, conv_output)[0]

#     gradient_function = K.function([input_model.input], [conv_output, grads])

#     output, grads_val = gradient_function([image])
#     output, grads_val = output[0, :], grads_val[0, :, :, :]

#     weights = np.mean(grads_val, axis=(0, 1))
#     cam = np.dot(output, weights)

#     # Process CAM
#     cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
#     cam = np.maximum(cam, 0)
#     cam = cam / cam.max()
#     return cam

import tensorflow as tf
import numpy as np
import cv2  # Ensure cv2 is installed

def grad_cam(input_model, image, cls, layer_name, H=320, W=320):
    # Create a model that gives both the conv layer and the final output
    grad_model = tf.keras.models.Model(
        [input_model.inputs],
        [input_model.get_layer(layer_name).output, input_model.output]
    )

    # Compute gradients of the class output with respect to conv layer
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image)
        loss = predictions[:, cls]
    
    grads = tape.gradient(loss, conv_output)  # Gradient of class score w.r.t conv_output
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Channel-wise mean

    # Multiply conv_output by importance weights
    conv_output = conv_output[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]

    # Generate CAM
    cam = np.mean(conv_output, axis=-1)
    cam = np.maximum(cam, 0)  # ReLU
    cam = cam / (np.max(cam) + 1e-8)  # Normalize

    # Resize to original image dimensions
    cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

    return cam


def compute_gradcam(model, img, image_dir, df, labels, selected_labels,
                    layer_name='bn'):
    preprocessed_input = load_image(img, image_dir, df)
    predictions = model.predict(preprocessed_input)

    print("Loading original image")
    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(load_image(img, image_dir, df, preprocess=False), cmap='gray')

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            print(f"Generating gradcam for class {labels[i]}")
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            plt.subplot(151 + j)
            #plt.subplot(1, len(selected_labels), j + 1)

            plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
            plt.axis('off')
            plt.imshow(load_image(img, image_dir, df, preprocess=False),
                       cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
            j += 1


def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals
