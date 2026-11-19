import tensorflow as tf
import numpy as np
import cv2


def generate_gradcam(model, img_array, layer_name=None):

    # Auto-detect the last convolutional layer
    if layer_name is None:
        for layer in reversed(model.layers):
            if "conv" in layer.name:
                layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: keep only positive activations
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    # ---------- Threshold out weak activations ----------
    threshold = 0.35
    heatmap[heatmap < threshold] = 0

    # Re-normalize after thresholding
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = (heatmap - threshold) / (max_val - threshold)
        heatmap = np.clip(heatmap, 0, 1)

    return heatmap


def create_lung_mask(img):
    """
    Create a lung region mask using fixed anatomical proportions.
    Based on standard PA chest X-ray anatomy:
    - Lungs are the two large dark areas on either side of the heart
    - Right lung (left side of image) is slightly larger than left
    - They span from roughly the clavicles down to the diaphragm
    """
    h, w = img.shape[:2]

    lung_mask = np.zeros((h, w), dtype=np.float32)

    # ---- Right lung (appears on LEFT side of X-ray image) ----
    # Larger lung, centered around 30% from left, 38% from top
    right_center = (int(w * 0.30), int(h * 0.38))
    right_axes = (int(w * 0.20), int(h * 0.32))
    cv2.ellipse(lung_mask, right_center, right_axes, 0, 0, 360, 1.0, -1)

    # ---- Left lung (appears on RIGHT side of X-ray image) ----
    # Slightly smaller due to heart, centered around 70% from left
    left_center = (int(w * 0.70), int(h * 0.38))
    left_axes = (int(w * 0.18), int(h * 0.32))
    cv2.ellipse(lung_mask, left_center, left_axes, 0, 0, 360, 1.0, -1)

    # Light blur for soft edges (small kernel to avoid mask expanding too much)
    lung_mask = cv2.GaussianBlur(lung_mask, (21, 21), 0)

    # Normalize
    max_val = np.max(lung_mask)
    if max_val > 0:
        lung_mask /= max_val

    return lung_mask


def overlay_heatmap(img, heatmap):

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Smooth the heatmap FIRST (before masking)
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)

    # THEN apply lung mask to constrain to lung regions only
    lung_mask = create_lung_mask(img)
    heatmap_resized = heatmap_resized * lung_mask

    # Hard cutoff: zero out anything where mask is very weak
    heatmap_resized[lung_mask < 0.1] = 0

    # Re-normalize after masking
    max_val = np.max(heatmap_resized)
    if max_val > 0:
        heatmap_resized = heatmap_resized / max_val

    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Create a mask: only blend where heatmap has significant values
    mask = heatmap_resized[..., np.newaxis]

    # Blend: show original image clearly, overlay color only where active
    alpha = 0.5
    superimposed_img = img * (1 - mask * alpha) + heatmap_color * (mask * alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255)

    return superimposed_img