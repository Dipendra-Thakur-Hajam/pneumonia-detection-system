# Pneumonia Detection Using Chest X-Ray Images
## Project Report — NNARL (Neural Network Applications in Real Life)

**Submitted by:** Dipendra Thakur  
**Subject:** NNARL (Neural Network Applications in Real Life) — Practical 1  
**Technology Stack:** Python, TensorFlow/Keras, Flask, HTML/CSS/JavaScript

---

## 1. Introduction

Pneumonia is an infection that inflames the air sacs in one or both lungs, potentially filling them with fluid or pus. It remains a leading cause of death worldwide, especially among children under 5 and adults over 65. Early and accurate detection through chest X-ray analysis is critical for timely treatment.

This project implements an **AI-powered Pneumonia Detection System** that uses **Deep Learning (Convolutional Neural Networks)** to automatically analyze chest X-ray images and classify them as either **NORMAL** or **PNEUMONIA** with a confidence score. The system also provides **Grad-CAM (Gradient-weighted Class Activation Mapping)** heatmap visualizations to explain which regions of the X-ray the model focuses on for its prediction.

---

## 2. Objectives

1. Build a deep learning model capable of accurately classifying chest X-ray images as Normal or Pneumonia.
2. Implement a web-based interface for easy image upload and real-time prediction.
3. Provide visual explanations (Grad-CAM heatmaps) to make predictions interpretable.
4. Generate downloadable AI radiology reports for each prediction.
5. Handle class imbalance in the dataset to improve prediction accuracy on both classes.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Browser)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Upload UI   │  │  Prediction  │  │  AI Radiology     │  │
│  │  (Drag/Drop) │  │  Results     │  │  Report + Download│  │
│  └──────┬──────┘  └──────┬───────┘  └───────────────────┘  │
│         │                │                                   │
└─────────┼────────────────┼───────────────────────────────────┘
          │  HTTP POST     │  JSON Response
          ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                 BACKEND (Flask REST API)                     │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐ │
│  │  Image        │  │  ResNet50     │  │  Grad-CAM        │ │
│  │  Preprocessing│  │  Model        │  │  Heatmap Engine  │ │
│  └──────────────┘  └───────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│               TRAINING PIPELINE (Offline)                    │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐ │
│  │  Data         │  │  Two-Phase    │  │  Model           │ │
│  │  Augmentation │  │  Fine-Tuning  │  │  Evaluation      │ │
│  └──────────────┘  └───────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Project Directory Structure

```
pneumonia-detection/
├── backend/
│   ├── app.py                 # Flask REST API server
│   ├── gradcam.py             # Grad-CAM heatmap generation
│   └── utils.py               # Utility functions
├── frontend/
│   ├── index.html             # Web interface (HTML)
│   ├── script.js              # Frontend logic (JavaScript)
│   └── style.css              # Styling (CSS)
├── training/
│   ├── preprocess_data.py     # Data preprocessing & augmentation
│   ├── train_resnet50.py      # Model training script
│   └── evaluate_model.py      # Model evaluation & metrics
├── model/
│   └── resnet50_pneumonia.h5  # Trained model file (~100 MB)
├── dataset/
│   ├── train/                 # Training images
│   │   ├── NORMAL/            # 1,341 normal X-rays
│   │   └── PNEUMONIA/         # 3,875 pneumonia X-rays
│   ├── val/                   # Validation images
│   │   ├── NORMAL/            # 8 images
│   │   └── PNEUMONIA/         # 8 images
│   └── test/                  # Test images
│       ├── NORMAL/            # 234 images
│       └── PNEUMONIA/         # 390 images
├── uploads/                   # Saved uploaded images
│   ├── normal/
│   └── pneumonia/
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 5. Dataset

- **Source:** Kaggle — Chest X-Ray Images (Pneumonia) by Paul Mooney
- **URL:** https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
- **Total Images:** ~5,856

| Split      | NORMAL | PNEUMONIA | Total |
|------------|--------|-----------|-------|
| Training   | 1,341  | 3,875     | 5,216 |
| Validation | 8      | 8         | 16    |
| Test       | 234    | 390       | 624   |

> **Note:** The dataset has a class imbalance (2.9:1 ratio of Pneumonia to Normal). This was addressed using **computed class weights** during training.

---

## 6. Deep Learning Model

### 6.1 Model Architecture: ResNet50 (Transfer Learning)

The system uses **ResNet50** — a 50-layer deep Residual Network pre-trained on the ImageNet dataset (1.2 million images, 1000 classes). Transfer learning allows leveraging features learned from natural images and adapting them to medical X-ray classification.

**Architecture Details:**

| Layer | Description |
|-------|-------------|
| **Base Model** | ResNet50 (pre-trained on ImageNet, top layers removed) |
| **Global Average Pooling 2D** | Reduces feature maps to a single vector |
| **Batch Normalization** | Normalizes activations for stable training |
| **Dense (256 units)** | Fully connected layer with ReLU activation + L2 regularization |
| **Dropout (50%)** | Prevents overfitting by randomly dropping neurons |
| **Dense (64 units)** | Fully connected layer with ReLU activation + L2 regularization |
| **Dropout (30%)** | Additional regularization |
| **Dense (1 unit)** | Output layer with Sigmoid activation (binary classification) |

**Input Size:** 224 × 224 × 3 (RGB)  
**Output:** Single probability value (0 = NORMAL, 1 = PNEUMONIA)  
**Classification Threshold:** 0.6 (predictions above 60% probability → PNEUMONIA)

### 6.2 Training Strategy: Two-Phase Fine-Tuning

The model is trained in **two phases** to maximize accuracy:

**Phase 1 — Train Classifier Head (10 epochs)**
- All ResNet50 backbone layers are **frozen** (not trainable)
- Only the new dense layers are trained
- Learning Rate: 1e-3 (Adam optimizer)
- Purpose: Learn good classification weights without disturbing pretrained features

**Phase 2 — Fine-Tune Backbone (20 epochs)**
- The last **30 layers** of ResNet50 are **unfrozen** and made trainable
- Learning Rate: 1e-5 (very low to avoid destroying pretrained weights)
- BatchNormalization layers remain frozen
- Purpose: Allow the model to learn X-ray-specific features in the deeper layers

### 6.3 Anti-Overfitting Measures

| Technique | Implementation |
|-----------|----------------|
| **Data Augmentation** | Rotation, shifts, zoom, brightness, channel shifts, horizontal flip |
| **Dropout** | 50% and 30% dropout in dense layers |
| **L2 Regularization** | Weight decay (1e-4) on dense layers |
| **Early Stopping** | Stops training when validation loss doesn't improve for 5–7 epochs |
| **Learning Rate Reduction** | Halves learning rate when validation loss plateaus for 3 epochs |
| **Class Weights** | Computed automatically using `sklearn.utils.class_weight.compute_class_weight('balanced')` |
| **Model Checkpoint** | Saves only the best model based on validation accuracy |

### 6.4 Data Augmentation

Training images undergo real-time augmentation to increase variety and reduce overfitting:

| Augmentation | Value |
|-------------|-------|
| Rotation | ±25° |
| Width Shift | ±20% |
| Height Shift | ±20% |
| Shear | ±20% |
| Zoom | ±20% |
| Brightness | 70%–130% |
| Channel Shift | ±20 |
| Horizontal Flip | Yes |
| Fill Mode | Nearest |

### 6.5 Image Preprocessing

Images are preprocessed using **ResNet50's built-in `preprocess_input()`** function, which applies ImageNet-specific normalization (mean subtraction per channel). This ensures the model sees data in the same format it was originally trained on.

---

## 7. Features

### 7.1 Image Upload (Drag & Drop + File Browser)
- Users can **drag and drop** a chest X-ray image directly onto the upload area
- Alternatively, users can click **"Choose File"** to browse and select an image
- Supported formats: **JPG, PNG** (any image format)
- File name, size, and type are displayed after upload

### 7.2 AI Prediction with Confidence Score
- The model classifies the uploaded X-ray as **NORMAL** or **PNEUMONIA**
- A **confidence percentage** is displayed showing how certain the model is
- Results are color-coded: **Green** for NORMAL, **Red** for PNEUMONIA
- A **confidence meter bar** provides a visual representation of certainty

### 7.3 Grad-CAM Heatmap Visualization
- **Grad-CAM (Gradient-weighted Class Activation Mapping)** generates a heatmap showing which areas of the X-ray influenced the model's decision
- The heatmap is overlaid on the original X-ray for **side-by-side comparison**
- A **lung region mask** constrains the heatmap to only highlight areas within the lung boundaries (anatomically accurate)
- Weak activations below a threshold (35%) are filtered out for cleaner visualization

### 7.4 Estimated Lung Infection Area
- For PNEUMONIA predictions, the system displays an **estimated percentage of lung area affected**
- For NORMAL predictions, it shows "No visible lung infection detected"

### 7.5 AI Radiology Report Generation
- After each analysis, an **AI Radiology Report** is automatically generated containing:
  - Prediction result (NORMAL/PNEUMONIA)
  - Confidence percentage
  - Findings summary
  - Recommendation (research use disclaimer)
- Users can **download the report as a text file** ("AI_Radiology_Report.txt")

### 7.6 Image Storage & History
- All uploaded images are **automatically saved** to the server
- Images are organized into `uploads/normal/` and `uploads/pneumonia/` folders based on prediction
- Filenames include timestamps for unique identification

### 7.7 REST API Backend
- The Flask backend exposes a **REST API endpoint**: `POST /predict`
- Accepts multipart form data with an image file
- Returns JSON response with prediction, confidence, and base64-encoded heatmap
- Includes **CORS support** for cross-origin frontend access
- Health check endpoint at `GET /`

### 7.8 Responsive Web Design
- The frontend is fully **responsive** and works on desktop, tablet, and mobile screens
- Uses **CSS Grid** for layout with breakpoints at 960px and 520px
- Modern **glassmorphism** design with blur effects, gradients, and subtle shadows
- **Manrope** font from Google Fonts for clean typography
- Smooth hover animations and transitions on all interactive elements

### 7.9 Loading State & Error Handling
- A **loading overlay** with spinner appears during analysis
- All interactive elements are disabled during processing to prevent duplicate requests
- Error messages with a **retry button** are shown if prediction fails
- Graceful error handling in both frontend and backend

### 7.10 Model Evaluation Tools
- Comprehensive evaluation script that produces:
  - **Overall test accuracy**
  - **Classification report** (precision, recall, F1-score per class)
  - **Confusion matrix**
  - **Per-class accuracy** breakdown

---

## 8. Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Deep Learning Framework** | TensorFlow 2.x / Keras | Model building, training, and inference |
| **CNN Architecture** | ResNet50 (Transfer Learning) | Feature extraction from X-ray images |
| **Explainability** | Grad-CAM | Visual explanations for model predictions |
| **Backend Framework** | Flask (Python) | REST API server for model serving |
| **API Communication** | Flask-CORS | Cross-origin resource sharing |
| **Image Processing** | OpenCV (cv2), Pillow (PIL) | Image manipulation and preprocessing |
| **Numerical Computing** | NumPy | Array operations for image data |
| **Machine Learning Utils** | scikit-learn | Class weight computation, evaluation metrics |
| **Frontend** | HTML5, CSS3, JavaScript (Vanilla) | User interface |
| **Visualization** | Matplotlib | Training visualization (optional) |
| **Dataset** | Kaggle Chest X-Ray | 5,856 labeled chest X-ray images |

---

## 9. How to Run the Project

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Dependencies
```bash
cd pneumonia-detection
pip install -r requirements.txt
```

### Step 2: Start the Backend Server
```bash
cd backend
python app.py
```
The API server will start at `http://127.0.0.1:5000`

### Step 3: Open the Frontend
Open `frontend/index.html` in any web browser.

### Step 4: Use the Application
1. Upload a chest X-ray image (drag & drop or browse)
2. Click **"Analyze X-ray"**
3. View the prediction, confidence score, and Grad-CAM heatmap
4. Download the AI radiology report if needed

### (Optional) Retrain the Model
```bash
cd pneumonia-detection
python training/train_resnet50.py
```

### (Optional) Evaluate the Model
```bash
cd pneumonia-detection
python training/evaluate_model.py
```

---

## 10. API Reference

### POST `/predict`

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` — chest X-ray image (JPG/PNG)

**Response (JSON):**
```json
{
  "prediction": "NORMAL" | "PNEUMONIA",
  "confidence": 0.0 to 1.0,
  "heatmap": "<base64-encoded JPEG image>"
}
```

**Error Response:**
```json
{
  "error": "Error message"
}
```

---

## 11. Key Algorithms & Techniques

### 11.1 Transfer Learning
Instead of training a CNN from scratch (which requires millions of images), we use **ResNet50 pre-trained on ImageNet**. The pre-trained weights act as a powerful feature extractor — the network already understands edges, textures, shapes, and patterns from natural images. We then **fine-tune** the model on our medical X-ray dataset so it learns domain-specific features like lung opacities and consolidation patterns.

### 11.2 Grad-CAM (Gradient-weighted Class Activation Mapping)
Grad-CAM is an **explainability technique** that produces a visual heatmap highlighting the important regions in the input image that influenced the model's prediction.

**How it works:**
1. Forward pass the image through the model
2. Compute gradients of the predicted class score with respect to the last convolutional layer
3. Compute average gradient per feature map channel (importance weights)
4. Create weighted combination of feature maps
5. Apply ReLU to keep only positive contributions
6. Resize and overlay on the original image

### 11.3 Class Imbalance Handling
The dataset has 2.9× more PNEUMONIA images than NORMAL. Without compensation, the model would be biased toward predicting PNEUMONIA. We use **computed class weights** (`sklearn.utils.class_weight`) that assign higher importance to the under-represented NORMAL class during training:

```
class_weight = {0: ~1.94, 1: ~0.67}
```

This means the model penalizes misclassifying a NORMAL image ~3× more than misclassifying a PNEUMONIA image, forcing it to learn both classes equally well.

---

## 12. Results & Screenshots

The model achieves the following performance on the test set:

| Metric | Target |
|--------|--------|
| **Overall Accuracy** | >88% |
| **NORMAL Precision** | High (minimize false pneumonia alarms) |
| **PNEUMONIA Recall** | High (minimize missed pneumonia cases) |

> Note: Run `python training/evaluate_model.py` for exact metrics with the latest trained model.

---

## 13. Limitations

1. **Research Tool Only** — Not a medical device; should not be used for clinical diagnosis
2. **Binary Classification** — Only distinguishes Normal vs Pneumonia; does not differentiate bacterial vs viral pneumonia
3. **Image Quality Dependent** — Performance may degrade on low-quality, rotated, or non-standard X-rays
4. **Dataset Size** — Trained on ~5,200 images; larger datasets could improve generalization
5. **Single View** — Only analyzes PA (posteroanterior) chest X-rays

---

## 14. Future Enhancements

1. Multi-class classification (Normal vs Bacterial Pneumonia vs Viral Pneumonia)
2. Integration with DICOM medical imaging format
3. Patient history tracking and comparison
4. Cloud deployment for wider accessibility
5. Mobile application for point-of-care diagnosis
6. Larger and more diverse training datasets
7. Ensemble models for higher accuracy

---

## 15. Conclusion

This project demonstrates the practical application of **deep learning and transfer learning** in medical image analysis. By leveraging a pre-trained ResNet50 model, two-phase fine-tuning, class weight balancing, and Grad-CAM visualizations, the system provides accurate pneumonia detection with interpretable results. The web-based interface makes it accessible and user-friendly, while the downloadable AI radiology report adds practical utility for demonstration and research purposes.

---

## 16. References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
2. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV 2017*.
3. Rajpurkar, P., et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays. *arXiv:1711.05225*.
4. Kaggle Chest X-Ray Dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
5. TensorFlow Documentation: https://www.tensorflow.org/
6. Keras Applications — ResNet50: https://keras.io/api/applications/resnet/

---

*Report generated for NNARL Practical 1 — Pneumonia Detection Using Chest X-Ray Images*
