# Pneumonia Detection Using Deep Learning

A web application that uses a trained ResNet50 model to classify chest X-ray images as **NORMAL** or **PNEUMONIA**. The app also generates a Grad-CAM heatmap to highlight the regions that influenced the prediction.

## Features

- Upload a chest X-ray image through the web interface
- Run pneumonia prediction with confidence score
- Display an explanation heatmap using Grad-CAM
- View model evaluation metrics
- Save uploaded images by predicted category

## Requirements

- Windows, Linux, or macOS
- Python 3.10 (recommended)
- `py` launcher on Windows
- Internet access for the initial package installation

## Run the Project Locally

### 1. Open the project directory

```powershell
cd "C:\Users\dipen\OneDrive\Desktop\POGRESS\2nd year\SEM6\NNARL\NNARL_p1\pneumonia-detection"
```

On another computer, use `cd` with the location where this repository was cloned.

### 2. Create a virtual environment

Windows:

```powershell
py -3.10 -m venv .venv
```

Linux/macOS:

```bash
python3.10 -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Start the Flask server

```powershell
python backend/app.py
```

Wait until the terminal shows:

```text
Model loaded successfully
Starting Pneumonia Detection API...
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser.

> Keep the terminal open while using the application. Press `Ctrl+C` to stop the server.

## Run Using Flask Directly

```powershell
cd backend
python app.py
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Opens the web application |
| `POST` | `/predict` | Uploads an X-ray image and returns the prediction and Grad-CAM heatmap |
| `GET` | `/metrics` | Returns model evaluation metrics |

Example prediction request:

```powershell
curl -X POST -F "file=@backend/test_xray.jpg" http://127.0.0.1:5000/predict
```

## Project Structure

```text
backend/       Flask API and Grad-CAM implementation
frontend/      Browser interface
model/         Trained model and evaluation metrics
dataset/       Training, validation, and test datasets
training/      Dataset preprocessing and model training/evaluation scripts
uploads/       Uploaded images grouped by prediction
tools/         Utility scripts
```

## Model Information

- Architecture: ResNet50
- Input size: `224 x 224`
- Classes: `NORMAL`, `PNEUMONIA`
- Prediction threshold: `0.6`
- Trained model: `model/resnet50_pneumonia.h5`

Evaluation metrics are stored in `model/metrics.json` and available through `/metrics`.