# 🔍 Image Classifier

An AI-powered web app that identifies the main subject in any image using a pretrained ResNet50 neural network.

## Features
- Upload any image and get instant object recognition
- Returns top 5 predictions with confidence scores
- Built with PyTorch, FastAPI, and vanilla JavaScript

## Tech Stack
- **Backend:** Python, FastAPI, PyTorch, Torchvision
- **Frontend:** HTML, CSS, JavaScript
- **Model:** ResNet50 pretrained on ImageNet (1,000 classes)

## Getting Started

### 1. Clone the repo
git clone https://github.com/yourusername/image-classifier.git
cd image-classifier

### 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

### 3. Install dependencies
pip install torch torchvision fastapi uvicorn python-multipart pillow

### 4. Run the app
uvicorn main:app --reload

### 5. Open in browser
http://localhost:8000

## How It Works
1. User uploads an image via the web interface
2. JavaScript sends the image to the FastAPI backend
3. PyTorch runs it through ResNet50
4. Top 5 predictions are returned and displayed with confidence bars
