import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import urllib.request
import json

# Load pretrained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# Load ImageNet class labels
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
with urllib.request.urlopen(url) as f:
    labels = json.load(f)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def classify_image(image: Image.Image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top5 = torch.topk(probabilities, 5)
    results = []
    for prob, idx in zip(top5.values, top5.indices):
        results.append({
            "label": labels[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })
    return results