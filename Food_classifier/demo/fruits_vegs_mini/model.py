from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torchvision.models import EfficientNet_B1_Weights
from PIL import Image
from time import perf_counter as timer


CLASS_NAMES = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def create_model(num_classes):

    model = torchvision.models.efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(
        in_features=model.classifier[1].in_features,
        out_features=num_classes
    )
    return model

def load_model(model_path='effnetb1_6_epochs.pth',device='cpu'):

    model = create_model(num_classes=len(CLASS_NAMES))

    model.load_state_dict(torch.load(model_path,map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(
    image: Image.Image,
    model,
    device = "cpu"):
    
    start_time = timer()

    # Move model to the correct device
    model.to(device)
    model.eval()

    # Transform the image
    image_tensor = test_transforms(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]  # shape: [num_classes]

    pred_time = round(timer() - start_time, 5)

    # Create a dictionary mapping class_names to probabilities
    results = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return results, pred_time