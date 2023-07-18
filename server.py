# 1. Library imports
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import pickle
from PIL import Image
from io import BytesIO
import pandas as pd

app = FastAPI()

origins = ["https://64b406138584bd41eea70886--shimmering-frangipane-5bd0d2.netlify.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

device = torch.device("cpu")
model = Net()
model.load_state_dict(torch.load('devanagiri_weights_99_67', map_location=device))
model.eval()
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.get('/')
def index():
    return {'message': 'Hello, World'}

def load_pil_image(data):
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    pil_image = Image.fromarray(frame)
    return pil_image


@app.post('/predict')
async def predict_banknote(file:UploadFile=File(...)):

    # file_location = f"./saved_file/{file.filename}"
    # with open(file_location, "wb+") as file_object:
    #     file_object.write(file.read())

    pil_image = load_pil_image(await file.read())

    # transformedImage = torch.unsqueeze(transform(pil_image), 0)
    # # print(transformedImage.shape)
    # prediction = model(transformedImage)
    # _, predicted = torch.max(prediction.data, 1)
    # print("Prediction: ", prediction)
    # # print(prediction, predicted.item())
    # return {
    #     "predictedNumber": predicted.item(),
    # }

    print(model)

    return {
        "prefictedNumber": 11
    }

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload