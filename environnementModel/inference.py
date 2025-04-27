import onnx
import onnxruntime

from torchvision import models
from torchvision.models import ResNet50_Weights

import torch
from torch import nn

import numpy as np
from PIL import Image

from dataset import DailyPlacesDataset
from torchvision import transforms

import matplotlib.pyplot as plt
import time

print("Loading ONNX Model...")
onnx_model = onnx.load(r"environnementModel\model\quantized.onnx")
onnx.checker.check_model(onnx_model)



print("Loading ONNX session...")
onnx_session = onnxruntime.InferenceSession(r"environnementModel\model\quantized.onnx", providers=["VitisAIExecutionProvider"],
                                            provider_options=[{"config_file":"vaip_config.json"}])

device = 'cuda' if torch.cuda.is_available() else 'cpu'                        

print("Loading dataset and aquiring classes...")
dataset = DailyPlacesDataset(root_dir="dataset\Places")
labels = dataset.getClasses()


while True:
    img_path = input('Enter image path : ')
    time_before = time.time()
    with torch.no_grad():
        img_test = Image.open(img_path)
        img_test = img_test.resize((224,224))

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image = transform(img_test)
        image = image.type(torch.float32)
        image = image.reshape(1, 3, 224, 224).numpy()
        print("Generating Outputs")

        image = {"input": image}
        model_outs = onnx_session.run(None, image)
        
        time_after = time.time()
        model_outs = torch.from_numpy(model_outs[0]).squeeze(0).argmax(0).item()
        print(f'{labels[model_outs]} ({round((time_after-time_before), 2)}s)')

        plt.imshow(img_test)
        plt.title(f'Prediction: {labels[model_outs]}')
        plt.show()

        print('--------------------------------------')