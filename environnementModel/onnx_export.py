import torch
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch import nn
from dataset import DailyPlacesDataset
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet50_Weights



device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_path = r"dataset\Places\Places__agriculture-land\img\055ea879957955853f112f9199cd38af28743f59.jpg"
test_dataset = DailyPlacesDataset(root_dir = r'dataset/Places')
print(next(iter(test_dataset))[0].shape)

print("Loading model...")
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
label_map = test_dataset.getClasses()

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, len(label_map))
)
model.load_state_dict(torch.load("environnementModel\model\dplaces.pth", map_location=torch.device(device))["weights_sd"])
model.fc[1]=nn.Identity()
model.eval()

print(model)

with torch.no_grad():
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    print(image_tensor.shape)

    onnx_program =torch.onnx.export(
    model,
    image_tensor,
    r"environnementModel\model\resnet50_finetuned.onnx",
    opset_version=17,
    do_constant_folding=True,
    export_params=True,
    input_names=["input"],
    output_names=["output"])