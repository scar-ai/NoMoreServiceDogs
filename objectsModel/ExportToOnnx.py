import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image

image_path = "input.jpg"

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1).eval()


with torch.no_grad():
    image = Image.open(image_path).convert("RGB")
        

    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    print(image_tensor.shape)

    onnx_program =torch.onnx.export(
    model,
    torch.randn(1, 3, 1080, 1920),
    r"objectsModel\model\fasterrcnn_resnet50_fpn.onnx",
    opset_version=17,
    do_constant_folding=True,
    export_params=True,
    input_names=["input"],
    output_names=["output"])
