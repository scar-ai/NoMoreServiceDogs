import onnx
import onnxruntime

from PIL import Image, ImageDraw

from torchvision import transforms

import time

print("Loading ONNX Model...")
onnx_model = onnx.load(r"objectsModel\model\quantized.onnx")
onnx.checker.check_model(onnx_model)

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]



print("Loading ONNX session...")
onnx_session = onnxruntime.InferenceSession(r"objectsModel\model\quantized.onnx", providers=["VitisAIExecutionProvider"],
                                            provider_options=[{"config_file":"vaip_config.json"}])


time_before = time.time()
image = Image.open(r"input.jpg").convert("RGB").resize((1920, 1080))
    

transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0).numpy()

print("Generating Outputs")
image = {"input": image_tensor}
model_outs = onnx_session.run(None, image)


print('-------------')
time_after = time.time()

image = Image.open(r"input.jpg").convert("RGB").resize((1920, 1080))
draw = ImageDraw.Draw(image)

boxes = model_outs[0]
labels = model_outs[1]
scores = model_outs[2]


for box, label, score in zip(boxes, labels, scores):
            if score > 0.2: 
                print(COCO_INSTANCE_CATEGORY_NAMES[label], "-", score)
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
                draw.text((box[0], box[1]),
                          f"Class: {COCO_INSTANCE_CATEGORY_NAMES[label]}\nScore: {round(score.item(), 2)}", fill="red")



print(f'Done! ({round((time_after-time_before), 2)}s)')

image.save("output.jpg")
image.show()
print('--------------------------------------')