import torch
from torch import nn
import torch.fx as fx
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from pytorch_nndct import nn as nndct_nn
from pytorch_nndct.nn.modules import functional
from pytorch_nndct import QatProcessor

import time

from torch.utils.data import DataLoader
from dataset import DailyPlacesDataset

from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import ResNet50_Weights
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading dataset...")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




train_dataset = DailyPlacesDataset(root_dir=r'dataset/Places', transform=train_transform)
test_dataset = DailyPlacesDataset(root_dir=r'dataset/Places')

label_map = test_dataset.getClasses()

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

print('Loading model...')
time_before = time.time()

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               downsample=None,
               groups=1,
               base_width=64,
               dilation=1,
               norm_layer=None):
    super(BasicBlock, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 64:
      raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
      raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.relu1 = nn.ReLU(inplace=True)
    #self.relu1 = nn.functional.relu
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

    self.skip_add = functional.Add()
    self.relu2 = nn.ReLU(inplace=True)

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out = self.skip_add(out, identity)
    out = self.relu2(out)

    return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu1 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.skip_add = functional.Add()
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip_add(out, identity)
        out = self.relu3(out)

        return out

class ResNet(nn.Module):

  def __init__(self,
               block,
               layers,
               num_classes=1000,
               zero_init_residual=False,
               groups=1,
               width_per_group=64,
               replace_stride_with_dilation=None,
               norm_layer=None):
    super(ResNet, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    self._norm_layer = norm_layer

    self.inplanes = 64
    self.dilation = 1
    if replace_stride_with_dilation is None:
      # each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError(
          "replace_stride_with_dilation should be None "
          "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    self.groups = groups
    self.base_width = width_per_group
    self.conv1 = nn.Conv2d(
        3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = norm_layer(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(
        block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer(
        block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer(
        block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    self.quant_stub = nndct_nn.QuantStub()
    self.dequant_stub = nndct_nn.DeQuantStub()

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck):
          nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)

  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes * block.expansion, stride),
          norm_layer(planes * block.expansion),
      )

    layers = []
    layers.append(
        block(self.inplanes, planes, stride, downsample, self.groups,
              self.base_width, previous_dilation, norm_layer))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              groups=self.groups,
              base_width=self.base_width,
              dilation=self.dilation,
              norm_layer=norm_layer))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.quant_stub(x)

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    x = self.dequant_stub(x)
    return x


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def resnet50(*, weights, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__."""
    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, len(label_map))
)

model.load_state_dict(torch.load(r"environnementModel/model/dplaces.pth", map_location=torch.device('cpu'))['weights_sd'])

for name, param in model.named_parameters():
    if "fc"in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


model.fc[1] = nn.Identity() #batch norm 1d is not quantizable


label_map = test_dataset.getClasses()


qat_processor = QatProcessor(model, torch.randn([1, 3, 224, 224]), bitwidth=8)
model = qat_processor.trainable_model()


def train_epochs(epochs:int, model, device, criterion, optimizer, scheduler=None, save_weights:bool=False, weights_path=""):
    time_before = time.time()

    model.train()

    step_count = 0

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        
        for index, (image, label) in enumerate(train_dataloader):
            step_count += 1
            image = image.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()

            outputs = model(image)
            loss = criterion(outputs, label)

            preds = torch.argmax(outputs, 1)
            running_correct += torch.sum(preds == label.data)
            total_samples += image.size(0)
            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            
            if index % 1 == 0:
                for param_group in optimizer.param_groups:
                    clr = param_group['lr']
                    
                if save_weights:
                    if scheduler:
                        torch.save({'weights_sd':model.state_dict(),
                                    'optimizer_sd':optimizer.state_dict(),
                                    'scheduler_sd': scheduler.state_dict()}, weights_path)
                        
                    else:
                        torch.save({'weights_sd':model.state_dict(),
                                    'optimizer_sd':optimizer.state_dict()}, weights_path)
                
                time_current = time.time()
                print(f'Step {index}/{len(train_dataloader)} - Loss: {running_loss/(index+1):.4f} - Accuracy: {100.0 * running_correct / total_samples:.2f}% - Learning rate: {clr} - Time elapsed: {time_current-time_before:.2f} seconds')
                
        
        time_current = time.time()
        print('--------------------------------------------')
        print(f'Epoch {epoch}/{epochs} - Loss: {running_loss/len(train_dataloader):.4f} - Time elapsed: {time_current-time_before:.2f} seconds')
        print('--------------------------------------------')

        val_loss = validate(epoch, model, device, criterion)
        if scheduler:
                scheduler.step(val_loss)



def train(model, n_epochs:int, device, enable_scheduler:bool, load_checkpoint=False, load_path="", save_checkpoint=False, save_path=""):
    lr_fc = 1e-5

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_fc, weight_decay=1e-4)
    
    if enable_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)
    
    if load_checkpoint:
        checkpoint = torch.load(load_path, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer_sd'])
        if enable_scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_sd'])
    
    train_epochs(epochs=n_epochs, model=model, device=device, criterion=criterion, optimizer=optimizer,
                 save_weights=save_checkpoint, weights_path=save_path)



def validate(epoch, model, device, criterion):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for index, (image, label) in enumerate(test_dataloader):
            if index >= 250:
               break
            image = image.to(device)
            label = label.to(device)

            outputs = model(image)
            loss = criterion(outputs, label)

            preds = torch.argmax(outputs, 1)
            running_correct += torch.sum(preds == label.data)
            total_samples += image.size(0)
            running_loss += loss.item()
            
            if index %25 == 0:
                print('Accuracy of', (100*running_correct/total_samples).item(), '%')

        
        print(f'Validation - Epoch {epoch} - Loss: {running_loss/len(test_dataloader):.4f} - Accuracy: {100.0 * running_correct / total_samples:.2f}%')
        return running_loss/len(test_dataloader)




print("Training started...")
train(model=model, n_epochs=1, device=device, enable_scheduler=False, load_checkpoint=False, load_path="environnementModel/model/dplaces.pth",
      save_checkpoint=False, save_path="")

output_dir = 'qat_result'
deployable_model = qat_processor.to_deployable(model,output_dir)

validate(1, deployable_model, device, nn.CrossEntropyLoss())
print('Exporting model to ONNX...')
qat_processor.export_onnx_model(output_dir=output_dir)