import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import DailyPlacesDataset
from torchvision import transforms
import time
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
writer = SummaryWriter()

print("Loading dataset and dataloader...")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = DailyPlacesDataset(root_dir = r'dataset/Places', transform=train_transform)
test_dataset = DailyPlacesDataset(root_dir = r'dataset/Places', transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

print(next(iter(train_dataloader))[0].shape)

print("Loading model...")
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
label_map = train_dataset.getClasses()

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, len(label_map))
)

model.load_state_dict(torch.load("environnementModel\model\dplaces.pth", map_location=torch.device(device))["weights_sd"])
for name, param in model.named_parameters():
    if "fc"in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model = model.to(device)


def train_epochs(epochs:int, model, device, criterion, optimizer, scheduler=None, save_weights:bool=False, weights_path=""):
    time_before = time.time()

    model.train()

    writer.add_graph(model, (next(iter(train_dataloader))[0]).to(device))

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
            
            if index % 10 == 0:
                writer.add_scalar('Loss/train', (running_loss/(index+1)), (epoch*len(train_dataloader)) + index)
                writer.add_scalar('Accuracy/train', 100*running_correct/total_samples, (epoch*len(train_dataloader)) + index)
                
                if scheduler:
                    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], step_count)
                
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
    lr_fc = 1e-4

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_fc, momentum=0.9, weight_decay=1e-4)
    
    if enable_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)
    
    if load_checkpoint:
        checkpoint = torch.load(load_path, map_location=torch.device(device))
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
            image = image.to(device)
            label = label.to(device)

            outputs = model(image)
            loss = criterion(outputs, label)

            preds = torch.argmax(outputs, 1)
            running_correct += torch.sum(preds == label.data)
            total_samples += image.size(0)
            running_loss += loss.item()
            
            if index %25 == 0:
                writer.add_scalar('Loss/val', running_loss/len(test_dataloader), (epoch*len(test_dataloader)) + index)
                writer.add_scalar('Accuracy/val', 100*running_correct/total_samples, (epoch*len(test_dataloader)) + index)
                print('Accuracy of', (100*running_correct/total_samples).item(), '%')

        
        print(f'Validation - Epoch {epoch} - Loss: {running_loss/len(test_dataloader):.4f} - Accuracy: {100.0 * running_correct / total_samples:.2f}%')
        return running_loss/len(test_dataloader)




print("Training started...")
train(model=model, n_epochs=15, device=device, enable_scheduler=False, load_checkpoint=True, load_path="environnementModel/model/dplaces.pth",
      save_checkpoint=False, save_path="environnementModel/model/dplaces2.pth")