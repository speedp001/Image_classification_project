import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from customdataset import CustomDataset






def main() :
    
    DEVICE_MPS = torch.device("mps") # mac m1 or m2
    
    # model setting
    model = resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    # .pt load
    model.load_state_dict(torch.load(f="./paintings_project/best_paintings.pt"))
    # print(list(model.parameters()))

    
    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    
    test_dataset = CustomDataset("./paintings_project/data_art/val/", val_transforms)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model.to(DEVICE_MPS)
    model.eval()
    
    correct = 0
    
    with torch.no_grad() :
        for data, target in tqdm(test_loader) :
            data, target = data.to(DEVICE_MPS), target.to(DEVICE_MPS)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    print("test set : Acc {}/{} [{:.0f}]%\n".format(correct, len(test_loader.dataset), 100 * correct / len(test_loader.dataset)))


if __name__ == "__main__" :
    main()
    
    