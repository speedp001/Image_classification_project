import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
#F 라이브러리를 사용하는게 image.Resize()보다 속도가 빠르다.
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import mobilenet_v2
from customdataset import CustomData






def main() :
    
    device = torch.device("mps") # mac m1 or m2
    
    # model setting
    model = mobilenet_v2()
    in_features_ = 1280
    model.classifier[1] = nn.Linear(in_features_, 15)
    model.to(device)

    # .pt load
    model.load_state_dict(torch.load(f="./grass_project/best_grass.pt"))
    # print(list(model.parameters()))

    
    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    
    test_dataset = CustomData("./grass_project/grass_dataset/val/", val_transforms)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model.to(device)
    model.eval()
    
    correct = 0
    
    with torch.no_grad() :
        for data, target in tqdm(test_loader) :
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    print("test set : Acc {}/{} [{:.0f}]%\n".format(correct, len(test_loader.dataset), 100 * correct / len(test_loader.dataset)))


if __name__ == "__main__" :
    main()