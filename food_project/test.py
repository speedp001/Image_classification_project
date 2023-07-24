import torch
import torchvision
import torch.nn as nn
import albumentations as A
import torch.nn.functional as F
#F 라이브러리를 사용하는게 image.Resize()보다 속도가 빠르다.


from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import mobilenet_v2
from customdataset import MyFoodDataset






def main() :
    
    device = torch.device("mps") # mac m1 or m2
    
    # model setting
    model = mobilenet_v2()
    model.classifier[1] = nn.Linear(1280, 20)

    # .pt load
    model.load_state_dict(torch.load(f="./food_project/best_food.pt"))
    # print(list(model.parameters()))

    
    test_transforms = A.Compose([
        A.SmallestMaxSize(max_size=250),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.6),
        A.RandomShadow(),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.4),
        A.RandomBrightnessContrast(p=0.5),
        A.Resize(height=224, width=224),
        ToTensorV2()
    ])
    
    test_dataset = MyFoodDataset("./food_project/food_dataset/test/", test_transforms)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model.to(device)
    model.eval()
    
    correct = 0
    
    with torch.no_grad() :
        for data, target in tqdm(test_loader) :
            data, target = data.float().to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    print("test set : Acc {}/{} [{:.0f}]%\n".format(correct, len(test_loader.dataset), 100 * correct / len(test_loader.dataset)))


if __name__ == "__main__" :
    main()