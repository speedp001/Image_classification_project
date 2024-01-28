import torch
import argparse
import torchvision
import torch.nn as nn
import albumentations as A
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from torchvision.models.efficientnet import efficientnet_b0
from customdataset import MyUSLicensePlatesDataset






def main(args) :
    
    device = torch.device("mps") # mac m1 or m2
    
    # model setting
    model = efficientnet_b0()
    model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
    model.classifier[1] = nn.Linear(1280, out_features=50)
    model.to(device)

    # .pt load
    model.load_state_dict(torch.load(f="./US_license_plates_project/US_license_plates_best.pt"))
    # print(list(model.parameters()))

    
    test_transforms = A.Compose([
        A.Resize(width=224, height=224),
        ToTensorV2()
    ])

    # dataset and dataloader
    test_dataset = MyUSLicensePlatesDataset(args.test_dir, transform=test_transforms)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
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
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_dir", type=str, default="./US_license_plates_project/US_license_plates_dataset/test/",
                        help="directory path to the test dataset")
    
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for testing")
    
    args = parser.parse_args()
    
    main(args)
    