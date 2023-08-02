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
from customdataset_csv import SportsDataset_csv






def main(args) :
    
    device = torch.device("mps") # mac m1 or m2
    
    # model setting
    model = efficientnet_b0()
    model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
    model.classifier[1] = nn.Linear(1280, out_features=99)
    model.to(device)

    # .pt load
    model.load_state_dict(torch.load(f="./sports_project/sports_best.pt"))
    # print(list(model.parameters()))

    
    test_transforms = A.Compose([
        A.Resize(width=224, height=224),
        ToTensorV2()
    ])

    # dataset and dataloader
    test_dataset = SportsDataset_csv(args.csv_dir, mode="test", transform=test_transforms)

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
            #.sum()을 호출하면 True의 개수를 세어주게 됩니다.
            #.item()을 호출하는 이유는 .sum()이 스칼라 값을 반환하며, 해당 값을 파이썬의 정수로 변환하기 위해서입니다. 
            
    print("test set : Acc {}/{} [{:.0f}]%\n".format(correct, len(test_loader.dataset), 100 * correct / len(test_loader.dataset)))


if __name__ == "__main__" :
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--csv_dir", type=str, default="./sports_project/sports_dataset/sports.csv",
                        help="directory path to the test dataset")
    
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for testting")
    
    args = parser.parse_args()
    
    main(args)