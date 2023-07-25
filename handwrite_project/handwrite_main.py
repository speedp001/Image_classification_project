import torch
import numpy as np
import torchvision
import torch.nn as nn
import handwrite_utils as handwrite
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.optim import AdamW
from lion_pytorch import Lion
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from handwrite_model import CNN
from sklearn.model_selection import train_test_split
from handwrite_customdataset import HandwriteDataset



if __name__ == "__main__":
    
    device = torch.device('mps')

    #모델 정보를 device에게 넘긴다.
    model = CNN().to(device)
    
    data_transform = transforms.Compose([
        #이미지 크기조정
        transforms.Resize((28, 28)),
        #텐서화
        transforms.ToTensor()
    ])
    
    image_path = "./handwrite_project/handwrite_data/"
    dataset = HandwriteDataset(image_path, transform=data_transform)

    # 데이터를 리스트로 변환
    image_list = []
    label_list = []

    for image, label in dataset:
        image_list.append(image)
        label_list.append(int(label))
        #print(label)
        # print(image_list)
        # print(label_list)
        # print(f"Image : {image}, Label : {label}")

    #train data와 test data를 분할해준다.
    x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state=2023)
    
    
    train_loader = DataLoader(list(zip(x_train, y_train)), batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(list(zip(x_test, y_test)), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    
    #옵티마이저, 손실함수, 에포크 정의    
    num_epoch = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    handwrite.train(model, len(list(zip(x_train, y_train))), train_loader,
                    criterion, optimizer, num_epoch, device)
    
    handwrite.eval(model, test_loader, device)
    
    
    
    
    #코드가 비효율적이라고 생각하는 이유 : CustomDataset을 여러번 호출하여야 정상적으로 데이터를 분할할 수 있음. 
    # zip형태로 묶어서 진행을 하여도 전체 데이터를 나누는게 더 효율적임.
    #코드 리뷰 시 -> 전체 데이터셋에서 일정 비율을 지정하여 random_split함수를 사용하여 train과 test 데이터를 분할할 수 있다.
    #기존 dataset에서 sklearn split을 적용한 뒤 train과 test dataset을 따로 customdataset을 돌리는 것이 더 효율적이다.
    
    
    # 간결화 코드
    # total_dataset = HandwriteDataset(
    #     "./handwrite_project/handwrite_data/",
    #     transform=data_transform
    # )

    # test_len = int(len(total_dataset) * 0.2)
    # train_len = len(total_dataset) - test_len

    # train_subset, test_subset = random_split(
    #     total_dataset, 
    #     [train_len, test_len]
    # )
    
    
    
    
    
    
    
    
    




