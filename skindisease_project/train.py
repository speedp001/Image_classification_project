import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from customdataset import CustomData

from tqdm import tqdm
from torch.optim import AdamW
from lion_pytorch import Lion
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models.efficientnet import efficientnet_v2_s

def train(model, train_loader, val_loader, epochs, optimizer, criterion, device) :
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    print("Train ....")
    for epoch in range(epochs) :
        train_loss = 0.0
        val_loss = 0.0 
        val_acc = 0.0
        train_acc = 0.0

        model.train()
        # tqdm
        train_loader_iter = tqdm(train_loader, desc=(f"Epoch : {epoch + 1}/{epochs}"), leave=False)

        for i, (data, target) in enumerate(train_loader_iter) :
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # acc
            _, pred = torch.max(outputs, 1)
            train_acc += (pred == target).sum().item()

            train_loader_iter.set_postfix({"Loss" :  loss.item()})

        train_loss /= len(train_loader)
        train_acc = train_acc / len(train_loader.dataset)
        """train_loss /= len(train_loader)는 한 에포크에서의 총 손실 값을 배치 개수로 나누어 평균 손실 값을 계산합니다. 
        이렇게 함으로써 한 에포크에서의 평균 손실 값을 얻을 수 있습니다. """
        
        """len(train_loader.dataset)은 훈련 데이터셋의 총 샘플 개수를 반환합니다."""
        

        # eval
        model.eval()
        with torch.no_grad() :
            for data, target in val_loader :
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                #keepdim=True는 결과 텐서의 차원을 유지
                val_acc += pred.eq(target.view_as(pred)).sum().item()
                #target.view_as(pred)는 target 텐서를 pred 텐서와 동일한 크기로 변형하는 작업
                val_loss += criterion(output, target).item()

        val_loss /= len(val_loader)
        val_acc = val_acc / len(val_loader.dataset)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # save model
        if val_acc > best_val_acc :
            torch.save(model.state_dict(), "./skindisease_project/best_skindisease.pt")
            best_val_acc = val_acc
        print(f"Epoch [{epoch + 1} / {epochs}] , Train loss [{train_loss:.4f}],"
              f"Val loss [{val_loss :.4f}], Train ACC [{train_acc:.4f}],"
              f"Val ACC [{val_acc:.4f}]")

    #torch.save(model.state_dict(), "./skindisease_project/ex_last.pt")
    return  model, train_losses, val_losses, train_accs, val_accs

def main() :
    device = torch.device("mps")

    model = efficientnet_v2_s(pretrained=True)
    in_features_ = 1280
    model.classifier[1] = nn.Linear(in_features_, 6)
    model.to(device)

    # aug
    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(degrees=15),
        #transforms.ColorJitter(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.2,0.2,0.2))
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
    ])

    # dataset dataloader
    train_dataset = CustomData("./skindisease_project/skindisease_dataset/train/", transforms=train_transforms)
    val_dataset = CustomData("./skindisease_project/skindisease_dataset/valid/", transforms=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # loss function optimizer, epochs
    epochs = 20
    criterion = CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(),lr=0.001 , weight_decay=1e-2)

    train(model,train_loader,val_loader,epochs,optimizer,criterion,device)

if __name__ == "__main__" :
    main()