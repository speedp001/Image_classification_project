import cv2
import os
import glob
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms





class HandwriteDataset(Dataset):
    
    def __init__(self, file_paths, transform=None):
        
        self.file_paths = file_paths
        self.file_paths = glob.glob(os.path.join(file_paths, "*.[jp][pn]g"))
        print(self.file_paths)
        
        #변형 객체 class 내에서 선언
        self.transforms = transform
        
        
        
    def __getitem__(self, item):
        
        #str형태로 이미지 경로 읽어온 뒤 RGB형태로 변형
        file_path: str = self.file_paths[item]
        image = cv2.imread(file_path)
        
        if image is None:
            # 이미지 파일을 읽을 수 없는 경우, 예외 처리 또는 다른 대체 동작 수행
            # 예: 이미지 파일 경로 출력 또는 빈 이미지 반환
            print(f"Failed to read image: {file_path}")
            return None, None
        

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = Image.fromarray(image)
        #cv2로 읽어온 이미지는 배열 형태이기 때문에 transform을 적용하려면 PIL형식으로 바꿔줘야한다.
        
        folder_name = file_path.split("/")[2]
        #print(folder_name)
        file_name = file_path.split("/")[3]
        #print(file_name)
        label_temp = file_name.split("_")[1]
        label = label_temp.split(".")[0]
        #print(label)
        
        if self.transforms is not None :
            image= self.transforms(image)
            

        return image, label
            
    
    
    def __len__(self):
        return len(self.file_paths)
    

