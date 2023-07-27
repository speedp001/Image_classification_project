import os
import cv2
import glob
from torch.utils.data import Dataset

#커스텀 데이터 셋 정의
class MyDataset(Dataset):
    
    def __init__(self, directory_data, transform=None):
        self.directory_data = glob.glob(os.path.join(directory_data, "*", "*.png"))
        self.transform = transform
        self.label_dictionary = self.create_label_dict()
        
    #라벨 딕셔너리 생성 함수
    def create_label_dict(self):
        
        label_dictionary = {}
        
        for filepath in self.directory_data:
            label = os.path.basename(os.path.dirname(filepath))
            
            if label not in label_dictionary:
                label_dictionary[label] = len(label_dictionary)

        return label_dictionary
    
    
    
    def __getitem__(self, item) :
        
        image_filepath = self.directory_data[item]
        
        img = cv2.imread(image_filepath)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = os.path.basename(os.path.dirname(image_filepath))
        #print(label)
        
        label_idx = self.label_dictionary[label]
        #print(label, label_idx)
        
        if self.transform is not None:
            image = self.transform(image = img)['image']
            
        return image, label_idx
        
        
     
    def __len__(self):
        return len(self.directory_data)
    
    

# #test run
# test = MyDataset("./metal_project/metal_dataset/train", transform=None)
# for i in test:
#     print(i)