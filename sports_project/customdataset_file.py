import os
import glob
import cv2
import pandas as pd
from torch.utils.data import Dataset




class SportsDataset_file(Dataset) :
    
    def __init__(self, data_dir, transform=None):
        
        self.file_list = glob.glob(os.path.join(data_dir, "*", "*", "*.jpg"))
        #파일 경로를 받아와서 하위 폴더 밑의 이미지 파일을 list로 받도록 저장
        
        self.transform = transform
        #인자로 받은 transform을 내부 값으로 저장
        
        #label이 숫자값으로 반환되어야하기 때문에 dictionary를 이용해서
        #스포츠 이름 문자에 해당하는 숫자를 반환하도록 할 필요가 있음
        self.label_dict = self.create_label_dict()
        
        
    def create_label_dict(self) :
        
        label_dict = {}
        
        for filepath in self.file_list :
            label = os.path.dirname(filepath)  #상위 폴더 이름까지 모두 불러옴
            label = os.path.basename(label)  #마지막 디렉토리 이름만 정리
            
            if label not in label_dict :
                label_dict[label] = len(label_dict)
                
        return label_dict
    
    
    def construct_dataset_by_csv(self, csv_dir, transform=None):
        # csv 파일을 읽어서 구성하는 경우
        # 편의상 별도 함수로 나눴지만 이 함수가 생성자가 될 것임 !!!
        self.csv_data = pd.read_csv(csv_dir)
        self.file_list_by_csv = self.csv_data['filepaths'] 
        # csv 파일 내의 filepaths 칼럼에 파일명들이 저장되어 있으므로, 해당 내용을 데이터셋으로 사용
        # 나중에 __len__ 함수에서 len(self.file_list_by_csv)

        # label이 폴더 이름이 아닌, csv 내부에 직접 명시되어 있으므로 해당 내용을 사용하면 됨
        self.label_column = self.csv_data['labels']
        # class id가 이미 숫자로 표기되어 있으므로, dictionary 구성이 필요 없음
        self.class_id = self.csv_data['class id']

        self.transform = transform
    
    
    def __getitem__(self, index) :
        
        image = cv2.imread(self.file_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = os.path.basename(os.path.dirname(self.file_list[index]))
        label_idx = self.label_dict[label]
        
        if self.transform is not None :
            image = self.transform(img=image)['image']  #albumentation transform 사용시
        
        
        return image, label_idx
    
    def __len__(self) :
        return len(self.file_list)
    
    
# if __name__ == "__main__" :
    
#     test = SportsDataset_file("./sports_project/sports_dataset/")
#     for i in test :
#         print(i)
