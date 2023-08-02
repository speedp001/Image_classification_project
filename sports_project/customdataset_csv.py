import os
import glob
import cv2
import pandas as pd
from torch.utils.data import Dataset




class SportsDataset_csv(Dataset) :
    
    def __init__(self, csv_dir, mode=None, transform=None):
        self.csv_data = pd.read_csv(csv_dir)
        
        # csv 파일을 읽어서 구성하는 경우
        self.csv_data = self.csv_data.loc[self.csv_data['data set'] == mode]
        
        #csv안에 .lnk파일이 포함된 오류가 있으므르, 해당 파일을 걸러내는 작업 진행
        for i, item in enumerate(self.csv_data['filepaths']):
            if item.endswith(".lnk"):
                #axis=0은 행 방향
                self.csv_data.drop(index=i, axis=0, inplace=True)
                self.csv_data.reset_index(drop=True, inplace=True)
                #inplace=True로 설정되어 있으므로, drop() 메서드가 호출된 후에 self.csv_data 데이터프레임이 직접 변경
        
        self.file_list_by_csv = self.csv_data['filepaths'].to_list()
        # csv 파일 내의 filepaths 칼럼에 파일명들이 저장되어 있으므로, 해당 내용을 데이터셋으로 사용
        # 나중에 __len__ 함수에서 len(self.file_list_by_csv)

        # label이 폴더 이름이 아닌, csv 내부에 직접 명시되어 있으므로 해당 내용을 사용하면 됨
        self.label_column = self.csv_data['labels'].to_list()
        # class id가 이미 숫자로 표기되어 있으므로, dictionary 구성이 필요 없음
        self.class_id = self.csv_data['class id'].to_list()


        self.transform = transform
    
    def __getitem__(self, index) :
        
        file_path = os.path.join("./sports_project/sports_dataset/", self.file_list_by_csv[index])
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = self.class_id[index]
        
        if self.transform is not None :
            img = self.transform(image=img)['image']
            
        return img, label
        
    
    def __len__(self) :
        return len(self.file_list_by_csv)
    
    
# if __name__ == "__main__":
#     test = SportsDataset_csv("./sports_project/sports_dataset/sports.csv", mode="valid")
    
#     for item in test:
#         print(item)