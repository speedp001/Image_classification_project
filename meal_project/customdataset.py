import os
import glob
import cv2
import pandas as pd

from torch.utils.data import Dataset

class MealDataset(Dataset) :
    
    def __init__(self, csv_dir, transform=None) :
        self.csv_data = pd.read_csv(csv_dir)
        #self.txt_data = pd.read_csv(txt_dir, sep=" ", header=None)
        
        # csv 파일 내의 filepaths 칼럼에 파일명들이 저장되어 있으므로, 해당 내용을 데이터셋으로 사용            
        self.file_list_by_csv = self.csv_data['img_name'].to_list()
        
        
        # 파일 이름에 'train'이 포함되어 있다면 mode를 'train'으로, 'val'이 포함되어 있다면 mode를 'val'으로 설정
        if 'train' in csv_dir:
            self.file_dir = "./meal_project/meal_dataset/train_set/"
        elif 'val' in csv_dir:
            self.file_dir = "./meal_project/meal_dataset/val_set/"
        else:
            raise ValueError("Invalid CSV directory. It should contain 'train' or 'val' in the file name.")
            
        # 파일 이름과 경로를 결합하여 전체 파일 경로 리스트 생성
        self.file_full_path = [os.path.join(self.file_dir, filename) for filename in self.file_list_by_csv]
        
        # csv 파일의 라벨 정보
        self.label_list_by_csv = self.csv_data['label'].to_list()
            
        self.transform = transform
 
 
        # #Customdataset을 완성하니 깨달은 점은 라벨데이터가 int형식으로 지정되어있는 경우는 dict를 구성할 필요가 없다.
        # # 클래스 레이블과 클래스 ID를 매핑하는 라벨 딕셔너리 생성
        # # label이 폴더 이름이 아닌, txt 내부에 직접 명시되어 있으므로 해당 내용을 사용하면 됨       
        # self.label_column = self.txt_data.iloc[:, 1].tolist()

        
        # # txt를 읽어와서 label_idx 정의
        # self.class_id = self.txt_data.iloc[:, 0].tolist()

        # self.label_dict = {}
        # for label, label_idx in zip(self.label_column, self.class_id):
        #     self.label_dict[label] = label_idx
            
        
    def __getitem__(self, index) :
        
        image_filepath = self.file_full_path[index]
        img = cv2.imread(image_filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.label_list_by_csv[index]

        if self.transform is not None :
            image = self.transform(image=img)['image']
            
        return image, label
    
    def __len__(self) :
        return len(self.file_list_by_csv)