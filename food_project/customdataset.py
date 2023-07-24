import os
import cv2
import glob
from torch.utils.data import Dataset

class MyFoodDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = glob.glob(os.path.join(data_dir, "*", "*.jpg"))
        self.transform = transform
        self.label_dict = self.create_label_dict()

    def create_label_dict(self):
        label_dict = {}
        for filepath in self.data_dir:
            label = os.path.basename(os.path.dirname(filepath))
            #os.path.dirname(filepath)은 해당 파일의 디렉토리 경로를 반환
            #os.path.basename()은 디렉토리 경로에서 마지막 디렉토리 이름을 추출
            if label not in label_dict:
                label_dict[label] = len(label_dict)
                # 첫 번째 등장하는 라벨은 인덱스 0을 가지고, 두 번째 등장하는 라벨은 인덱스 1을 가지게 됩니다. 
                # 이런 식으로 등장하는 라벨의 개수에 따라 인덱스가 자동으로 할당됩니다.
        
        return label_dict

    def __getitem__(self, item):
        image_filepath = self.data_dir[item]
        img = cv2.imread(image_filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = os.path.basename(os.path.dirname(image_filepath))
        label_idx = self.label_dict[label]

        if self.transform is not None:
            image = self.transform(image=img)['image']
            # 딕셔너리의 키 'image'에 대한 값을 가져와서 image 변수
        else:
            image = img  # 이미지를 변환하지 않고 원본 이미지를 할당

        """self.transform(image=img)를 호출
        
        {
        'image': transformed_image,  # 변환된 이미지
        'info': additional_info,     # 추가 정보
        'metadata': metadata         # 메타 데이터
        }

        """

        return image, label_idx

    def __len__(self):
        return len(self.data_dir)


# test = MyFoodDataset("./food_project/food_dataset/train", transform=None)
# for i in test:
#     print(i)
