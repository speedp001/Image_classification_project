import os
import numpy as np
import json
import glob
import torch

from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset




#라벨 정보 리스트에 저장
labels = ['crease', 'crescent_gap', 'inclusion', 'oil_spot', 'punching_hole',
          'rolled_pit', 'silk_spot', 'waist_folding', 'water_spot', 'welding_line']

#학습에 사용될 데이터 셋 생성
def crop_and_save_image(path_json_file, dir_output, train_ratio=0.9) :
    #json파일 로드
    with open(path_json_file, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    # print(data_json)
    # exit()
    
    #데이터 폴더 생성
    dir_train = os.path.join(dir_output, "train")
    dir_valid = os.path.join(dir_output, "valid")
    os.makedirs(dir_train, exist_ok=True)
    os.makedirs(dir_valid, exist_ok=True)
    
    #데이터 폴더 안 라벨 폴더 생성
    for label in labels:
        dir_labels_train = os.path.join(dir_train, label)
        dir_labels_valid = os.path.join(dir_valid, label)
        os.makedirs(dir_labels_train, exist_ok=True)
        os.makedirs(dir_labels_valid, exist_ok=True)
        
    #데이터 정보 입력
    for information in tqdm(data_json.keys()):
        #print(type(information))
        #스트링 형태로 정보저장
        json_image = data_json[information]
        # print(json_image)
        """
        {'filename': 'img_08_4406772100_00002.jpg', 'width': 2048, 'height': 1000, 'anno': 
        {'label': 'crescent_gap', 'bbox': [46, 609, 294, 998]}]}
        """
        
        filename = json_image['filename']
        width = json_image['width']
        height = json_image['height']
        bboxes = json_image['anno']
        #print(filename, width, height, bboxes)

        #이미지 로드
        path_image = os.path.join("./metal_project/metal_org_data/images", filename)
        image = Image.open(path_image)
        image = image.convert("RGB")
        #print(path_image)
        
        
        #이미지 크롭
        #한개의 이미지안에 여러개의 bbox가 있을 수 있으므로 bbox_idx으로 순회
        for bbox_idx, bbox in enumerate(bboxes):
            #라벨 이름과 좌상단 우하단 좌표를 따온다
            label_name = bbox['label']
            bbox_xy = bbox['bbox']
            x1, y1, x2, y2 = bbox_xy
            #print(label_name, bbox_xy)
            
            #추출한 박스위치의 이미지를 크롭
            image_cropped = image.crop((x1, y1, x2, y2))
            
            #크롭한 이미지에 패딩 작업
            width_, height_ = image_cropped.size
            if width_ > height_ :
                image_padded = Image.new(image_cropped.mode, (width_, width_), (0,))
                padding = (0, int((width_ - height_) / 2))
                
            else :
                image_padded = Image.new(image_cropped.mode, (height_, height_), (0,))
                padding = (int((height_ - width_) / 2), 0)
                
            image_padded.paste(image_cropped, padding)
            
            #이미지 리사이징
            size = (255, 255)
            resize_image = F.resize(image_cropped, size)
            
            #train
            if np.random.rand() < train_ratio:
                dir_save = os.path.join(dir_train, label_name)
                
            else:
                dir_save = os.path.join(dir_valid, label_name)
            
            
            
            final_save = os.path.join(dir_save, f"{filename}_{label_name}_{bbox_idx}.png")
            image_padded.save(final_save)
            
        
        
        
        
        
#코드 실행
if __name__ == "__main__" :
    path_json_file = "./metal_project/metal_org_data/anno/annotation.json"
    dir_output = "./metal_project/metal_dataset/"
    
    crop_and_save_image(path_json_file, dir_output)