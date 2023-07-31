import os
import glob

from torch.utils.data import Dataset
from PIL import Image, ImageFile




class CustomData(Dataset) :
    def __init__(self, data_dir, transforms=None) :
        
        #data_dir -> ./ex_03/dataset/train/
        self.data_dir = glob.glob(os.path.join(data_dir, "*", "*.jpg"))
        self.transforms = transforms
        self.label_dict = {'Chickenpox' : 0, 'Cowpox' : 1, 'Healthy' : 2, 'HFMD' : 3,
                           'Measles' : 4, 'Monkeypox' : 5
                        }
        
        
    def __getitem__(self, item) :
        image_path = self.data_dir[item]
        label_name = image_path.split("/")[4]
        label = self.label_dict[label_name]
        #print(label)
        
        
        #print(image_path)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.transforms is not None :
            image= self.transforms(image)

        return image, label
        
        
        
    def __len__(self) :
        return len(self.data_dir)