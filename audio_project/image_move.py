import os
import shutil
import glob
from sklearn.model_selection import train_test_split




class ImageDataMove :
    def __init__(self, org_dir, train_dir, val_dir):
        self.org_dir = org_dir
        self.train_dir = train_dir
        self.val_dir = val_dir

    def move_images(self):

        # file path list
        file_path_list01 = glob.glob(os.path.join(self.org_dir, "waveshow", "*", "*.png"))
        file_path_list02 = glob.glob(os.path.join(self.org_dir, "STFT", "*", "*.png"))
        file_path_list03 = glob.glob(os.path.join(self.org_dir, "MelSepctrogram", "*", "*.png"))

        # data split
        wa_train_data_list , wa_val_data_list = train_test_split(file_path_list01, test_size=0.2)
        st_train_data_list , st_val_data_list = train_test_split(file_path_list02, test_size=0.2)
        ms_train_data_list , ms_val_data_list = train_test_split(file_path_list03, test_size=0.2)

        # file move
        self.move_file(wa_train_data_list, os.path.join(self.train_dir, "waveshow"))
        self.move_file(wa_val_data_list, os.path.join(self.val_dir, "waveshow"))
        self.move_file(st_train_data_list, os.path.join(self.train_dir, "STFT"))
        self.move_file(st_val_data_list, os.path.join(self.val_dir, "STFT"))
        self.move_file(ms_train_data_list, os.path.join(self.train_dir, "MelSepctrogram"))
        self.move_file(ms_val_data_list, os.path.join(self.val_dir, "MelSepctrogram"))

    def move_file(self, file_list, mov_dir):
        os.makedirs(mov_dir, exist_ok=True)
        for file_path in file_list:
            shutil.move(file_path, mov_dir)

org_dir = "./audio_project/audio_org_data/"
train_dir = "./audio_project/audio_data/train/"
val_dir = "./audio_project/audio_data/val/"

move_temp = ImageDataMove(org_dir, train_dir, val_dir)
move_temp.move_images()