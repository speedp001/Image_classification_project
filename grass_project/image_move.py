import os
import glob
import random
import shutil
from sklearn.model_selection import train_test_split

# org data path
label_folder_path = "./grass_project/grass_org_data"

# new data path
dataset_folder_path = "./grass_project/grass_dataset"

# train or val folder path
train_folder_path = os.path.join(dataset_folder_path, "train")
val_folder_path = os.path.join(dataset_folder_path, "val")

# train or val folder create
os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(val_folder_path, exist_ok=True)

# org_folder
org_folders = os.listdir(label_folder_path)

for org_folder in org_folders:
    if org_folder == ".DS_Store":  # Skip .DS_Store file
        continue

    org_folder_full_path = os.path.join(label_folder_path, org_folder)
    images = os.listdir(org_folder_full_path)
    random.shuffle(images)

    # label folder create
    train_label_folder_path = os.path.join(train_folder_path, org_folder)

    # val folder create
    val_label_folder_path = os.path.join(val_folder_path, org_folder)
    os.makedirs(train_label_folder_path, exist_ok=True)
    os.makedirs(val_label_folder_path, exist_ok=True)

    # image -> train folder move
    split_index = int(len(images) * 0.9)
    for image in images[:split_index]:
        if image == ".DS_Store":  # Skip .DS_Store file
            continue

        src_path = os.path.join(org_folder_full_path, image)  # org image path
        dst_path = os.path.join(train_label_folder_path, image)  # ./grass_project/grass_dataset/train/folder/image.png
        shutil.copyfile(src_path, dst_path)

    # image -> val folder move
    for image in images[split_index:]:
        if image == ".DS_Store":  # Skip .DS_Store file
            continue

        src_path = os.path.join(org_folder_full_path, image)  # org image path
        dst_path = os.path.join(val_label_folder_path, image)  # ./grass_project/grass_dataset/val/folder/image.png
        shutil.copyfile(src_path, dst_path)

print("OK")
