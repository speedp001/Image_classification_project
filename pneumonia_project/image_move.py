import os
import random
import shutil

org_data_folder_path = "./pneumonia_project/pneumonia_org_dataset"

dataset_folder_path = "./pneumonia_project/pneumonia_dataset"

#train or val folder path
train_folder_path = os.path.join(dataset_folder_path, "train")
val_folder_path = os.path.join(dataset_folder_path, "valid")
test_folder_path = os.path.join(dataset_folder_path, "test")


#train, val, test folder create
os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(val_folder_path, exist_ok=True)
os.makedirs(test_folder_path, exist_ok=True)



subdirectorys = os.listdir(org_data_folder_path)
#print(org_folders)
#['.DS_Store', 'Pneumonia_bacteria', 'Normal', 'Pneumonia_virus']

for subdirectory in subdirectorys :
    org_folder_full_path = os.path.join(org_data_folder_path, subdirectory)
    
    
    
    # Skip if it's not a directory or DS_Store(mac os only)
    if not os.path.isdir(org_folder_full_path) or subdirectory == ".DS_Store":
        continue
    
    images = os.listdir(org_folder_full_path)
    random.shuffle(images)


    #label folder path
    train_label_folder_path = os.path.join(train_folder_path, subdirectory)
    val_label_folder_path = os.path.join(val_folder_path, subdirectory)
    test_label_folder_path = os.path.join(test_folder_path, subdirectory)
    
    
    os.makedirs(train_label_folder_path, exist_ok=True)
    os.makedirs(val_label_folder_path, exist_ok=True)
    os.makedirs(test_label_folder_path, exist_ok=True)
    
    
    
    #train_split_index는 80% 지점, val_split_index는 90% 지점을 나타냅니다. 
    # Split the images into train, val, test sets
    train_split_index = int(len(images) * 0.8)
    val_split_index = int(len(images) * 0.9) #->train, val, test 3등분할 때 이용

    
    
    # Move images to the train folder
    for image in images[:train_split_index]:
        src_path = os.path.join(org_folder_full_path, image)
        dst_path = os.path.join(train_label_folder_path, image)
        shutil.copyfile(src_path, dst_path)

    #train, val, test 3등분할 때 이용
    # Move images to the validation folder
    for image in images[train_split_index:val_split_index]:
        src_path = os.path.join(org_folder_full_path, image)
        dst_path = os.path.join(val_label_folder_path, image)
        shutil.copyfile(src_path, dst_path)

    # Move images to the test folder
    for image in images[val_split_index:]:
        src_path = os.path.join(org_folder_full_path, image)
        dst_path = os.path.join(test_label_folder_path, image)
        shutil.copyfile(src_path, dst_path)
