import os
import matplotlib.pyplot as plt

class DataVisualizer :
    def __init__(self, data_dir) :
        
        self.data_dir = data_dir
        self.all_data = {}
        self.train_data = {}
        self.val_data = {}
        self.test_data = {}
    
    def load_data(self) :
    
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'validation')
        test_dir = os.path.join(self.data_dir, 'test')
    
        for label in os.listdir(train_dir) :
            label_dir = os.path.join(train_dir, label)
            #print(label_dir)
            #./food_project/food_dataset/train/pizza
            
            #print(len(os.listdir(label_dir)))
            count = len(os.listdir(label_dir))
            #라벨 딕셔너리 생성 및 배정
            self.all_data[label] = count
            self.train_data[label] = count
            
        for label in os.listdir(val_dir) :
            label_dir = os.path.join(val_dir, label)
            #print(label_dir)
            #./food_project/food_dataset/train/pizza
            
            #print(len(os.listdir(label_dir)))
            count = len(os.listdir(label_dir))
            #라벨 딕셔너리 생성 및 배정
            self.val_data[label] = count
            
            #all_data에 정보가 있다면 추가 or 없다면 생성
            if label in self.all_data :
                self.all_data[label] += count
            else :
                self.all_data[label] = count
                
        
        for label in os.listdir(test_dir) :
            label_dir = os.path.join(test_dir, label)
            #print(label_dir)
            #./food_project/food_dataset/train/pizza
            
            #print(len(os.listdir(label_dir)))
            count = len(os.listdir(label_dir))
            #라벨 딕셔너리 생성 및 배정
            self.test_data[label] = count
            
            #all_data에 정보가 있다면 추가 or 없다면 생성
            if label in self.all_data :
                self.all_data[label] += count
            else :
                self.all_data[label] = count
        
        #전체 라벨 속 데이터 수 출력
        #print(self.all_data)
        
    def vissualize_data(self) :
        
        labels = list(self.all_data.keys())
        counts = list(self.all_data.values())
    
        # print(labels)
        # print(counts)
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts)
        plt.title("label data number")
        plt.xlabel("label")
        plt.ylabel("data number")
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.show()
    
if __name__ == "__main__" :
    
    test = DataVisualizer("./food_project/food_dataset")
    test.load_data()
    test.vissualize_data()