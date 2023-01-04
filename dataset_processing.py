import os
import cv2

class dataset_processing:
    
    def __init__(self,opt):
        
        self.tar_path = opt.save
        
        self.train_num = int(opt.num*opt.split)
        self.cur_num = 1
        print(f"total num :{opt.num} , train num :{self.train_num} , split percent : {opt.split}")
        
        if not os.path.isdir(self.tar_path):
            os.mkdir(self.tar_path)
            os.mkdir(os.path.join(self.tar_path,"./images"))
            os.mkdir(os.path.join(self.tar_path,"./labels"))
            os.mkdir(os.path.join(self.tar_path,"./images/train"))
            os.mkdir(os.path.join(self.tar_path,"./images/val"))
            os.mkdir(os.path.join(self.tar_path,"./labels/train"))
            os.mkdir(os.path.join(self.tar_path,"./labels/val"))
            
    def save_to_dataset(self,img,area):
        
        print(f"processing {self.cur_num}")
        img = cv2.resize(img,(160,160))
        
        where = "train"
        if self.cur_num > self.train_num:
            where = "val"
            
        dataset_image_path = os.path.join(self.tar_path,"./images",where,str(self.cur_num)+".png")
        dataset_label_path = os.path.join(self.tar_path,"./labels",where,str(self.cur_num)+".txt")
        
        cv2.imwrite(dataset_image_path,img)
        x_min , y_min ,x_max ,y_max = area[0] ,area[1] , area[2] ,area[3]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0), 3, cv2.LINE_AA)

        h , w  ,d =img.shape
        
        x_center = float(x_min+x_max)/2.0 * float(1.0/w)
        y_center = float(y_min+y_max)/2.0 *float(1.0/h)
        yolo_w = (x_max - x_min)* float(1.0/w)
        yolo_h = (y_max - y_min)* float(1.0/h)
        
        
        
        
        with open(dataset_label_path,"w") as f:
            f.write(f"0 {x_center} {y_center} {yolo_w} {yolo_h}\n")
        
        self.cur_num+=1
