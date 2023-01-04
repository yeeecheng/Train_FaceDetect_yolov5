from retinaface import RetinaFace 
import os  
import cv2
from dataset_processing import *
import argparse

def process(opt):
    
    src_path =opt.train_dataset
    training_img_num = opt.num
   
    
    data_processing = dataset_processing(opt)
    cnt = 0
    model = RetinaFace
    for r, d ,f in os.walk(src_path):
        
        for file in f :
            if cnt  < opt.start:
                cnt+=1
                data_processing.cur_num=cnt
                continue
            
            img_path = os.path.join(r,file)
            faces = model.detect_faces(img_path)
            
            if len(faces) != 1:
                continue
            img = cv2.imread(img_path)
            
            area = faces["face_1"]["facial_area"]
            
            area[0]-=20
            if area[0]  < 0 : 
                area[0] = 0
            area[1] -= 30
            if area[1]  < 0 : 
                area[1] = 0
            area[2] += 10
            if area[2]  >= img.shape[1] : 
                area[2] = img.shape[1]-1
            area[3] -= 40
            if area[3]  >= img.shape[0] : 
                area[3] = img.shape[0]-1
            
            data_processing.save_to_dataset(img ,area)
                
            cnt+=1
            if cnt >= training_img_num+opt.start:
                break
            
def main(opt):
    process(opt)

def parse_opt(known =False):
    
    ROOT = os.getcwd()
    print(ROOT)
    parser =argparse.ArgumentParser()
    parser.add_argument("--root",type = str ,default=ROOT)
    parser.add_argument("--train_dataset",type = str,default=os.path.join(ROOT,'./dataset/img_align_celeba'))
    parser.add_argument("--save",type = str,default=os.path.join(ROOT,"./dataset/face_detect_dataset"),help="the path where you save processing img")            
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--num",type = int ,default=1500,help="the number which you want to train")
    parser.add_argument("--split",type = float ,default=0.9,help="the percentage which you want split the dataset")
    return parser.parse_known_args()[0] if known else parser.parse_args()

def run(**kwargs):
    
    opt =parse_opt(True)
    for k ,v in kwargs.items():
        setattr(opt,k,v)
    main(opt)
    return opt


if __name__ == "__main__":
    run()