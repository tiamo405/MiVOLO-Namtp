import glob
import os
import pickle
import json 
import cv2

import numpy as np
import torch.utils.data as data
from PIL import Image

# from tools.function import get_pkl_rootpath


class PedesAttr(data.Dataset):

    def __init__(self, cfg, split, dir_image, dir_json):

        # --------- ADDENDUM  --------- # 
        # Ghi lại những item chưa có hoặc chưa đủ nhãn 
        
        # --------- INITIAL ASSUMPTIONS ---------- # 
        # __getitem()__ lấy index từ self. 

        # --------- DEBUGGING PURPOSES  --------- # 
        # if cfg.DATASET.TYPE == 'pedes':
        # train_set = PedesAttr(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=train_tsfm,
        #                       target_transform=cfg.DATASET.TARGETTRANSFORM)
        # valid_set = PedesAttr(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
        #                       target_transform=cfg.DATASET.TARGETTRANSFORM)
        # --------------------------------------- #


        # print(os.getcwd())


        if split == "train": 
            with open(dir_json, 'r') as f:
                data=json.load(f) 
            self.root_path = dir_image
        elif split == "val": 
            with open('/mnt/nvme0n1/locpv/cxview-person-attributes/data/THISOMALL/val_thiso/annotations/default.json', 'r') as f:
                data=json.load(f) 
            self.root_path = '/mnt/nvme0n1/locpv/cxview-person-attributes/data/THISOMALL/val_thiso/images/default/'
        else: 
            print('INVALID CONFIG, check ./data/PA100k for further examination') 
            return None 
        
        # img_id = dataset_info.image_name
        # '000000' + '.png' 
        img_id = [item['id'] +'.jpg' for item in data['items']]
        # print('expected length to be 37: ', len(img_id))
        # print(img_id) 

        # self.attr_id là những attribute (male female ...)
        # self.attr_num = len(self.attr_id) 
        # attribute trong json
        # nhóm độ tuổi cho Thisomall: 0-15, 15-30, 30-45. 45-60, 60+' 
        # gender là meo và phi meo
        self.attr_id = ['male', 'female', '1-13', '13-25', '25-39', '40-60', '60+']
        attr_label = [[0] * len(self.attr_id) for _ in range(len(img_id))]

        self.attr_num = len(self.attr_id)

        # cfg.DATASET.LABEL mặc định là 'all' 
        # cfg này chọn lọc các attribute cần thiết và theo một thứ tự nhất định 
        # Theo đó ta sẽ làm như phần if stmt ở dưới đầu tiên 
        self.eval_attr_num = self.attr_num 

        # Cần gắn nhẵn 
        incomplete = set()
        # vd: [[0, 1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0], ...]
        # O(N^2)

        for i in range(len(img_id)): 
            curr_item = data['items'][i]['annotations']
            #print(len(curr_item))
            for j in range(len(curr_item)):
                if 'gender' in curr_item[j]['attributes']: 
                    gender = curr_item[j]['attributes']['gender']
                if 'age' in curr_item[j]['attributes']: 
                    age = curr_item[j]['attributes']['age']
            #print(gender, age)
            # tìm gender age của mỗi item và cập nhật index tương tự trong attr_label
            attr_label[i][self.attr_id.index(gender)] = 1
            attr_label[i][self.attr_id.index(age)] = 1
            # 1 label mỗi attribute 
            if sum(attr_label[i][:2]) != 1: 
                incomplete.add(i)
            if sum(attr_label[i][2:]) != 1: 
                incomplete.add(i)

        self.img_id = img_id 
       
       # nếu đã tách train với val ra thì img_idx của cả train_set và valid_set bắt đầu từ 0. 
        self.img_idx = [i for i in range(len(img_id))]
        self.label = np.array(attr_label) 
        self.img_num = len(self.img_idx)
        
        # print(incomplete)
        # print(self.attr_id)
        # print(self.attr_num) 
        

    def __getitem__(self, index):
        
        # print(index)
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]

        imgpath = os.path.join(self.root_path, imgname)
        
        img = cv2.imread(imgpath)

        gt_label = gt_label.astype(np.float32)

        return imgpath, gt_label  # noisy_weight

    def __len__(self):
        return len(self.img_id)

def getitem(data_thiso, index) :
    attr_id = ['male', 'female', '1-13', '13-25', '25-39', '40-60', '60+']
    array = data_thiso.__getitem__(index)[1]
    imgpath = data_thiso.__getitem__(index)[0]
    selected_attributes = [attr for attr, value in zip(attr_id, array) if value == 1]
    gender, age = selected_attributes[0], selected_attributes[1]
    return imgpath, gender, age
if __name__ == "__main__" :
    dir_json = '/mnt/nvme0n1/locpv/cxview-person-attributes/data/THISOMALL/train_thiso/annotations/default.json'
    dir_image = '/mnt/nvme0n1/locpv/cxview-person-attributes/data/THISOMALL/train_thiso/images/default/'
    data_thiso = PedesAttr(cfg= None, split= "train", dir_json= dir_json, dir_image= dir_image)
    print(getitem(data_thiso, 3))
    print(data_thiso.__len__())
    
    
    