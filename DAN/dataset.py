import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset

'''
    Specific dataset classes for person re-identification dataset. 
'''

class SYSUGroupDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']
        
        self.gids = np.load(os.path.join(root, 'npy_21',  mode+'_gids.npy'), allow_pickle=True).tolist() 

        self.root = root
        self.t_paths = np.load(os.path.join(root, 'npy_21', mode+'_path.npy'), allow_pickle=True).tolist()
        self.cam_ids = np.load(os.path.join(root, 'npy_21',  mode+'_cam_ids.npy'), allow_pickle=True).tolist()
        self.transform = transform
        self.num_ids = len(self.gids)
        self.num_person = np.load(os.path.join(root, 'npy_21',  mode+'_num_person.npy'), allow_pickle=True).tolist()
        self.ids = np.load(os.path.join(root, 'npy_21',  mode+'_ids.npy'), allow_pickle=True).tolist()


    


    def __len__(self):
        return len(self.t_paths)


    def __getitem__(self, item):
        t_path = self.t_paths[item]
        nums_p = self.num_person[item]

        t_img_paths = sorted(os.listdir(t_path))

        t_imgs = []
        len_bbox = []

        t_len = len(t_img_paths)

        if t_len<=11:
            nums_f = t_len
            for i in range(t_len):
                img = np.load(os.path.join(t_path, t_img_paths[i]), allow_pickle=True)
                len_bbox.append(len(img))
                imgs_p = []

                for j in range(len(img)):
                    if self.transform is not None:
                        imgs_p.append(self.transform(img[j]).unsqueeze(0))

                while len(imgs_p)<8:
                    imgs_p.append(imgs_p[0])

                t_imgs.append(torch.cat(imgs_p, dim=0).unsqueeze(0))

        elif t_len<41:
            nums_f = 11
            indices = random.sample(range(t_len), nums_f)
            indices = sorted(indices)
            for i in indices:
                img = np.load(os.path.join(t_path, t_img_paths[i]), allow_pickle=True)
                len_bbox.append(len(img))
                imgs_p = []

                for j in range(len(img)):
                    if self.transform is not None:
                        imgs_p.append(self.transform(img[j]).unsqueeze(0))

                while len(imgs_p)<8:
                    imgs_p.append(imgs_p[0])

                t_imgs.append(torch.cat(imgs_p, dim=0).unsqueeze(0))

        else:
            nums_f = 11
            end = t_len-41
            
            start = random.randint(0, end)  # include 0 & end
            for i in range(start,start+41,4):

                img = np.load(os.path.join(t_path, t_img_paths[i]), allow_pickle=True)
                len_bbox.append(len(img))
                imgs_p = []

                for j in range(len(img)):
                    if self.transform is not None:
                        imgs_p.append(self.transform(img[j]).unsqueeze(0))

                while len(imgs_p)<8:
                    imgs_p.append(imgs_p[0])

                t_imgs.append(torch.cat(imgs_p, dim=0).unsqueeze(0))
                
        
        while len(t_imgs) < 11:
            t_imgs.append(t_imgs[-1])
            len_bbox.append(len_bbox[-1])
    
        
        t_imgs = torch.cat(t_imgs, dim=0)

        nums_p = torch.tensor(nums_p, dtype=torch.long)      
        nums_f = torch.tensor(nums_f, dtype=torch.long)
        len_bbox = torch.tensor(len_bbox, dtype=torch.long)
        gid = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
    

        return t_imgs, gid, cam, nums_p, nums_f,len_bbox


