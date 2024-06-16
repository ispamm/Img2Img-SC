import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from matplotlib import pyplot as plt

from io import open
import unicodedata
import re




class Flickr8kDataset(Dataset):
    def __init__(self,images_dir_path, capt_file_path):
        super().__init__()
        #read data
        data=open(capt_file_path).read().strip().split('\n')
        data=data[1:]

        img_filenames_list=[]
        captions_list=[]

        for s in data:
            templist=s.lower().split(",")
            img_path=templist[0]
            caption=",".join(s for s in templist[1:])
            caption=self.normalizeString(caption)
            img_filenames_list.append(img_path)
            captions_list.append(caption)

        self.images_dir_path=images_dir_path
        self.img_filenames_list=img_filenames_list
        self.captions_list=captions_list
        self.length=len(self.captions_list)
        self.transform=Compose([Resize((224,224), antialias=True), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    
    def __len__(self):
        return self.length
    
    #unicode 2 ascii, remove non-letter characters, trim
    def normalizeString(self,s): 
        sres=""
        for ch in unicodedata.normalize('NFD', s): 
            #Return the normal form form ('NFD') for the Unicode string s.
            if unicodedata.category(ch) != 'Mn':
                # The function in the first part returns the general 
                # category assigned to the character ch as string. 
                # "Mn' refers to Mark, Nonspacing
                sres+=ch
        #sres = re.sub(r"([.!?])", r" \1", sres) 
        # inserts a space before any occurrence of ".", "!", or "?" in the string sres. 
        sres = re.sub(r"[^a-zA-Z!?,]+", r" ", sres) 
        # this line of code replaces any sequence of characters in sres 
        # that are not letters (a-z or A-Z) or the punctuation marks 
        # "!", "," or "?" with a single space character.
        return sres.strip()


    
    def __getitem__(self,idx):
        imgfname,caption=self.img_filenames_list[idx],self.captions_list[idx]
        
        imgfname=self.images_dir_path+imgfname

        return imgfname, caption 
    
import os
class Only_images_Flickr8kDataset(Dataset):
    def __init__(self,images_dir_path):
        super().__init__()
        #read data
        img_filenames_list=[]
        
        for root, dirs, files in os.walk(images_dir_path):
            for file in files:
                if file.endswith('.jpg'):
                    img_filenames_list.append(file)

        
        #print(img_filenames_list)
        self.images_dir_path=images_dir_path
        self.img_filenames_list=img_filenames_list
        self.length = len(img_filenames_list)

    
    def __len__(self):
        return self.length
    

    
    def __getitem__(self,idx):

        imgfname = self.img_filenames_list[idx]
        imgfname = self.images_dir_path+imgfname

        return imgfname



if __name__ == "__main__":
    

    capt_file_path=   "path/to/captions.txt"          #"G:/Giordano/Flickr8kDataset/captions.txt"
    images_dir_path=  "path/to/Images"                #"G:/Giordano/Flickr8kDataset/Images/"
    
    dataset=Flickr8kDataset(images_dir_path, capt_file_path)


    batch_size=1
    train_dataloader=DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True)


    for i in train_dataloader:
        print(i[0][0])
        print(i[1])
        break