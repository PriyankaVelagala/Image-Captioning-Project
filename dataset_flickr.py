import os,sys
import re
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils import data as data
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
import json
import pandas as pd 
import transformers



logger = True

"""
TODO:
1. might need to update how captions are stored (self.img_captions, returns dataframe for now)
"""



class Flickr30kData(data.Dataset):
    
    def __init__(self, **kwargs):
        self.stage = kwargs['stage']
        self.ds_path = kwargs['ds_path']
        self.captions_f = kwargs['ds_path'] + kwargs['captions_dir'] +  '/' + kwargs['captions_fname']
        self.imgs_dir = kwargs['ds_path'] + kwargs['images_dir'] + '/' + self.stage 
        self.img_captions_df = self.load_captions()
        self.img_files_dict = self.map_img_to_idx() 
        
        if logger:
            print("stage: ", self.stage)
            print("ds_path: ", self.ds_path)
            print("captions_f: ", self.captions_f)
            print("imgs_dir: ", self.imgs_dir)
            
            
    """
    Returns contents of file 
    for specified stage as dataframe 
    
    """
    def load_captions(self):
        #load into a dataframe + transformation 
        col_names = ["img_name", "comment_number", "comment"]
        df = pd.read_csv(self.captions_f, delimiter = "|", names=col_names, header=None, skiprows = 1)
        df = df.pivot(index='img_name', columns='comment_number', values='comment').reset_index()
         
        #filter to stage files 
        files = os.listdir(self.imgs_dir)
        df = df[df.img_name.isin(files)]
        
        #print(df.head())        
        return df 
    
    
    """
    Given an index
    returns all image captions as a list of text captions
    """
    def get_image_caption(self, idx):
        fname = str(self.img_files_dict[idx])
        df = self.img_captions_df
        captions = list(df[df.img_name == fname].values[0][1:]) 
        return captions
    
     """
    Given an index
    returns all image captions as a list of tokens
    """
    def get_tokenized_captions(self, idx):
        captions = self.get_image_caption(idx)
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
        tokenized_captions = tokenizer(captions,
                                   return_token_type_ids = False, 
                                   return_attention_mask = False, 
                                   max_length = 40, #max caption length = 83, mean = 40.1  - hperparameter?
                                   padding = "max_length",
                                   return_tensors = "pt")
        return tokenized_captions
        

        
    """
    Maps file names to indexes 
    """
    def map_img_to_idx(self):
        #get all file names for stage
        fnames = os.listdir(self.imgs_dir)
        fname_dict = {} 
        
        for idx, fname in enumerate(fnames):
            fname_dict[idx] = fname 
        
        return fname_dict  
    
    
    """
    Load image
    """
    def load_img(self, idx):
        #get file name
        fname = self.img_files_dict[idx]
        path = self.imgs_dir + '/' + fname 
        img = np.array(PILImage.open(path))
        tfrm = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = tfrm(img)
        return img 
    
    """
    Plots image
    """
    def plot_img(self, idx):
        #get file name
        fname = self.img_files_dict[idx]
        path = self.imgs_dir + '/' + fname 
        return PILImage.open(path)
    
    
    """
    magic methods 
    """
    def __len__(self):
        return len(self.img_files_dict)
    
    def __getitem__(self, idx):
        X = self.load_img(idx)
        y = self.get_tokenized_captions(idx)
        return X,y 
        