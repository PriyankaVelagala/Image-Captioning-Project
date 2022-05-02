import os,sys
import re
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils import data as data
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
import transformers
import csv 
import vocab
from collections import defaultdict



logger = True

"""
TODO:
1. might need to update how captions are stored (self.img_captions, returns dataframe for now) - DONE 
"""



class Flickr7kData(data.Dataset):
    
    def __init__(self, **kwargs):
        self.stage = kwargs['stage']
        self.ds_path = kwargs['ds_path']
        self.captions_f = kwargs['ds_path'] + kwargs['captions_dir'] +  '/' + kwargs['captions_fname']
        self.imgs_dir = kwargs['ds_path'] + kwargs['images_dir'] + '/' + self.stage 
        #should this be initialized once and passed in? 
        self.vocabulary = vocab.Vocabulary(kwargs['freq_threshold'], self.captions_f )
        
        self.img_files_dict = self.map_img_to_idx() 
        self.captions_dict = self.load_captions()

        self.idx_to_caption = self.map_idx_to_caption()
        
        if logger:
            print("stage: ", self.stage)
            print("ds_path: ", self.ds_path)
            print("captions_f: ", self.captions_f)
            print("imgs_dir: ", self.imgs_dir)
            print("Initialized {} words in vocabulary".format(len(self.vocabulary)))
            
            
    """
    Returns contents of file 
    for specified stage as dataframe 
    """
    def load_captions(self):
        #all files in current stage 
        files = os.listdir(self.imgs_dir)
        
        captions_dict = defaultdict(list)
        
        line_count = 0 
        
        with open(self.captions_f, 'r') as file:
            reader = csv.reader(file, delimiter = "|")
            for row in reader:
                #ignore first line = column names 
                if line_count == 0:
                    line_count += 1
                    continue
                    
                if str(row[0]) in files:
                    img_id = row[0].split('.')[0]
                    #caption = "<SOS> " + row[2].lstrip().rstrip() + " <EOS>"
                    caption = row[2].lstrip().rstrip() 
                    captions_dict[img_id].append(caption)
                
                line_count += 1 
                
        return captions_dict 
    
    """
    Returns idx to caption
    """
    def map_idx_to_caption(self):
        idx_to_cap_dict = {} 
        for i in range(len(self.captions_dict)):
            img_id = str(self.img_files_dict[5*i].split('.')[0])
            for j in range(5):
                idx_to_cap_dict[5*i +j ] = self.captions_dict[img_id][j]

        return idx_to_cap_dict
    
                        
    
    """
    Given an index
    returns the first image caption
    """
    def get_image_caption(self, idx):
        return self.idx_to_caption[idx]


        
    """
    Given an index
    returns all captions
    """
    def get_all_captions(self, idx):
        fname = str(self.img_files_dict[idx])
        img = fname.split('.')[0]
        return self.captions_dict[img]
    
    
    
    """
    Given an index
    returns the first caption as a list of tokens
    """
    def get_vectorized_caption(self, idx):
        caption = self.get_image_caption(idx)
        vectorized_cap = []
        vectorized_cap.append(self.vocabulary.str_to_idx["<SOS>"])
        vectorized_cap += self.vocabulary.vectorize(caption)
        vectorized_cap.append(self.vocabulary.str_to_idx["<EOS>"])
        return vectorized_cap      

        
    """
    Maps file names to indexes 
    """
    def map_img_to_idx(self):
        #get all file names for stage
        fnames = os.listdir(self.imgs_dir)
        fname_dict = {} 
        
        for idx, fname in enumerate(fnames):
            for i in range(0,5):
                fname_dict[5*idx + i] = fname 
            
        
        return fname_dict  
    
    
    """
    Load image
    """
    def load_img(self, idx):
        #get file name
        fname = self.img_files_dict[idx]
        path = self.imgs_dir + '/' + fname 
        img = PILImage.open(path).convert('RGB')
        
        
        if self.stage == "train":
            
            transform_train = transforms.Compose([ 
                transforms.Resize(256),                          # smaller edge of image resized to 256
                transforms.RandomCrop(224),                      # get 224x224 crop from random location
                transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
                transforms.ToTensor(),                           # convert the PIL Image to a tensor
                transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                     (0.229, 0.224, 0.225))]
            )
            """
            transform_train = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor()
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]) 
            """
            
            img = transform_train(img)
                
        else: 
            transform_test = transforms.Compose([ 
                            transforms.Resize((224,224)),                   # smaller edge of image resized to 256
                            transforms.ToTensor(),                           # convert the PIL Image to a tensor
                            transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                                 (0.229, 0.224, 0.225))])


            img = transform_test(img)
            
        return img 
    
    """
    Plots image
    """
    def plot_img(self, idx):
        #get file name
        fname = self.img_files_dict[idx]
        path = self.imgs_dir + '/' + fname 
        return PILImage.open(path)
    
    

    def __len__(self):
        return len(self.img_files_dict)
    
    def __getitem__(self, idx):
        X = self.load_img(idx)
        y = torch.tensor(self.get_vectorized_caption(idx))
        return idx, X,y 
        