import nltk 
import os
import csv
from collections import Counter


class Vocabulary(object):
    def __init__(self, threshold, ann_path = '/home/Datasets/flickr30k/annotations/results_new.csv'):
        self.idx_to_str = {} 
        self.str_to_idx = {} 
        self.threshold = threshold 
        self.idx = 0 
        self.captions_file = ann_path 
        self.build_vocab() 
        
        
    def build_vocab(self):
        self.add_reserved_words()
        self.parse_captions()
        
        
    def add_word(self, word):
        if word not in self.str_to_idx:
            self.idx_to_str[self.idx] = word
            self.str_to_idx[word] = self.idx
            self.idx += 1
            
    def add_reserved_words(self):
        words = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        for word in words:
            self.add_word(word)
             
            
    def parse_captions(self):
        line_count = 0 
        counter = Counter()
        
        #tokenize captions
        with open(self.captions_file, 'r') as file:
            reader = csv.reader(file, delimiter = "|")
            for row in reader:
                if line_count == 0:
                    line_count += 1
                    continue
                else:
                    caption = row[2].lstrip().rstrip()
                    tokens = nltk.tokenize.word_tokenize(caption.lower())
                    counter.update(tokens)
                    line_count += 1
        #filter to word that exceed specified threshold 
        words = [w for w, c in counter.items() if c >= self.threshold]
        
        for word in words:
            self.add_word(word)
            
            
    def vectorize(self, caption):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        vec = []
        for token in tokens:
            if token in self.str_to_idx:
                vec.append(self.str_to_idx[token])
            else:
                vec.append(self.str_to_idx["<UNK>"])
        return vec 
    
    def __len__(self):
        return len(self.str_to_idx)
            
        
        
        

        
        
        
            
        
            
            