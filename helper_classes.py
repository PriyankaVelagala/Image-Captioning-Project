from torch.nn.utils.rnn import pad_sequence  
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import os


CHECKPOINT_DIRECTORY = "/content/drive/MyDrive/Colab Notebooks/INM706_CW/model_checkpoints"


"""
Class to create batches
"""
class CollateCustom:
    
    def __init__(self, pad_value, batch_first=True):
        self.pad_val = pad_value
        self.batch_first = batch_first
    
    def __call__(self, batch):
        
        idxs = [item[0] for item in batch]
        
        X = [item[1].unsqueeze(0) for item in batch]
        X = torch.cat(X, dim = 0)
        
        y = [item[2] for item in batch]
        y = pad_sequence(y, batch_first=self.batch_first, padding_value=self.pad_val)
        
        return idxs, X, y
    
"""
TODO:
add losses in checkpoint?
move to helper class
""" 

"""
saves model weights + optimizer params + epoch 
"""
def save_checkpoint(checkpoint, fname):
    path = CHECKPOINT_DIRECTORY + "/" + fname 
    torch.save(checkpoint, path)
    print(f"Saved checkpoint {fname}!") 
    
"""
reload model from checkpoint
"""
def load_checkpoint(fname, model, optimizer):
    path =  CHECKPOINT_DIRECTORY + "/" + fname 
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Loaded checkpoint {fname}!")

"""
For inference
""" 
def save_model(model, fname):
    path = CHECKPOINT_DIRECTORY + "/" + fname 
    torch.save(model, path) 
    print(f"Saved model {fname}!")
    
def load_model(model, fname):
    path = CHECKPOINT_DIRECTORY + "/" + fname 
    model.load_state_dict(torch.load(path)) 
    print(f"Loaded from model {fname}!")
    
"""
calculate bleu score between expected and predicted
"""
def get_bleu_score(reference, predicted, weights=(0.25,0.25,0.25,0.25)):
    cc = SmoothingFunction()
    return  sentence_bleu(reference, predicted, smoothing_function=cc.method1,
        weights=weights)



    
    
    
    
    
    

        

