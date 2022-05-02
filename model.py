import torch
import torch.nn as nn
import statistics
import torchvision.models as models


logger = True

"""
Feature Extractor 
Image --> Embedding 
"""

class Encoder(nn.Module):
    
    def __init__(self, embed_size, extract_features):
        super().__init__()
        
        #initialize model  
        #self.model = models.resnet50(pretrained = True, progress = False)
        self.model = models.inception_v3(pretrained=True, aux_logits=False)
        # disable requires_grad for all layers
        self.set_parameter_requires_grad(extract_features)
        #re-intialize last layer 
        self.model.fc = nn.Linear(self.model.fc.in_features, embed_size) 
        #self.model.fc = nn.Linear(self.inception.fc.in_features, embed_size)

        
        #add activation and dropout 
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(0.5) 
        
        if logger:
            print("Initialized Encoder!")
        
        
    def forward(self, X): 
        X = self.model(X) 
        X = self.dropout(self.relu(X))
        
        return X
    
    """
    Initializes requires_grad based on extract_feature 
    - True - disables grad 
    - False - enables grad 
    """
    def set_parameter_requires_grad(self, extract_features):
        if extract_features:
            for param in self.model.parameters():
                param.requires_grad = False
        
    
"""
Caption Generator 
Embedding --> Caption 
"""
    
class Decoder(nn.Module):
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        
        #resize vocab 
        self.embedding = nn.Embedding(vocab_size, embed_size)
        #hidden_size =# of features in hidden size 
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size) 
        self.dropout = nn.Dropout(0.5)
        
        if logger:
            print("Initialized Decoder!")
        
        
    """
    Forward method: only used during training
    X -> embedded features from image 
    y -> image caption 
    """
    def forward(self, X, y):
        #captions --> embedding 
        embedded_captions  = self.dropout(self.embedding(y)) 
        
        #join features + embedding from captions 
        # num of features extracted from encoder = # size of embedding from caption
        embedding = torch.cat((X.unsqueeze(0), embedded_captions), dim = 0) 
        hiddens, states = self.lstm(embedding) 
        
        return self.linear(hiddens)
    
    
    
"""
Join Encoder and Decoder
"""

class EncodertoDecoder(nn.Module):
    
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size, extract_features):
        super(EncodertoDecoder, self).__init__()
        
        #super.__init__()
        self.encoder = Encoder(embed_size, extract_features) 
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers)
        
    """
    Forward methods
    X --> images 
    y --> image captions
    """ 
    def forward(self, X, y):
        image_features = self.encoder(X) 
        out = self.decoder(image_features, y) 
        return out 
    
    
    """
    Used during test time to generate captions
    """
    def caption_image(self, image, vocab, max_length = 10): 
        caption = []
        
        with torch.no_grad():
            #get image features 
            x = self.encoder(image).unsqueeze(0)
            states = None 
            
            #generate captions of (max) specified length
            for i in range(max_length):
                
                #pass previous state + word/image features to predict next words 
                hiddens, states = self.decoder.lstm(x, states) 
                pred_word = self.decoder.linear(hiddens.squeeze(0)).argmax(1)
                    
                caption.append(pred_word.item()) 
                
                #if end of sentence stop predicting 
                if vocab.idx_to_str[pred_word.item()] == "<EOS>":
                    break
                    
                #if not, use predicted output as input for next prediction 
                x = self.decoder.embedding(pred_word).unsqueeze(0)
          
        #convert predicted indices back to words 
        return [vocab.idx_to_str[idx] for idx in caption]
        
        
        
                
                
        
    