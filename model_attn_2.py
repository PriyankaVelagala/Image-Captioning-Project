import torch
import torch.nn as nn
import statistics
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


logger = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Feature Extractor 
Image --> Embedding 
"""

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        
        inception_v3 = models.inception_v3(pretrained=True, aux_logits=False)
        for param in inception_v3.parameters():
            param.requires_grad_(False)
        
        #remove last 3 layers to egt feature maps 
        modules = list(inception_v3.children())[:-3]
        self.model = nn.Sequential(*modules)
        
        
        if logger:
            print("Initialized Encoder!")
        
        
    def forward(self, X): 
        """
        X = self.model(X) 
        X = self.dropout(self.relu(X))
        """
        X = self.model(X)
        #resize to (batch, size*size, featuremaps) 
        batch, feature_maps, size_1, size_2 = X.size() 
        X = X.permute(0,2,3,1) 
        X = X.view(batch, size_1*size_2, feature_maps) 
        
        return X
    

"""
Caption Generator 
Embedding --> Caption 
"""

class Decoder(nn.Module):
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, attn_dim):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_features = 2048 #feature maps from cnn 
        self.attn_dim = attn_dim
        
        #resize vocab 
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = BahdanauAttention(self.num_features, hidden_size, attn_dim)
        
        self.init_h = nn.Linear(self.num_features, hidden_size)
        self.init_c = nn.Linear(self.num_features, hidden_size)

        
        #hidden_size =# of features in hidden size 
        self.lstm = nn.LSTMCell(embed_size + self.num_features, hidden_size, num_layers) #check first arg 
        #self.f_beta = nn.Linear(hidden_size, vocab_size) 
        
        
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
        embedded_captions  = self.embedding(y) 
        
        hiddens, states = self.init_hidden(X) 
        
        seq_len = y.size(1)-1
        feature_size = X.size(1) 
        batch_size = X.size(0)
        
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(device)
        weights =  torch.zeros(batch_size, seq_len, feature_size).to(device)
        
        for s in range(seq_len):
            alpha,context = self.attention(X, hiddens)
            lstm_input = torch.cat((embedded_captions[:, s], context), dim=1)
            hiddens, states= self.lstm(lstm_input, (hiddens, states))
                    
            output = self.linear(self.dropout(hiddens))
            
            outputs[:,s] = output
            weights[:,s] = alpha  
        
        
        return outputs, weights
        
    
    def init_hidden(self, features):
        mean_annotations = torch.mean(features, dim = 1)
        h0 = self.init_h(mean_annotations)
        c0 = self.init_c(mean_annotations)
        return h0, c0

"""
Join Encoder and Decoder
"""

class EncodertoDecoder(nn.Module):
    
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size, attn_dim):
        super(EncodertoDecoder, self).__init__()
        
        #super.__init__()
        self.encoder = Encoder() 
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, attn_dim)
        
        
        
    """
    Forward methods
    X --> images 
    y --> image captions
    """ 
    def forward(self, X, y):
        features = self.encoder(X) 
        out, weights = self.decoder(features, y) 
        return out#, weights 
    
    
    
    def generate_caption(self,features,vocab=None, max_len=20):

        batch_size = features.size(0)
        hidden, states = self.decoder.init_hidden(features)  # (batch_size, decoder_dim)
        alphas = []
        
        #starting input
        word = torch.tensor(1).unsqueeze(0).view(1,-1).to(device)
        embeds = self.decoder.embedding(word)

        captions = []
        
        for i in range(max_len):
            alpha,context = self.decoder.attention(features, hidden)
            
            #store the apla score
            alphas.append(alpha.cpu().detach().numpy())
            
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            hidden, states = self.decoder.lstm(lstm_input, (hidden, states))
            output = self.decoder.linear(self.decoder.dropout(hidden))
            output = output.view(batch_size,-1)
            
            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)
            
            #save the generated word
            captions.append(predicted_word_idx.item())
            
            #end if <EOS detected>
            if predicted_word_idx.item() == 2:
                break
            
            #send generated word as the next caption
            embeds = self.decoder.embedding(predicted_word_idx.unsqueeze(0))
        
        #covert the vocab idx to words and return sentence
        return [vocab.idx_to_str[idx] for idx in captions]
        

"""
Attention Module
"""
    
class BahdanauAttention(nn.Module):
    def __init__(self, num_features,hidden_dim, attn_dim):
        super(BahdanauAttention, self).__init__()
        
        self.W = nn.Linear(hidden_dim,attn_dim)
        self.U = nn.Linear(num_features,attn_dim)
        self.A = nn.Linear(attn_dim,1)
        
        
    def forward(self, features, hidden_state):
        u_hs = self.U(features)     #(batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state) #(batch_size,attention_dim)
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) #(batch_size,num_layers,attemtion_dim)
        
        attention_scores = self.A(combined_states)         #(batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)     #(batch_size,num_layers)

        alpha = F.softmax(attention_scores,dim=1)          #(batch_size,num_layers)

        attention_weights = features * alpha.unsqueeze(2)  #(batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)   #(batch_size,num_layers)

        return alpha,attention_weights

    
    
    
    
    

