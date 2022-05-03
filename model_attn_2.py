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

Takes an image and encodes it into  a fixed length vector
"""

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Use Inception V3, a Convnet trained on Imagenet
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
        # resize to (batch, size*size, featuremaps) 
        batch, feature_maps, size_1, size_2 = X.size() 
        X = X.permute(0,2,3,1) 
        features = X.view(batch, size_1*size_2, feature_maps) 
        
        return features
    

"""
Caption Generator 
Embedding --> Caption 

Decodes the fixed length vector and ouputs a predicted sequence
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
        
        self.short_memory = nn.Linear(self.num_features, hidden_size)
        self.long_memory = nn.Linear(self.num_features, hidden_size)

 
        self.lstm = nn.LSTMCell(embed_size + self.num_features, hidden_size, num_layers) 
        
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
        short_m, long_m = self.init_hidden(X) 
        
        seq_len = y.size(1)-1
        feature_size = X.size(1) 
        batch_size = X.size(0)
        
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(device)
        weights =  torch.zeros(batch_size, seq_len, feature_size).to(device)
        
        for s in range(seq_len):
            # generate attention and weights
            alpha, context = self.attention(X, short_m)

            lstm_input = torch.cat((embedded_captions[:, s], context), dim=1)
            short_m, long_m= self.lstm(lstm_input, (short_m, long_m))
                    
            output = self.linear(self.dropout(short_m))
            
            outputs[:,s] = output
            weights[:,s] = alpha  
        
        
        return outputs, weights
        
    
    def init_hidden(self, features):
        mean_annotations = torch.mean(features, dim = 1)
        short_m = self.short_memory(mean_annotations)
        long_m = self.long_memory(mean_annotations)
        
        return short_m, long_m

"""
Join Encoder and Decoder

Encapsulates the Encoder/Decoder class and converts the decoder output to human 
readable captions
"""

class EncodertoDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, vocab_size, attn_dim):
        super(EncodertoDecoder, self).__init__()
        
        self.encoder = Encoder() 
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, attn_dim)
        
        
        
    """
    Forward methods
    X --> images 
    y --> image captions
    """ 
    def forward(self, X, y):
        features = self.encoder(X) 
        out, _ = self.decoder(features, y) 
        return out
    
    
    def generate_caption(self,features,vocab=None, max_length=20):
        batch_size = features.size(0)
        hidden, states = self.decoder.init_hidden(features)
        alphas = []
        
        # Starting input. Initialize word with index of "<SOS>"
        word = torch.tensor(1).unsqueeze(0).view(1,-1).to(device)
        embeds = self.decoder.embedding(word)

        captions = []
        
        for i in range(max_length):
            # Pass features (annotations) from encoder along with hidden state 
            # from decoder into the attention to generate attention and weights.
            # the attention scores are softmaxed and will be refered to as alpha
            alpha, attention_weights = self.decoder.attention(features, hidden)
            
            # Move alpha score to cpu and convert to np array
            alphas.append(alpha.cpu().detach().numpy())
            
            # Concat embeddings and weights and format for lstm
            lstm_input = torch.cat((embeds[:, 0], attention_weights), dim=1)
            
            hidden, states = self.decoder.lstm(lstm_input, (hidden, states))
            
            output = self.decoder.linear(self.decoder.dropout(hidden))
            output = output.view(batch_size,-1)
            
            # select the word with most val
            predicted_word_idx = output.argmax(dim=1)
            
            #save the generated word
            captions.append(predicted_word_idx.item())
            
            # end if 2 (<EOS>) detected. 
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
        
        self.fc1 = nn.Linear(hidden_dim, attn_dim)
        self.fc2 = nn.Linear(num_features,attn_dim)
        self.attention_score = nn.Linear(attn_dim,1)
        
        
    def forward(self, encoder_output, decoder_hidden_state):   
        x_decoder = self.fc1(decoder_hidden_state) 
        x_encoder = self.fc2(encoder_output)

        combined_states = torch.tanh(x_encoder + x_decoder.unsqueeze(1)) 
        
        attention_scores = self.attention_score(combined_states) 
        attention_scores = attention_scores.squeeze(2)

        soft_maxed_attention = F.softmax(attention_scores,dim=1)

        attention_weights = encoder_output * soft_maxed_attention.unsqueeze(2)
        attention_weights = attention_weights.sum(dim=1) 

        return soft_maxed_attention, attention_weights

    
    
    
    
    
    

