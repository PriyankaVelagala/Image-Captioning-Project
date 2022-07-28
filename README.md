# INM706 : Image Captioning using Soft Attention Mechanism 
Anthony Hoang, Priyanka Velagala

For full report, see: [INM706_Image Captioning_Report.pdf](https://github.com/PriyankaVelagala/Image-Captioning-Project/blob/main/INM706_%20Image%20Captioning%20-%20Report.pdf)

# Overview 
Image captioning is a complex task that lies in the intersection of the field of computer vision and natural
language processing. It involves generating syntactically and semantically meaningful text descriptions
based on the content of an image. This task has impactful applications from the ability to automate
image indexing to adding enhanced capabilities to virtual assistants used by visually impaired persons.

# Dataset 
We chose Flickr30k, a benchmark dataset to train and test our model. In using a benchmark dataset, this
facilitates easy comparison of our model performance with industry standards. The dataset itself
contains 31,000 images, each with 5 descriptive captions. Due to time and compute restrictions, we’ve
chosen to work with a subset of 10,000 images using all 5 reference captions for training. As there are
no standard data splits for this dataset, a 7:2:1 train-validation-test split was used resulting in a total of
35,000 training, 10,000 validation and 5,000 testing instances.

# Model Architecture 
## Encoder 
The purpose of the encoder is to extract meaningful features from the input image. For this, we use a
convolutional neural network (CNN) pre-trained on an image classification task. By using a pre-trained model, the model forms a good representation of objects in the world and can extract defining features with a high degree of accuracy. This allows us to save on the computationally intensive task of training the CNN from scratch and apply transfer learning where the features learned from the ImageNet dataset are used to extract key features from the images in our chosen dataset.

## Decoder 
For our decoder, as specified in the original paper, we use a recurrent neural network (RNN) for
sequence modeling. RNNs perform particularly well in tasks such as image captioning due to their ability
to correlate information extracted from prior inputs to influence both inputs and outputs at a given time
step[7]. These networks have a “memory” component which allows them to remember and account for
the previous stream and position of inputs giving them strong predictive capabilities. 
![image](https://user-images.githubusercontent.com/26152595/181476459-fc6aac7b-c21d-4ca1-80f8-0c438b705b34.png)

We motivate our choice of using LSTM RNNs as a result of this problem, as well as traditional RNNs’
inability to handle long term dependencies[7]. The LSTM introduces a memory cell that is used to store
information for longer periods of time. There are three key components that regulate what is stored in
this cell (see Figure 3.3):
1. Forget gate (ft) - decides which information to omit from the previous time step
2. Input gate (i
t) - decides which information passes through based on relevance in the current
timestep
3. Output gate (ot) - decides how much of the cell state impacts the output
These three gates let the memory cell decide when to remember or ignore inputs in the hidden states.

## Attention 
An enhancement that was made to our baseline model to improve the quality of captions generated was
through the addition of an attention mechanism. Soft Attention considers all regions of interest and context as its input, and outputs the regions most relevant at that time step. The weights of the attention module are learnable parameters which are adjusted based on features maps from the encoder and the context thus far. To ensure the regions
output by the attention module are only those most relevant to the context, summing the two allows the
context most similar to the region to be amplified. Passing these intermediary values through a softmax
layer produces a probability distribution of the attention weights. The dot product of these weights and
the regions highlight the context which is then passed as input to the LSTM.

![image](https://user-images.githubusercontent.com/26152595/181476712-efdb6bc4-7287-4fa5-92cc-db233f373c73.png)

# Models 

## Baseline Model 
The setup for our baseline model is as seen below. This closely follows the NIC model
specified in the paper by Vinyals et al (2014).
![image](https://user-images.githubusercontent.com/26152595/181477474-290ee57a-35c5-4c56-a6d2-4c4acc31da65.png)

## Baseline+ Model (w/ Soft attention) 
The setup for our baseline+ model follows closely with the baseline model however two key differences from the baseline model exist: 
1. The addition of an attention module in the decoder
2. Omission of the final activation layer of the encoder such that the feature maps from the CNN
can be passed in directly to the decoder
![image](https://user-images.githubusercontent.com/26152595/181477656-e446b539-0ddd-4c0b-8c4d-6f746d054902.png)


# Results & Findings 
In our implementation, although we used best practices and mimicked the architecture set out by the
NIC model in Vinayals et al (2014) and Xu et al (2015), unfortunately our model boded quite poorly in
comparison. The BLEU scores attained indicate that the model has not sufficiently learned to form a
meaningful text description based on contents of an image. Between the two implementations, the
Baseline+ model, which incorporates soft attention on the Baseline model, expectedly performed better
given that the model has a mechanism to more strategically focus on certain regions of the image when
generating the caption.

One reason as to why our model was unable to perform as well as the architectures we followed in our
implementation could be due to the use of only a subset of the full data which means the model was
exposed to fewer examples during training. We also deviated from the implementation set out in the
papers we followed in other regards. One such being that although we used the same word frequency
threshold (5) as indicated in the paper, with the subset of the data we used, this yielded a vocabulary of
approximately 8k words, whereas Vinayals et al (2014) specified a vocabulary of 10k which implies our
model had to learn nearly just as many representations with nearly one third of the training examples.
These discrepancies from the original architecture could have all contributed to the overall worse
performance.

An enhancement to the model we would have liked to explore is to evaluate differences in model
performance between greedy and beam search for caption generation. Typically, beam search tends to
yield higher quality captions as instead of choosing the next token in the sequence with the highest
probability, based on the hyperparameter beam size (k), the algorithm continues to build the caption by
choosing k next tokens with the highest probabilities until maximum caption length is reached or the
<EOS> token is encountered.








