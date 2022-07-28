
## Directions to run performance_analysis.ipynb on Google Colab
On Google Colab place the INM706_CW folder into the Colab Notebooks folder.

**Note: When unzipping the zip file locally, there may be nested INM706_CW folders.**
**Please use the bottom level INM706_CW folder.**

The path to the folder should be '/content/drive/MyDrive/Colab Notebooks/INM706_CW'

It is currently seeded with 1 image for the train/val/test subfolders of images_split/

We have split up 10k images from the flickr30k set into these folders on the google drive link below.

https://drive.google.com/drive/folders/1-Lz3OtVRpanYgkKX0ugTVF8MxYxgf2yo?usp=sharing

The original dataset can be found here:

https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

To load our trained model, get the .pth files from the link below and copy them in the INM706_CW/model_checkpoints folder.

https://drive.google.com/drive/folders/1-56n9AWjGCpCq08-XATVmc3bw0bIY1AG?usp=sharing

The notebook should be able to run with this setup. We recommend copying over the 1000 images from the test folder
on the google drive link above into your own drive to replicate the tests.
