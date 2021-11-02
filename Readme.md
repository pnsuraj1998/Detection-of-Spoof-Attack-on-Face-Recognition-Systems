# Spoof Attack detection on Face Recognition Systems #

With the rise of Face recognition systems,there has also been a steady increase in number of spoof attacks on the face recongiton system. In this porject, we build a Convolution Neural Network (CNN) model capable of detecting spoof attacks on the system.

## Training ##

In order to train our CNN model, we use MSU-USSA dataset which consits of 9,000 images (1,000 live subject and 8,000 spoof attack) of the 1,000 subjects. For any queries regarding dataset, please visit this [link](http://biometrics.cse.msu.edu/Publications/Databases/MSU_LFW+_back/). 

1. Run the following code to create a h5 file containing data and their associated labels (0 - Spoof, 1 - Live) :  
    ``` python data_extraction.py -i "path_to_dataset" ```
