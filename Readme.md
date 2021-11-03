# Spoof Attack detection on Face Recognition Systems #

With the rise of Face recognition systems,there has also been a steady increase in number of spoof attacks on the face recongiton system. In this porject, we build a Convolution Neural Network (CNN) model capable of detecting spoof attacks on the system.

## Training and Inference ##

In order to train our CNN model, we use MSU-USSA dataset which consits of 9,000 images (1,000 live subject and 8,000 spoof attack) of the 1,000 subjects. For any queries regarding dataset, please visit this [link](http://biometrics.cse.msu.edu/Publications/Databases/MSU_LFW+_back/). 

Given an image or a video frame, we use MTCNN to detect face and extract 96X96 patch around SIFT key points located on the face image.


1. Run the following code to create a h5 file containing data and their associated labels (0 - Spoof, 1 - Real) from dataset :  
    ``` python data_extraction.py -i "path_to_dataset" ```
2. To display model architecture, run following code:  
   ``` python model.py ```
3. To train CNN model, run following code:   
  ``` python -i "path_to_h5_file" -e no_of_epochs -b batch_size ```.   
    After training and testing, the model will be saved defaultly in the same project folder for every 10 epochs.
4. In order to run inference on a video, run following code :  
  ``` python inference.py -i "path_to_trained_cnn_model" ```

