## Face_Recognition
This is one of the famous applications of computer vision.
In most of the companies, they use facial recognition for their attendence systems and to identify the culprit, we can use this system.
In this project, goal is to train the model with some person's facial image with their name as a label and try to test with the image which is not in the part of training.

## Folder structure
Fo this project, there are two ways to pass the label to the model.
One is passing the image name along with it's label in an text file.
Second one is by keeping the images of particular label in an folder and keep the folders in an structural way.

I used the second way so I kept the daata as in image.
https://github.com/aravindsraj/Face_Recognition/blob/master/images/folder%20structure.png

## Training
For training I have used InceptionV3 pre-trained model.
This is one of the powerful model which have trained on many layers.
Here I have removed the last layer of the model since it has 1000 layers. 
For my training, I used only 4 labels. So instead of 1000, I removed that last layer, flattened it and added a dense layer with 4 layers softmax activation function.
For dynamic usage, instead of mentioning the number of layers, I passed the number of folders.
I followed a particular folder structure as mentioned above so that it can get the number of layers from there.

The code "train.py" is to train the model.

## Testing with test data
This is one such a crude way I have used.
Once the training is over, we can proceed to test the model with new images.
Since my model has train with new four labels, I checked the predicted value one by one for each label.

The code "test.py" is to test the model.
