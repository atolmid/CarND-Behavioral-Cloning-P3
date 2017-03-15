#**Behavioral Cloning Writeup** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/initial_data.jpg "Sample Initial Image"
[image2]: ./examples/additional_data.jpg "Recovery Image"
[image3]: ./examples/additional_data1.jpg "Recovery Image"
[image4]: ./examples/additional_data2.jpg "Recovery Image"
[image5]: ./examples/final_loss.png "Final loss"
[image6]: ./examples/model.png "Model Structure"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* preprocess.py containing the script to load and preprocess the data, and store them as a pickle file
* model.py containing the script to load and preprocess a second buch of data, and store them as a pickle with a different file name
(The above two could be merged, and use input parameters to define what gets loaded and what gets saved, however was not implemented due to time constraints)
* model.py containing the script to create and train the model
* model1.py containing the script to reload and further train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes.
The first layer is a Cropping2D layer, that removes the parts of the image that are of no interest (code line 83).
(The top where there is the sky, and the bottom where there is a part of the car visible) 

Then, there is a Lambda layer, that normalises the pixel values (code line 84)
There is a MaxPooloing2D layer after the last convolution, and after a Dropout layer, a Flatten layer (code line 108), that feeds the data to the fully connected (Flatten) layers.

The model initially included RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer .
However, an ELU activation seemed to produce better results, so it replaced RELU in the final model version. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 105, 114, 124). 

The model was trained and validated on different data sets using Keras generators to ensure that the model was not overfitting (code line 133). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 132).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and made adjustments, in order to use the left and right camera pictures in addition to the center camera.
In principle, I added a correction of 0.25 to the center angle in order to derive the left image angle, and reduced it by an equal amount,
so as to get the angle for the right camera image.
Then, I doubled the available data, by flipping each of the images, and using the oposite angle value (multiplication by -1)
Then, I created one new image for each of the images above, by randomly adjusting the brightness of each image.
I then created a second one for each image, this time adding random shadow to the original images.
The final step was to add the newly generated data to the original dataset.

A validation set was derived, after splitting the training data, and using 20% as validation set.

After not getting the desired car behaviour, I decided to generate additional data, so I added two more functions:
* one that was randomly shearing images, and 
* one that was randomly rotating them.

At this point, the number of data instances was the following:
* Training set:  192840
* Validation set:  48240

The car behaviour was now much better, however since it was still not the desired, I created some additional data, recovering from the left and right sides of the road.
They were processed in the same way as the previous set of data, and stored as well as pickle file.

The number of additional training data instances created was 75300

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to Create a number of convolution layers and fully connected layers, that I thought would suffice.

The following is the structure of the final model, which has one dropout layer in addition to the original, in order to reduce overfitting.




Then I trained the first version of the model for 7 epochs, and 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 81-129) consisted of a convolution neural network with the following layers and layer sizes:

![alt text][image6]

####3. Creation of the Training Set & Training Process


Initially, I used the data provided to us by Udacity. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I performed the procedure described above:

* Image flipping 
* Random brightness adjustment
* Random shadow addition

And after looking for more ways to generate data:
* Randomly image shear 
* Random image rotation

I randomly shuffled the data set and put 20% of the data into a validation set. 

As already mentioned, the number of data points was initially:
* Training set:  192840
* Validation set:  48240


The final training set consisted of the above, as well as the data manually created and preprocessed by me afterwards, ended up being 75300.




I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 6 as evidenced by the fact that when I initially trained the model for 7 epochs, the loss stoped decreasing after the sixth epoch.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

The training and validation loss for each epoch was the following:
____________________________________________________________________________________________________
Epoch 1/6
207900/207900 [==============================] - 7189s - loss: 0.0712 - val_loss: 0.0174

Epoch 2/6
207900/207900 [==============================] - 7112s - loss: 0.0265 - val_loss: 0.0152

Epoch 3/6
207900/207900 [==============================] - 7118s - loss: 0.0229 - val_loss: 0.0123

Epoch 4/6
207900/207900 [==============================] - 7091s - loss: 0.0205 - val_loss: 0.0117

Epoch 5/6
207900/207900 [==============================] - 7065s - loss: 0.0192 - val_loss: 0.0106

Epoch 6/6
207900/207900 [==============================] - 7053s - loss: 0.0183 - val_loss: 0.0107


Since the result was not initially the expected, I added an additional dropout layer, in order to reduce overfitting.
In addition, I changed the activation function from 'relu' to 'elu', which seemed to produce better results.

When trying to work with the extended dataset, there was an issue when trying to load all data.
The kernel was restarting at a particular point, and I decided that probably it was a memory issue.
Thus, instead of re-training the model from scratch, I loaded the already trained model, and trained it further, using the extra generated 'recovery' dataset.

I let it train for 6 epochs, again, with the following training and validation loss:

Epoch 1/6
75300/75300 [==============================] - 3134s - loss: 0.1275 - val_loss: 0.0215

Epoch 2/6
75300/75300 [==============================] - 3171s - loss: 0.1038 - val_loss: 0.0206

Epoch 3/6
75300/75300 [==============================] - 3066s - loss: 0.0931 - val_loss: 0.0240

Epoch 4/6
75300/75300 [==============================] - 2937s - loss: 0.1282 - val_loss: 0.0315

Epoch 5/6
75300/75300 [==============================] - 2927s - loss: 0.1068 - val_loss: 0.0299

Epoch 6/6
75300/75300 [==============================] - 2901s - loss: 0.0822 - val_loss: 0.0253



![alt text][image5]


After running the model, the car was driving, staying always inside the track.
I recorded a video, which shows the first two laps [Autonomous Driving in Simulator](https://github.com/atolmid/CarND-Behavioral-Cloning-P3/blob/master/run1.mp4).

