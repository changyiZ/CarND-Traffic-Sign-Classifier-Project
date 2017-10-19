#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./samples/training_data_graph.jpg "Train graph"
[image2]: ./samples/sample_grayscale.jpg "Grayscaling"
[image3]: ./samples/sample_augmentation.jpg "Random Noise"
[image4]: ./test_images/example_0.png "Traffic Sign 1"
[image5]: ./test_images/example_21.png "Traffic Sign 2"
[image6]: ./test_images/example_28.png "Traffic Sign 3"
[image7]: ./test_images/example_39.png "Traffic Sign 4"
[image8]: ./test_images/example_40.png "Traffic Sign 5"
[image9]: https://camo.githubusercontent.com/3b43f4d1f9a91e44b0373838537daed273b740a0/68747470733a2f2f6769746875622e636f6d2f6a6572656d792d7368616e6e6f6e2f4361724e442d4c654e65742d4c61622f7261772f636434626139373930363137366536303230613462336330383462373531386566336464656435652f6c656e65742e706e67 "Source: Yan LeCun"
[image10]: ./samples/augmentation_data_graph.jpg "Augmentation graph"
[image11]: ./samples/sample_normalized.jpg "Normalized sample"
[image12]: ./samples/sample_prediction.jpg "New image prediction"
[image13]: ./samples/modified_LeNet.jpg "Modified LeNet"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/changyiZ/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this worked well for Sermanet and LeCun as described in their traffic sign classification article and do help computation performance as dimension reduced from 3 to 1.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data  to the range (-1,1) because it benifits for training data and weights tuning, as described in this article [glossary-of-deep-learning-batch-normalisation](https://medium.com/deeper-learning/glossary-of-deep-learning-batch-normalisation-8266dcd2fa82)
![alt text][image11]


I decided to generate additional data because the number of samples varies comparatively great for different classes. 

To add more data to the the data set, I used 5 functions for augmenting the dataset: random_translate, random_scale, random_warp, random_rotate and random_brightness.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 
![alt text][image10]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

At first, I used the same architecture from the LeNet Lab solution [LeNet](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb)
![alt text][image9]
But it seemed that the accuracy was hard to meet the least requirement(around 0.9).
So I decided to make a little change on it, adapted from Sermanet/LeCunn traffic sign classification journal article.
![alt text][image13]

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| ReLU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| ReLU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x400 	|
| ReLU					|	
| Flatten layers (5x5x16 -> 400) |
| Flatten layers (1x1x400 -> 400) |
| Concatenate 400 + 400 -> 800 |
| Dropout 0.5 |
| Fully connected		| 800 -> 43 |  
 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer (already implemented in the LeNet lab). 
The final settings used were:
* batch size: 128
* epochs: 20
* learning rate: 0.001
* mu: 0
* sigma: 0.1
* dropout keep probability: 0.5

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.952 
* test set accuracy of 0.936
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)      		| Speed limit (20km/h)   									| 
| Double curve     			| Double curve 										|
| Children crossing					| Children crossing											|
| Keep left      		| Keep left				 				|
| Roundabout mandatory			| Roundabout mandatory      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. I think it is more lucky comparing favorably to the accuracy on the training set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the last two code cell of the Ipython notebook.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0 | Speed limit (20km/h) | 
| 1.0 | Double curve 								|
| 1.0	| Children crossing				|
| 1.0	| Keep left				 				   |
| 1.0	| Roundabout mandatory |


Here is top 3 prediction graph below:
![alt text][image12]


