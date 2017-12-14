# **Traffic Sign Recognition** 

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

[data]: ./images/data-distribution.png "Data Distribution"
[lenet]: ./images/lenet.png "LeNet"
[prediction]: ./images/prediction-1.png "5 signs and predictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/html5cat/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `35,288`
* The size of test set is `12,630`
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed between different classes (traffic signs) and numbers of training(blue), test(green) and validation(lightblue).

![alt text][data]

### Design and Test a Model Architecture

The code for this step is contained in the 4th code cell of the IPython notebook.

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set (90%) and validation set (10% of data).

My final training set had 35288 number of images. My validation set and test set had 3921 and 12630 number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. It's important to shuffle the data, otherwise ordering of training data can have negative impact on the correctness of model.

The code for my final model is located in the 7th cell of the ipython notebook. 
For the model we'll be using Yan LeCun's LeNet with ReLU activation function:
![alt text][lenet]

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 	     			|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten       		| input 5x5x16, outputs 400       				|
| Fully connected		| input 400, outputs 120 						|
| RELU					|												|
| Fully connected		| input 120, outputs 84       					|
| RELU					|												|
| Fully connected		| inputs 84, outputs 43  						|
 


The code for training the model is located in the 11th cell of the ipython notebook. 

We're going to use 10 epochs for now since it produces a good balance of time to train and results.
Batch size is guided by how many images can fit in memory at given point and we'll use 128 like we did in LeNet lab.

For optimizer we are using AdamOptimizer with learning rate of `0.001`. 
For convolutional layers the parameters `mu = 0` and `sigma = 0.1` are used.

I've tried a few simpler models before this one, but after reading http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf and going through https://github.com/udacity/CarND-LeNet-Lab it became obvious that this is a great architecture for the model.

The code for calculating the accuracy of the model is located in the 12th cell of the Ipython notebook.

My final model results were:
<!-- * training set accuracy of `` -->
* validation set accuracy of `0.939` 
* test set accuracy of `0.864`

After just 10 epochs of training and a pretty disbalanced data-set these results seem great.

### Test a Model on New Images

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

Here are five German traffic signs that I found on the web and the corresponding prediction probabilities:

![alt text][prediction] 

Here are the results of the prediction:

After finding a mistake in classification using 5 epocs I've upped the number to 10 and the model now predicts correct signs with 98-100% probability rate and very low rates of wrong suggestions.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Stop sign   									| 
| 0.98     				| Speed Limit (50) 								|
| 1.00					| Yield											|
| 1.00	      			| No passing					 				|
| 1.00				    | Priority Road      							|

The images used for testing are very clear and exactly same size as the training set. As you can see accuracy is 80%. While test accuracy being 85.8%. The model incorrectly classified 60km/h sign as 50km/h. At least it added a 1% confidence that it's 60 :) 

Depending on the training luck the model sometimes actually correctly predicts all 5.

I tried it on the bigger images from the internet with little luck, for that to work I'd need to add the processing to increase contrast and reduce the size.

For future development this model could be improved by altering the training set and balancing it.
