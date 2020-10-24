# Terminology, Examples and Social Impact of Machine Learning
idea: make any drawing into pepe, tweetbot for hasan lmao, pacman but the ghosts learn, mario but the enemies learn, kpop song lyrics generator, RAP LYRICS GENERATOR, calssical music in the style of "x composer" web app, wait but i think those already exist, make bass tabs from a recording, what clothes do i like, ai that speedruns mario 64 lmao mb start with a 2d game, train model to play among us omg they use tts or they type to speak idk, make a math rock song lmao, omg make the "ideal" anime opening from ML data, turn rap audio into violin midi lmao, 

AI: the broadest term
Machine Learning: a subset of ai
Deep Learning: Area of machine learning that uses deep neural networks to make predictions form data, a subset of machine learning

## Applications of ML

__Computer Vision:__ Image data
- object detection
- segmentation
- handwriting recognition

__NLP:__ Text and speech data
- translation
- summarization
- sentiment analysis

__Reinforcement Learning__
- game-playing
- decision-making
- idea: Agent occupies a state in the world and the agent decides which actions to perform in order to get the highest rewad possible.

and many more!

## Terminology

__Dataset:__ collection of data used for learning |
e.g. the MNIST dataset, the iris flower dataset

Datasets are composed of instances (examples) | e.g. this image (along with the label "3") is an instance of MNIST

Take your data set instance and turning it into a language the model can understand

A dataset instance consists of two parts:
- Features/input variables: relevant information about the instance (e.g. pixel values in MNIST image, petal length of an iris flower)

Reinforcement Learning

Unsupervised Learning

Supervised Learning

Most common flavour of ML.

Goal: given the features of an instance, predict its label(s)

All example come with labels. the dataset is split into training, validation and test sets. 

During training we show our ML model examples formteh training set:
- get the model to predict the labels given the features
- compute the model loss (our way of measuring error)
-adjust model in an attemp to minimize loss i.e how far off was the models prediction on the training set from what they actually were.

Once trained, we evaluate the model's predictions on the vlaidation set. Usually we have more than one model, we use the validation peroformance to decide which model to use.

You test set is your measure on how well the model does on examples never seen before. You do not see the test set at all when choosing or training the model. You only see it at teh end during the final test phase and it will be the measure of generalization. How well does it do on new examples. You only evaluate on your test set ONCE!!

If we evaluate on the test set more than once, we fall into the trap of overfitting. We become really good at predicting for our specific test case. Overfitting is like looking at the answers to a test before hand lmao. sure youll get 100 but you only know how to do those specific questions you couldn't do anything else.

Generally speaking, there are a lot of things you learn about model training. There are hyper parameters in models that depend on you, chosen based on intuition or wtvr. If you model is catastrophic ya ok u need to fix smthing but often just changing small stuff in your training process will significantly improve results.

"Sometimes the features in the dataset are just not good at predicting the label!! You might just need more data or more features or better feature extraction" - William Tang

Each row represents an instance and each column is a feature i.e petal length or sepal length in the columns and their values in the x.

We put these features from the dataset into our model andwe get an out put and then compare them to the actual output for those instances. 

### Regression vs Classification

Regression

Learning a function for a continuous output

e.g. predicting sale prices of houses


Classification


## Unsupervised Leaning

### Dimensionality Reduction

Learn a way to reduce the dimension of the data and reduce the number offeatures and learn which combinations of features have the most variability. 

## The ML pipeline

The answer to 'How can we solve a problem with mahcine learning?'

7 step process

1. Data,data and more data

Figure out what data is needed and where to retrieve it. Does similar data exist or do we need to generate it? Remember: we cannot use ML if we don't have data.

2. Formulate the Problem

Supervised or unsupervised? If it's supervised are we doing classification or regression? what are we tring to predict (labels)? What would we consider to be good predictors (features)?

3. Process Data

Format data that can be interpreted by a computer. That includes cleaning, manipulating and extracting important features to feed into the trianing model. SPlit into training and test sets. Decide how validation will done. Two ways: a spearate validation set or cross-validation

2. Choose Model(s)
Which machine learning models will we train on our data? Different amchine leanring models make diferent assumptionsabout the data they are dealing with. Some work for some problems and others really don't work. So you need to choose a model that goes with the problem youre triyng to solve

5. Choose evaluation metric
How do we measure the performance of our models? this requires some thinking. though it's easy to default to accuracy, this is not always a good choice. 

6. Train and validate

Train the model and then perform validation. Use the validation performance as an indicator of how well the model is doing. Decide what changes to the model you want to make. Repeat.

Validation: allows us to undestand how well our trained model performs before proceeding to final testing. we cant know this from the trianing set where the model has access to the labels.

Two oinions:
- validation set: keep a portion of the dataset for validation (separate from training and test sets)
- Cross validation: split up the training set into "folds"
     - expanation he skipped
     - take training set and split it into pieces and we train the model on the 
     - you end up training and validating k times and take the perofrmance avg on the k folds to get the final validation score
     - only rly recommended for a smaller data set bc you would need to iterate over huge amounts of data multiple times.

Validation lets us tune model hyperparameters. These are model settings that stay constant during training (e.g. the choice of model itself, the leanring rate during gradient descent, $\lambda$ fr regularization, fropout for neural networks, etc.). This notion will become clearer as we explore specific ML models

It can tells us if the model is overfitting to the training data. We can see this happen when, during training, validation error starts going up,even though the training error is going down. When this happens we say that the model is learning the noise in the data as opposed to the patterns we want it to learn. 


image on how overfitting affects prediction

7. 


## K-nearest Neighbours

First ML model

When faced with a new situation we rely on our experience from similar situations to know what to do. Similar in ML. If we see a new data point, we can look at instances in the training set that are most similar to this data point and use their labels to help decide what label we give to the new point.

A neighbour is any point for which we know the label. 

The K-value is really important. Super small values of k would result in a region around outliers where we'd predict blue even when the overwhelming majority of points are orange. So we need a larger value of K to no be as sensitive to outliers and get a better decision boundary. 

### Difficulties 

Very memory intensive algorithm and the training set needs to be stored in memory for the duration of the algorithm and can be super inefficient.

Breakdown of local methods in High Dimensions

Say we have a dataset of 10 images of cats and dogs and we want to build a classifier. We first consider one feautre like the avg red color in the image but its not great so we tae the blue and so on and then its better but oof as the number of dimensions increases we need exponentially more amount of data to get the same data density for each added dimension.

The curse of dimensionality applies to all of ML and is why we want to reduce dimensionality and ??? i missed the other aspect

## Bias, Fairness and Ethics

One of the most important issues in ML. ML learning systems deplayoed in the real world have been found to discriminate based on the basis of race and gender, among others. All machine learning practioners ...

People are biased so the data people generate is biased. UNbalanced data distributions (e.g. dataset of mostly white faces) Correlation between input features and race, gender, etc. (e.g. correlation between race and healthcare expenditures, heavy police presence in racialized communities in the US) these arise from deeper societal problems.

The loos functions of machine leanring models do not penalize bias or enforce fairness.

__Talks about this issue:__

Ruha Benjamin's talk at ICML is v informative about the interplay between racial bias and technology 

[Kate Crawford at Neurips 2017](https://www.youtube.com/watch?v=6Uao14eIyGc&ab_channel=SurajNath)

### Ethics Concerns
Collecting users' personal data and feeding them into ML systems

Facial recognition for surveillance

Autonomous weapons

Which decisions do we trust an algorithm to make
- how "accurate" whould such an algorithm have to be 
- would you put your safety in the hands of self-driving car
- would you like decisions about your medical care to be made by an algorithm

Recource: MAIEI



