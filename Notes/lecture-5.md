# Regularization
Weight Regularization
#### Recall: L2 regularization
- penalizes you for having values that minimize the cost function but where the weights are too large
    - weights with very large magnitude are a sign of overfitting, so you can add a penalty term to your loss function
- when you compute with backpropagation the regularization will affect all subsequent weights/weight updates and the weights learnt in the NN
- the L2 norm is the root of the sum of the squared weights

#### L1 regularization
- The L1 norm takes the sum of the absolute values
    - this encourages sparse weights (a lot of the weights will be 0)
    - the penalty forces you to decide which weights are important and as a consequence which hidden units does the network think are most important 
    - it breaks down our tranditional notion of vectors and instead shows the steps to get there along each dimension

### Early Stopping
- Recall: we split our dataset into training, validation and test sets
- the validation hasa measurement for how much we're overfitting, the intermediate stage between training and testing 
- we use the vlaidation set to our advantage to stop us from overfitting 
- the training cost doesnt factor in data that isn't seen i.e data from test and validation set
- so as we train the cost should go down and we should get close r to minimizing the cost function
    - does this necessarily mean we're good on unseen data? no necessarily
- often we see that if we also at every epoch (1 epoch = seen every point in training set once)
- at each epoch compute validation cost, if validation cost stops improving after several iterations stop training 

### Dropout
- Recall: In lecture 4, we discussed bagging 
    - draw m samples from our dataset (sampling with replacement)
    - train m models 
    - average the predictions of the m models 
    - how can we reproduce this in neural networks since it allows us to do this averaging
- we consider subnetworks of our nn and we take the prediction of the avg of those subnetworks
- Find a way to average the predictions of all possible subnetworks
    - we don't want to train all these subnetworks separately
    - idea: train them all at once
-For each minibatch during trianing, randomly eliminate non output nodes

# PCA
Rigorous Lin Algebra 


# Autoencoders 
(∩｀-´)⊃━☆ﾟ.*･｡ﾟ deep learning magic

For cases where we can't project it linearly

Decoder takes the encoder as input and the decoder tries to reconstruct the original input. We're glueing together two separate neural networks into one

Try and squish it into a lower dimensional representation and try and reconstruct the original value 

So we have input x and have an encoded representation of x called h and then the decoder tries to reconstruct x. The enncoder f takes x as input and spits out h and then the decoder g takes g and spits out x so g is approximately the inverse of f. think of it as some inversion mapping 

Denoising autoencoders
- take original input x and make a computer version x hat
- eg randomly  out some components add daussian noise to each
----

Mb you want to input english and decode it and then turn it into a french sentence during the decoding 

The recurrent trend of being able to black box most of these models
A neural net can learn any function

Just bc you can ge ta neural net to do what other things can do that doesn't mean you have to do it. For time, resource and knowing what to use at what time. Some cases its harder to train a nn than an svm so its a design problem/decision
