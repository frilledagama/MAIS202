# Neural Networks

## History

This is part of a school of AI called connectionism, inspired by the way the brain works.

## Basics

__Perceptrons:__ the neurons of artifical neural networks. How can we model this in a mathematical sense.

OG method can lienarly classify data. So given some data it'll draw a line and separate it. It just draws a line not THE best line (the SVM draws the line of best fit in comparison

The sigmoid should always be positive

Here you punish the model the same way you'd punish linear regression. There is some cost function that we are minimizing and we use the L2 norm

If z is the output of the model and say you are approximating g


one hot encoding: turns dog/cat into a 2 vector where its 1 0 for cat and 0 1 for cat !! i seeee

So explain the note on/note off 


without back propagation there is no trianing of neural networks. it wouldnt be feasible 

COmputational graph

have nodes a and b that get fed into a new node c and added together.

y hat = signmore function (w dot inputs) can also be drawn asa ocmputation graph

Represent cimputation of the cost with a graph (forward pass)
-----


A training step is one gradient update. In one step batch_size many examples are processed.

An epoch consists of one full cycle through the training data. This is usually many steps. As an example, if you have 2,000 images and use a batch size of 10 an epoch consists of 2,000 images / (10 images / step) = 200 steps.

If you choose our training image randomly (and independent) in each step, you normally do not call it epoch. [This is where my answer differs from the previous one. Also see my comment.]

---
Aritifically augmenting data from r trianing set is something cool/important to do 