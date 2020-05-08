# Adam v/s Gradient Descent

Let us begin by understanding the principle behind Gradient Descent.

## Gradient Descent:
Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model.
The aim of Gradient Descent is to alter the weights in such a way that we are able to reduce the cost. This allows the Neural Network to update its weights thereby allowing it to learn and understand the data it is given.

### Featrues of a Gradient Descent:

#### 1. Loss Section:
A Loss Functions tells us “how good” our model is at making predictions for a given set of parameters. The cost function has its own curve and its own gradients. The slope of this curve tells us how to update our parameters to make the model more accurate.

#### 2. Iterations/Steps:
The basic premise of Gradient Descent is to reduce our Loss Function. We can understand that the loss function can be represented graphically whereby, changing different weights gives different Costs. We can also understand that for any learnable Dataset there will a Global Minima, where we can say that the model has learnt to predict/classify the Dataset to its best ability. Each iteration of Gradient Descent aims to alter its weights to reach closer to this Global Minima. 
For further development of our understnaidng let us assume that the Cost Function has a Graphical Representation. Therefore, we can say that the slope of any particular point, tells us the relative point our data is compared to local minima. 
Slope or differential here, if greater than zero tells us that if the weight is increased further, it will continue to move away from the global minima. 
However, if the slope is lesser than zero, it tells us that the weight if increased will move us closer to the Global Minima. Therefore, we can clearly see that the weights are to be adjusted in the opposite direction to that of the current differential. So, we can write the explaination as follows:

weight_adjusted := weight - learning_rate*(cost_slope)

This is continued for multiple steps, thereby further bringing the cost lower and lower. This allows us to reach an optimised weight and thereby a better learned Neural Network.

#### 3. Learning Rate:
The size of these steps is called the learning rate. With a high learning rate we can cover more ground each step, but we risk overshooting the lowest point since the slope of the hill is constantly changing. With a very low learning rate, we can confidently move in the direction of the negative gradient since we are recalculating it so frequently. A low learning rate is more precise, but calculating the gradient is time-consuming, so it will take us a very long time to get to the bottom.

This concludes Gradient Descent.

Let us take a look at the disadvantages of Gradient Descent:

1. Convergence rate for Gradient Descent depends on the Learning Rate. 
..a. Although a greater Learning Rate allows for a faster convergence, however it brings out the possibility of Overshooting. Overshooting: basically we never realise convergence rather it skips the Minima due to its large Learning Rate which alters the weight drastically.
..b. A smaller Learning Rate on the other hand, will lead to the optimisation will need to be run a lot of times (taking a long time and potentially never reaching the optimum). This may not be practical becuase of the amount of the aggregate time it may require to finally reach a Convergence.

2. A single Learning Rate is poor. For example let us take a dataset, where, two features have vast variations in their dataset. We can will see that having the same Learning Rate gives poor performance.

Now let us begin with our understanding of the Adam Optimizer.

## Adam Optimizer:
The Adam optimization algorithm is an extension to stochastic gradient descent that has recently seen broader adoption for deep learning applications in computer vision and natural language processing. Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.

Adam was presented by Diederik Kingma from OpenAI and Jimmy Ba from the University of Toronto in their 2015 ICLR paper (poster) titled “Adam: A Method for Stochastic Optimization“. I will quote liberally from their paper in this post, unless stated otherwise.
Adam is different to classical stochastic gradient descent.
Stochastic gradient descent maintains a single learning rate (termed alpha) for all weight updates and the learning rate does not change during training. A learning rate is maintained for each network weight (parameter) and separately adapted as learning unfolds.
```
The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.
```

### Features of Adam:
#### 1. Alpha:
Also referred to as the learning rate or step size. The proportion that weights are updated (e.g. 0.001). Larger values (e.g. 0.3) results in faster initial learning before the rate is updated. Smaller values (e.g. 1.0E-5) slow learning right down during training
#### 2. beta1: 
The exponential decay rate for the first moment estimates (e.g. 0.9).
#### 3. beta2: 
The exponential decay rate for the second-moment estimates (e.g. 0.999). This value should be set close to 1.0 on problems with a sparse gradient (e.g. NLP and computer vision problems).
#### 4. epsilon: 
Is a very small number to prevent any division by zero in the implementation (e.g. 10E-8).

### Understanding Adam Algorithm:
Adam can be looked at as a combination of RMSprop and Stochastic Gradient Descent with momentum. It uses the squared gradients to scale the learning rate like RMSprop and it takes advantage of momentum by using moving average of the gradient instead of gradient itself like SGD with momentum.
Adam uses estimations of first and second moments of gradient to adapt the learning rate for each weight of the neural network.
#### Moment:
N-th moment of a random variable is defined as the expected value of that variable to the power of n. Basically: it can be written as,
`m = E(X^n)`
