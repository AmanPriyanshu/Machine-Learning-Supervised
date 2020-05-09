# Activation Functions:
In artificial neural networks, the activation function of a node defines the output of that node given an input or set of inputs. 

## Importance of ReLU:
Major benefits of ReLUs are sparsity and a reduced likelihood of vanishing gradient.
1. Benefit of ReLUs is sparsity. Sparsity arises when a≤0. The more such units that exist in a layer the more sparse the resulting representation. Sigmoids or any other activation function on the other hand are always likely to generate some non-zero value resulting in dense representations. Sparse representations seem to be more beneficial than dense representations.
2. One major benefit is the reduced likelihood of the gradient to vanish. This arises when a>0. In this regime the gradient has a constant value. In contrast, the gradient of sigmoids becomes increasingly small as the absolute value of x increases. The constant gradient of ReLUs results in faster learning.