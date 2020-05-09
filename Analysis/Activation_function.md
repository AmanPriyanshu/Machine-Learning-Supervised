# Activation Functions:
In artificial neural networks, the activation function of a node defines the output of that node given an input or set of inputs. 

## Importance of ReLU:
Major benefits of ReLUs are sparsity and a reduced likelihood of vanishing gradient.
1. Benefit of ReLUs is sparsity. Sparsity arises when a≤0. The more such units that exist in a layer the more sparse the resulting representation. Sigmoids or any other activation function on the other hand are always likely to generate some non-zero value resulting in dense representations. Sparse representations seem to be more beneficial than dense representations.
2. One major benefit is the reduced likelihood of the gradient to vanish. This arises when a>0. In this regime the gradient has a constant value. In contrast, the gradient of sigmoids becomes increasingly small as the absolute value of x increases. The constant gradient of ReLUs results in faster learning.

## Importance of Binary Encoding in the Last Layer:
Here, we are deciding the class/label of the attack. 
1. Having a binary classification allows the model to better differentiate between unscalable data, for example the different attakcs, which cannot be scaled over integeral values. 
2. Also having each label range from [0,1] allows to have an accurate probability with which a particular session may have been categorized as a particular class.