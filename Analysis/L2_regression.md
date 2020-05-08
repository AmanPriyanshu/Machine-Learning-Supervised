# REGULARIZATION:

## Overfitting:
is a phenomenon that occurs when a machine learning or statistics model is tailored to a particular dataset and is unable to generalise to other datasets. This usually happens in complex models, like deep neural networks.
## Regularisation:
is a process of introducing additional information in order to prevent overfitting.

#### L1 meaning: ||w||(1) := |W1| + |W2| + |W3| + ... + |Wn|
#### L2 meaning: ||w||(2) := ( |W1|^2 + |W2|^2 + |W3|^2 + ... + |Wn|^2 ) ^ (1/2)

Let us say predicted_y is:
#### y_pred = W1X1 + W2X2 + W3X3 + ... + WnXn 

#### loss = Error(y, y_pred)
#### loss_with_reg_L1 = Error(y, y_pred) + lambda * ( |W1| + |W2| + |W3| + ... + |Wn| )
#### loss_with_reg_L2 = Error(y, y_pred) + lambda * ( |W1|^2 + |W2|^2 + |W3|^2 + ... + |Wn|^2 )

Note there is no sqrt(2) during loss calculation.
Here, lambda is greater than zero. It is manually tuned in both cases.

If we attempt to differentiate both of them we get the following:
### Gradient:
#### 1. L1:
... d(Error(y, y_pred))/dW + d(lambda * ( |W1| + |W2| + |W3| + ... + |Wn| ) )/dW
... so basically other than W = 0, d(|W|)/dW is either 0 or 1 depending on whether W is greater than or lesser than 0.
... Final: `W_new = W - alpha * (2x(wx + b -y) + lambda) if W>0`
... .      `W_new = W - alpha * (2x(wx + b -y) - lambda) if W<0`

#### 2. L2:
... d(Error(y, y_pred))/dW + d(lambda * ( |W1|^2 + |W2|^2 + |W3|^2 + ... + |Wn|^2 ) )/dW
... so basically other than W = 0, d(|W|)/dW is (2 * W1).
... Final: `W_new = W - alpha * (2x(wx + b -y) + 2*lambda*w)`


### Prevention of Overfitting:

1. Let’s say calculating w-H, for a prticular model, gives us a w value that leads to overfitting. Then, intuitively, w_new from regularization will reduce the chances of overfitting because introducing λ makes us shift away from the very w that was going to cause us overfitting problems in the previous sentence.
2. Let’s say an overfitted model means that we have a w value that is perfect for our model. ‘Perfect’ meaning if we substituted the data (x) back in the model, our prediction ŷ will be very, very close to the true y. Sure, it’s good, but we don’t want perfect. Why? Because this means our model is only meant for the dataset which we trained on. This means our model will produce predictions that are far off from the true value for other datasets. So we settle for less than perfect, with the hope that our model can also get close predictions with other data. To do this, we ‘taint’ this perfect w in w_new with a penalty term λ.
3. Notice that H or final evaluation is dependent on the model (w and b) and the data (x and y). Updating the weights based only on the model and data in can lead to overfitting, which leads to poor generalisation. On the other hand, in Equations weights with regularization, the final value of w is not only influenced by the model and data, but also by a predefined parameter λ which is independent of the model and data. Thus, we can prevent overfitting if we set an appropriate value of λ, though too large a value will cause the model to be severely underfitted.

### L1 vs L2:
1. Take a look at L1, If w is positive, the regularisation parameter λ>0 will push w to be less positive, by subtracting λ from w. Conversely, if w is negative, λ will be added to w, pushing it to be less negative. Hence, this has the effect of pushing w towards 0.
As mentioned above, as w goes to 0, we are reducing the number of features by reducing the variable importance. In the equation above, we see that x_2, x_4 and x_5 are almost ‘useless’ because of their small coefficients, hence we can remove them from the equation. This in turn reduces the model complexity, making our model simpler. A simpler model can reduce the chances of overfitting.
2. Taking a look at L2, we can see that though it originally does not create sparse models, it can however, create a much elegant and uniformed regularization. It allows for regularizing over a near spaced data.
