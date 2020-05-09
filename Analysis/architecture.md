# Neural Architecture
## Reason for Number of Layers and Cells:
The number of layers and cells required in an ANN might depend on several aspects of the problem:
1. The complexity of the dataset, such as the number of features, the number of data points, etc.
2. The data-generating process. For example, the prediction of oil prices compared to the prediction of GDP is a well-understood economy. The latter is much easier than the former. Thus, predicting oil prices might require more ANN memory cells to predict, with the same accuracy, as compared to the GDP.
3. The accuracy required for the use case. The number of memory cells will heavily depend on this. If the goal is to beat the state-of-the-art model, in general, one needs more LSTM cells. Compare that to the goal of coming up with a reasonable prediction, which would need fewer ANN cells.

### Smaller Number of Hidden Layers/ Neurons in each Hidden Layer:
1. Time required drasitcally reduces as well as time consumed.
2. Reduces possibility of overfitting by not overlearning any particular pattern based on the Dataset.


