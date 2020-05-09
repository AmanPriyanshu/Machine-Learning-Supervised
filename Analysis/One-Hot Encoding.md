# One Hot Encoding:

One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction.

## Use-Case:
One hot encoding is used for a categorical classification. Let us take a deeper look at the methods to encode any classifcation Dataset.

### What is Categorical Data:
Categorical data are variables that contain label values rather than numeric values.
The number of possible values is often limited to a fixed set.
Categorical variables are often called nominal.
So basically, we have various labels being assigned to a particular set. For eg. a Dataset to classify pictures of pets can have the labels, "Dogs", "Cats", "Mice", etc.

However, Many machine learning algorithms cannot operate on label data directly. They require all input variables and output variables to be numeric.
In general, this is mostly a constraint of the efficient implementation of machine learning algorithms rather than hard limitations on the algorithms themselves.
This means that categorical data must be converted to a numerical form. If the categorical variable is an output variable, you may also want to convert predictions by the model back into a categorical form in order to present them or use them in some application.

Also important is that some categorical data do not have a sequential pattern to labels, i.e. the labels are unrelated to each other, For Eg: Identifying one of the seven wonders of the world are unrelated to each other. This type of categorical variable is called an ordinal variable.

## Methods of Encoding:

### 1. Integer Encoding
As a first step, each unique category value is assigned an integer value.
For example, “red” is 1, “green” is 2, and “blue” is 3.
This is called a label encoding or an integer encoding and is easily reversible.
For some variables, this may be enough.
The integer values have a natural ordered relationship between each other and machine learning algorithms may be able to understand and harness this relationship.
For example, ordinal variables like the “place” example above would be a good example where a label encoding would be sufficient.

### 2. One-Hot Encoding
For categorical variables where no such ordinal relationship exists, the integer encoding is not enough.
In fact, using this encoding and allowing the model to assume a natural ordering between categories may result in poor performance or unexpected results (predictions halfway between categories).
In this case, a one-hot encoding can be applied to the integer representation. This is where the integer encoded variable is removed and a new binary variable is added for each unique integer value.
In the “color” variable example, there are 3 categories and therefore 3 binary variables are needed. A “1” value is placed in the binary variable for the color and “0” values for the other colors.

This creates an array, for Eg:
Red : [1,0,0]
Green: [0,1,0]
Blue: [0,0,1]

When dealing with cyber attacks we definitely know that there is no numeric relations among them, i.e. is to say we cannot say that SYN-attack is greater or less than UDP-attack. On the other hand in the case where we know that a relationship of sequence does exist. Such as, the size of animal we can assign values like: Mouse = 0, Cat = 1, Tiger = 2, Elephant = 3. This would allow the model to better develop an understanding even if there are no exact relations between the different features (/attacks) by one hot encoding.