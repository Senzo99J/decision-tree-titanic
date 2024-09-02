# This task we are creating a decision tree that can predict the survival of the passengers on the titanic.The dataset contains information about the passengers who were aboard the Titanic, including demographics, ticket class, and fare information.
## Install and Import Libraries on Jupyter Notebook to continue with this task
import pandas as pd 
from sklearn.model.selection import train.test.split, GridSearchCV
fromsklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy.score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
### Load and Explore the Data
Present first five observations
Check datatypes
Drop any column from the dataset
#### One-Hot Encoding
One-hot encoding is a technique used to ensure that categorical variables are better represented in the machine. Let's take a look at the "Sex" column
Machine Learning classifiers don't know how to handle strings. As a result, you need to convert it into a categorical representation. There are two main ways to go about this:

Label Encoding: Assigning, for example, 0 for "male" and 1 for "female". The problem here is it intrinsically makes one category "larger than" the other category.

One-hot encoding: Assigning, for example, [1, 0] for "male" and [0, 1] for female.
In this case, you have an array of size (n_categories,) and you represent a 1 in the correct index, and 0 elsewhere.
In Pandas, this would show as extra columns. For example, rather than having a "Sex" column, it would be a "Sex_male" and "Sex_female" column.
Then, if the person is male, it would simply show as a 1 in the "Sex_male" column and a 0 in the "Sex_female" column.

##### Select relevant columns
Filter the dataframe only include relevant
Handle missing values in 'age' column by filling with median age
Split the data into features(x) and target(y)
Split the data into training(70%) and remaining(30%)
Split the remaining data into development(15%) and test(15%) set.
Verify the splits
Train a decision tree classifier
Plot the decision tree

Compute your model accuracy on the development set
List to store accuracies
Loop through the different values of max-depth
Train a decision tree classifier with current max-depth
Predict on training and development sets.

Plot the decision tree
Print accuracies for the current max-depth
Plot accuracies

Plot line of your training accuraciesand another of your development accuracies in the same graph.
Write down the shape of the lines and what is shape means.

Identify the optimal max-depth
Match the range start at 2
Train the final model using the optimal max-depth
Evaluate the final model on the test set
Plot the final decision tree

Train Bagging, RandomForest, and Boosted models
Fit the model 
Predict on the development set
Calculate accuracy
Plot features importances

Tune n_estimators and max-depth for the best performing model
Get the best parameters and best estimator
Report final accuracies
Identify the best performing model
