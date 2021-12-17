# titanic_ai
AI regression algorithm that suggests which people died in titanic and which did not. Made with pandas and scikit-learn libraries in python

# Description of the way the code works
First things first we do data cleaning. We remove unneeded rows and fill null values in rows. So, for "Embarked" we simply fill all null values with 'S' string, since this a port in which the most people embarked on the vessel. Then we find the median age of every class and apply the function which fills in null age values. In a nutshell, we suggest that every person's unknown age is equal to whatever the average age is of that particular class. Sex column is converted to int.

Now I want to explain in detail what happens in 35-37 lines of code).
We define an object of StandardScaler class. Then we apply fit_transform method to x_train array (this is the data that our AI will be training on). X_train is 9-dimension's array. Fit_transform basically does fit and then transform the values with an algorithm that i'll explain in moment. And keeps the record of values while 'fitting' the data.
Fit_transform algorithm:
  1. For every column (in our case for 9 columns) find the mean and standard deviation. (fit part)
  2. For every value in every column transform the value following the formula: (x - μ) / σ. In which x is the value, μ - mean, σ - standard deviation.
This is what's called Z-Order.

After preparing the data we'll use K-Neighbors algorithm for predicting the values. I don't want to go too much in detail on how k-neighbors classifier works because it's well documented, unlike the fit_transform algorithm 
