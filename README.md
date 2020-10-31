# Wine quality neural network

## About
A neural network for prediction of red wine quality grades. This project was made for the Artificial intelligence course using Python.

## Description
The dataset consists of 1599 rows of data and 12 columns - the first 11 columns are the features (physiochemical characteristics), and the last column represents the target values which range from 0 to 10.

A statistical analysis is conducted before the modeling and execution of the neural network. This analysis consists of data description, correlation matrices and distribution histograms for each column in the dataset.

Values in the final column (wine quality grades) are reduced from 11 to only 3 (5, 6 and 7) due to low presence of other values in the dataset.
The dataset is divided into a training set (80%) and a test set (20%). The deep learning model runs 1700 epochs, with a batch size of 800.
The model is made of 11 input values, 4 hidden layers with 11, 10, 8 and 6 nodes, respectively, and 3 output values.

The evaluation process leads to an unpromising precision value and a relatively high loss value, most likely due to the subjective method of quality assesment.
Ten samples are used during the prediction process as demonstration of how well the model can predict the quality selected wine samples. Grades are mostly accurate, but can deviate by 1 in certain cases, in accordance with the results of the model evaluation.

## Libraries Used

* Pandas
* Matplotlib
* Numpy
* TensorFlow
* scikit-learn
