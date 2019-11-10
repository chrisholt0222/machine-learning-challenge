# Machine Learning - Exoplanet Exploration

![exoplanets.jpg](Images/exoplanets.jpg)

## Background

Over a period of nine years in deep space, the NASA Kepler space telescope has been out on a planet-hunting mission to discover hidden planets outside of our solar system. The project goal is to develop machine learning models capable of classifying candidate exoplanets from the raw dataset.

The field: koi_disposition (depenendent variable) contains the assigned disposition of each observation in the raw data. The assigned disposition has the following options: 
 * "CONFIRMED": object confirmed to be exoplanets.
 * "CANDIDATE": have not yet been formally classified.
 * "FALSE POSITIVE" objects determined not to be exoplanets.
For training purposes, all rows from the raw data assigned "CANDIDATE" have been dropped. 

## Results

Using several models (logistic regression, svm, and neural network model) trained on "CONFIRMED" and "FALSE POSITIVE" records, the "CANDIDATE" records were assigned to either "CONFIRMED" (an exoplanet) and "FALSE POSITIVE" (not an exoplanet). The neural network model (accuracy of 0.9910) was selected for the final result ("Final_Selected" in the file: (https://github.com/chrisholt0222/machine-learning-challenge/blob/master/candidates.csv)).

The jupyter notebook: (https://github.com/chrisholt0222/machine-learning-challenge/blob/master/model_final.ipynb) loads each of the models and applies the scalers to predict the exoplant classification of each "CANDIDATE" record.

### Preprocess the Data

The mean and standard devation was calculated for each field (independent vcariable or feature) by koi_disposition ("CONFIRMED", "FALSE POSITIVE"). A histrogram for each independent variable was generated. Based on the review of the mean, the standard devation, and the histograms the following variables koi_period_err2, koi_time0bk_err2, koi_duration_err2, koi_depth_err2 were removed. See Jupyter Notebooks for each model: logistic regression, svm, neural networks - (https://github.com/chrisholt0222/machine-learning-challenge/tree/master/models_outcomes2). The logistic regression model, model_logreg.ipynb, contains the feature review along with the recursive feature elimination process.

### Histograms

#### Fields 1 - 16:
![Group1.jpg](models_outcomes2/Images/feature_hist_0_15.png)

#### Fields 17 - 32:
![Group2.jpg](models_outcomes2/Images/feature_hist_16_31.png)

#### Fields 33 - 40:
![Group3.jpg](models_outcomes2/Images/feature_hist_32_39.png)

The following was applied to the data before use in each model:
* NAs were removed.
* Drop all rows were koi_disposition = "CANDIDATE".
* Drop the fields: koi_period_err2, koi_time0bk_err2, koi_duration_err2, koi_depth_err2.
* The data was separated into training and testing data using the `train_test_split` function.
* The `MinMaxScaler` function is used to scale the numerical independent data.
* The `LabelEncoder` and `to_categorical` were used to cattegorize the dependend variable (koi_disposition).

### Feature Selection - RFE

The RFE function (recursive feature elimination) was used with the LogisticRegression model to eliminate unnecesasry features (independent variables). Features ranked below 1 where removed from future models. 

### Tune Model Parameters

The `GridSearchCV` fundtion was applied to the following models: `SVC` (support vector machine) and `Sequential` (neural network) to tune model parameters for each model. (See model_logreg.ipynb for the removed features.)

### Analysis

The inital model was a binary classifier using logistic regression without scaling the features. The model yielded predictive accuracy of approximately 0.6606 on the testing data. Using MinMaxScaler to scale the independent varialbes, the result of the logistice regression model inmproved to 0.9902. After applying the feature elimination, the model test results are unchanged at 0.9902.

The next two model are applied to the fully transformed and reduced data. 

Usnig a Support Vector Machine (SVM), the reasults are similar at 0.9902. Tuning the hyperparameters C and Gamma using `GridsearchCV`, the best model is C = 1.0, and Gamma = 0.0001, with a predictive accuracy of 0.9902 on the testing data. No real change from the initial SVM model.

A `Sequential` model achieved a predictive accuracy score of 0.9917 on the test data. This model used the `adam` optimizer, the `categorical crossentropy` loss function and two hidden layers with 40 and 20 nodes. Using `GridsearchCV` with option for the parameters: batch_size, epochs, and optimizer did not improve the results.

The selected model is the `Sequential` model.