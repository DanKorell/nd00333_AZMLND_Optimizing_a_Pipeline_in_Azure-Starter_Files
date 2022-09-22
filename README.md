# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
This dataset contains data about bank marketing, with a CSV file containing details on 32950 people, with information such as their age, job, marital status, education level, along with some numerical columns. Unfortunately, a description for the dataset has not been made available, only a download link, so no descriptions of these numerical values are present. The final column of the bank marketing CSV is labelled "y" and contains values of 0 and 1. This is the field we want to predict and so this is a binary classification problem. This field could represent whether or not the bank was successful in marketing some product to this customer and therefore being able to predict these successes would enable the bank to more effectively target customers for the marketing of this product.

Two machine learning solutions were developed.

The first being a Scikit-learn logistic regression model, for which Hyperdrive was used in the Azure ML SDK to run it multiple times with different values for the two hyperparameters (`C`: inverse of regularization strength, `max_iter`: the maximum number of iterations to converge).

The second being an AutoML solution running in the Azure ML SDK. With accuracy used as the primary metric, the best Hyperdrive run resulted in 91.029% accuracy, whilst the best AutoML model resulted in 91.745% accuracy. This AutoML model was a Voting Ensemble, consisting of XGBoost and LightGBM classifiers.

## Scikit-learn Pipeline
#### Pipeline architecture
The Scikit-learn pipeline was provided by Udacity for this project via the train.py script.

With the bank marketing CSV file identified by its URL, this was loaded into a tabular dataset and run through a cleaning and one-hot encoding process as follows:
- rows with missing values dropped (though as it happens, none were removed as the cleaned dataset still had 32950 rows, so there were no missing values)
- the `job`, `education` and `contact` columns one-hot encoded
- the `marital`, `default`, `housing` and `loan` columns encoded numerically with positive values being encoded as 1
- the `month` and `day_of_week` columns being encoded as numerical values as per the dictionaries defined in the script
- the `poutcome` column being encoded as a 1 for values of "success" and 0 otherwise
- the `y` column being split out into a separate dataframe and encoded as 1 for the "yes" values, 0 otherwise

The `x` and `y` dataframes were then split into training and testing sets and the Scikit-learn Logistic Regression model fitted to the training data with the accuracy score calculated. This Logistic Regression model took parameters of `C` and `max_iter` (described above in the summary).

Finally, the script then saved the model to the outputs folder.

The Hyperdrive configuration that I used to carry out the various runs of the experiment for various hyperparameter values was as follows:
```
ps = RandomParameterSampling({
    "--C": uniform(0.001, 1), # Inverse of regularization strength
    "--max_iter": choice(50, 100, 150, 200) # Maximum number of iterations to converge
    })

policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

sklearn_env = Environment.from_conda_specification(name='sklearn-env', file_path='conda_dependencies.yml')

src = ScriptRunConfig(source_directory='.',
                      script='train.py',
                      compute_target=cpu_cluster,
                      environment=sklearn_env)

hyperdrive_config = HyperDriveConfig(run_config=src,
                                     hyperparameter_sampling=ps,
                                     policy=policy,
                                     primary_metric_name='Accuracy',
                                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                     max_total_runs=20,
                                     max_concurrent_runs=4)
```
This used the parameter sampling and early stopping policy I defined, with the primary metric being "Accuracy" and the goal being to maximize it. It was set to run for a maximum of 20 runs, with up to 4 runs taking place at any one time.

#### Parameter sampler
The parameter sampler chosen was `RandomParameterSampling`, which chooses parameter values from a set of discrete values or from a distribution over a continuous range.

As shown above in the pipeline architecture, there were two hyperparameters (`C` and `max_iter`) used for the Scikit-learn Logistic Regression model. As the default number of iterations for this model is 100, I chose four discrete values around this default, from 50 to 200. For the `C` hyperparameter, the inverse of the regularization strength, this must be a positive float, with smaller values specifying a stronger regularization. Because of this, I set this hyperparameter to be chosen from a uniform distribution between 0.001 and 1, so any value within this range could be chosen without any bias.

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

#### Early stopping policy
The early stopping policy that I implemented was a Bandit policy. This is based on slack factor/slack amount and evaluation interval, so it early terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing training run.

https://learn.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py

I set this Bandit policy with an evaluation interval of 2 and a slack factor of 0.1, meaning that after every two iterations it would check to see if it needs to terminate that job early, and if at the point of checking that particular model's accuracy+10% is below the current best model's accuracy, it will then terminate that job. This saves compute resources from being wasted on jobs that are already underperforming.

## AutoML
AutoML allows us to run many different machine learning models without needing to create new pipelines for each one, and to then pick the best performing model automatically. The parameters that I used in setting up the AutoML run were as follows:
```
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=train_data,
    label_column_name='y',
    n_cross_validations=5,
    compute_target=cpu_cluster)
```
This set up AutoML to use classification models (but not utilizing Deep Learning), with accuracy as the primary metric to align with what was done in the Hyperdrive run.

The best AutoML model was a Voting Ensemble, consisting of XGBoost and LightGBM classifiers. The highest weighted of these being estimator 31, an XGBoost classifier, with a weight of 0.429. The parameters of this were:

`XGBoostClassifier(booster='gbtree', colsample_bylevel=0.6, colsample_bytree=1, eta=0.001, gamma=0, max_depth=6, max_leaves=15, n_estimators=800, n_jobs=1, objective='reg:logistic', problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=0, reg_alpha=0, reg_lambda=2.5, subsample=1, tree_method='auto')`

## Pipeline comparison
Best Hyperdrive model: Scikit-learn Logistic Regression, `C` = 0.9054340254976188, `max_iter` = 100, `Accuracy` = 0.9102937606215101  
Best AutoML model: Voting Ensemble, `accuracy` = 0.9174506828528074

Both of these resulted in a similar accuracy value, but AutoML was ever so slightly higher. This is possibly due to how AutoML was able to utilize a variety of machine learning models, whilst the Hyperdrive run was limited to the Scikit-learn Logistic Regression with just the two hyperparameters to vary.

In terms of differences in their architectures, with the Hyperdrive run the train.py script split the data into training and testing sets and fitted the Logistic Regression model using the training set, but with the AutoML run this training and testing split was not carried out explicitly, and instead was carried out by using 5-fold cross-validation (`n_cross_validations=5` within the `automl_config`). This means that there were five different trainings, each training using 4/5 of the data, and each validation using 1/5 of the data with a different holdout fold each time.

https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cross-validation-data-splits

## Future work
As part of the AutoML run, it carries out some Data Guardrails. One of these guardrails resulted in an alert, highlighting an imbalance in the data. Of the two classes in the data, 29258 had a class of 0 whilst only 3692 had a class of 1. This is just 11% being class 1, meaning that there is a lot of bias towards class 0 within the dataset, and even if our prediction just picked class 0 every time, that itself would reach an accuracy of 89% for our dataset. Because of this, a possible improvement for future experiments would be to address this imbalance in the data. For instance, capturing more cases for class 1, or using some method to resample the data. Alternatively, using a different metric to measure the performance of the model, such as Weighted AUC or F1-Score, may be a better measure to use.

Additionally, if compute power and time allowed, more experiments could be run to try to find a better model, such as additional AutoML runs (this may require a longer timeout), and additional options for the hyperparameters in the Hyperdrive run.

## Proof of cluster clean up
The compute cluster was cleaned up within the code of the Jupyter Notebook.
