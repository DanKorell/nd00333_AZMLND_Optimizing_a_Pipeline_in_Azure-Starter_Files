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
This dataset contains data about bank marketing, with a CSV file containing details on 32950 people, with information such as their age, job, marital status, education level, along with some numerical columns. Unfortunately, a description for the dataset has not been made available, only a download link, so no descriptions of these numerical values are present. The final column of the bank marketing CSV is labelled "y" and contains values of 0 and 1. This is the field we want to predict and so this is a classification problem. This field could represent whether or not the bank was successful in marketing some product to this customer and therefore being able to predict these successes would enable the bank to more effectively target customers for the marketing of this product.

Two machine learning solutions were developed. The first being a Scikit-learn logistic regression model, for which Hyperdrive was used in the Azure ML SDK to run it multiple times with different values for the two hyperparameters (C: inverse of regularization strength, max_iter: the maximum number of iterations to converge). The second being an AutoML solution running in the Azure ML SDK. With accuracy used as the primary metric, the best Hyperdrive run resulted in 91.029% accuracy, whilst the best AutoML model resulted in 91.745% accuracy. This AutoML model was a Voting Ensemble, consisting of XGBoost and LightGBM classifiers.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

**What are the benefits of the parameter sampler you chose?**

**What are the benefits of the early stopping policy you chose?**

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

This set up a classification model (but not utilizing Deep Learning), with accuracy as the primary metric, to align with what was done in the Hyperdrive run.

The best AutoML model was a Voting Ensemble, consisting of XGBoost and LightGBM classifiers. The highest weighted of these being estimator 31, an XGBoost classifier, with a weight of 0.429. The parameters of this were:

`XGBoostClassifier(booster='gbtree', colsample_bylevel=0.6, colsample_bytree=1, eta=0.001, gamma=0, max_depth=6, max_leaves=15, n_estimators=800, n_jobs=1, objective='reg:logistic', problem_info=ProblemInfo(gpu_training_param_dict={'processing_unit_type': 'cpu'}), random_state=0, reg_alpha=0, reg_lambda=2.5, subsample=1, tree_method='auto')`

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
The compute cluster was cleaned up within the code of the Jupyter Notebook.
