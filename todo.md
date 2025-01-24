# TODO

## Milestone 1: Get data

* [X] Get a scope of which data can be tracked

  * [X] Try out with 10 different api requests
  * [X] Save the outputs in `data/raw/`
* [X] KDE data augmentation

  * [ ] Create an script for data augmentation -> take care of variables distribution
  * [X] Test the script into a notebook and save the data (I would like to have ~50k datapoints)
  * [X] Make a simple sketch of how a data transformation pipeline would work, let's try out with figma
* [X] Mount data in Databricks

  * [X] Mount data
  * [X] Try to manage using spark
  * [X] Connect MLFLOW
* [X] Experiment with MLFLOW

  * [X] Focus on linear models and feature engineering
  * [X] Try to achieve good r2 and rmse metrics
  * [X] Visualize model performance results using cool plots
  * [X] How to reproduce experiments?
* [ ] Learn DataBricks

  * [ ] Interact with databricks
* [ ] Connect repo to aws

# Milestone 2: MLFlow Experimentation

* [ ] Try out with XGB, LGBM

# Milestone 3: Automation

* [X] Automate Data Ingestion
* [ ] Automate Data Preprocessing
  * [X] Cleaning + Gaussian Noise
  * [ ] Data Augmentation
* [ ] Automate ML Modeling
  * [ ] Train
  * [ ] Inferences
