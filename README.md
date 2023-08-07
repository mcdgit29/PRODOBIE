# 4.	Pro-DOBIE: A Protocol for Domain and Outcome Bias Exploration in Healthcare Machine Learning Models
### Objective
The goal of this work is to demonstrate the usage and value of a novel protocol to enable researchers to detect and describe bias

### Methods

Pro-DOBIE was applied to models developed on two publicly available datasets and a novel data set from Medical University of South Carolina (MUSC) describing mammogram screening follow through. 

The protocol has three calls for three types of analysis:

+ (i) subgroup performance equitability; 
+ (ii) application and training data similarity; and 
+ (iii) training data grouping variable congruence.

### Applying The Protocol
This specific application uses the following methods to accomplish the three part analysis:

+  Subgroup performance equitability. This was accomplished with micro and macro average performance across groups
+  Application and training data similarity.  This was accomplished with PCA inversion analysis across subgroups
+  Training data grouping variable congruence. This was accomplished using silhouette score analysis

## Notebooks
The following jupyter notebooks demonstrate the Pro-DOBIE Protocol on different datasets
These notebooks may be adapted for usage on novel data. 

+ [Classification with Mamogram Screening Update.ipynb](notebooks%2FClassification%20with%20Mamogram%20Screening%20Update.ipynb)
+ [Classification with Wisconson Breast Cancer Data.ipynb](notebooks%2FClassification%20with%20Wisconson%20Breast%20Cancer%20Data.ipynb)
+ [Classification with Heart Failure.ipynb](notebooks%2FClassification%20with%20Heart%20Failure.ipynb)


## Domain Explorer
A python package to explore datasets with used for training and inference on supervised predictive tasks.  This
helps visualize the primincials of Pro-DOBIE on new data sets 


## Reduce Clustered Evaluation Tool
A python package to visualize feature domain spaces, predictions and outcomes


### Usage from Python3

This package may be used directly on pandas data frames that contain columns

  + 'y' label column (0,1) for binary classification
  + 'p' column of numeric probablities
  + 'c' optional column integer of cluster labels, 
      if not provided cluster_range will be search to find optimial number of clusters

Parameters for initializing RCE Class are all follows

        param metric: str reg or class for regression of binary classification evaluation
        param cluster_range: tuple(int, int) range in witch to search clusters
        param random_state: int randomization state
        param method: Sklearn cluster class method
        param kwargs: key word arguments for the cluster method object



```python3
import numpy as np
from sklearn.datasets import make_classification
from ReduceClusterEval.transformers import RCE
X, y = make_classification(1000, 10, n_informative=8, n_classes=2)
p = y + np.random.normal(np.mean(y), np.std(y), y.shape)
rce = RCE(metric='class', cluster_range=(3,10))
rce.fit(X)

## Core evaluation method
results = rce.evaluate(X,y,p)
### Plotting Methods
s = rce.get_silouette_samples(X)
fig = rce.plot_cluster_silouette_samples(X)
fig = rce.plot_scatter(X, y, p)
fig = rce.plot_performance_scatter(X, y, p)
fig = rce.plot_performance_table(X, y, p)
```


#### Command Line Usage
This is an example of using a data.csv file, and running the RCE Method from the command line.  This
method requires the following columns with headers in the csv and to be save without pandas index :
  + 'y' label column (0,1) for binary classification
  + 'p' column of numeric probablities
  + 'c' optional column integer of cluster labels, 
      if not provided cluster_range will be search to find optimial number of clusters
  

```shell
python3 ReduceClusterEval -f data.csv -metric_type class -n_clusters=5'
```
#### Installation Guide
Install python package from git 
It is strongly recommended to install into a virtual environment

```shell
git clone https://git.musc.edu/mad221/reduceclustereval.git
cd into reduceclustereva directory
python3 -m pip install -r requirments.txt
python3 -m pip install . 
```

#### Post Installation Testing 
```shell
pytest test_reducecluster_eval.py
```

