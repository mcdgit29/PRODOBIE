
import logging
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from ReduceClusterEval.setup_logger import logger
from ReduceClusterEval.transformers import RCE
from ReduceClusterEval.utils import get_subgroup_names
from ReduceClusterEval.utils import  OptimalClusters, find_optimal_clusters_size
from sklearn.cluster import Birch
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import tempfile
import os
logger.info('running ReduceClusterEval test ...')


def test_RCE_class():
    logger.info('Testing Reggresson Methods')
    X, y = make_classification(1000, 10, n_informative=8, n_classes=2)
    p = y + np.random.normal(np.mean(y), np.std(y), y.shape)
    rce = RCE()
    rce.fit(X)
    logger.debug(F'cluster colors : {rce._get_discrete_color_dict()}')
    labels = np.argmin(rce.transform(X), axis=1)
    assert len(labels) == X.shape[0]
    logger.info(F' sillouette score {rce.get_clusters_silhouette_score(X)}')
    rce.get_labeled_tsne(X)
    rce.get_outlier_scores(X)
    s = rce.get_silouette_samples(X)
    fig = rce.plot_cluster_silouette_samples(X)

    fig = rce.plot_scatter(X, y, p)

    fig = rce.plot_performance_scatter(X, y, p)

    fig = rce.plot_performance_table(X, y, p)

    logger.debug('testing classification')
    y = np.random.choice([0, 1], size=X.shape[0])
    p = np.random.choice([0, .1, .2, .4, .5, .7, .9, 1], X.shape[0])
    rce = RCE(metric='class').fit(X)
    distances = rce.transform(X, y)
    labels = rce.predict(X, y).flatten()
    rce.evaluate(X, y, p, c=labels)
    logger.debug("evaluating using provided clusters")
    logger.info('rce test competed')


def test_RCE_reg():
    logger.info('Testing Classification Methods')
    X, y = make_regression(1000, 10,  n_informative=8)
    p = y + np.random.normal(np.mean(y), np.std(y), y.shape)
    rce = RCE(metric='reg')
    rce.fit(X)
    logger.debug(F'cluster colors : {rce._get_discrete_color_dict()}')
    labels = np.argmin(rce.transform(X), axis=1)
    assert len(labels) == X.shape[0]
    logger.info(F' sillouette score {rce.get_clusters_silhouette_score(X)}')
    rce.get_labeled_tsne(X)
    rce.get_outlier_scores(X)
    s = rce.get_silouette_samples(X)
    fig = rce.plot_cluster_silouette_samples(X)

    fig = rce.plot_scatter(X, y, p)

    fig = rce.plot_performance_scatter(X, y, p)

    fig = rce.plot_performance_table(X, y, p)

    logger.debug('testing classification')
    y = np.random.choice([0, 1], size=X.shape[0])
    p = np.random.choice([0, .1, .2, .4, .5, .7, .9, 1], X.shape[0])
    rce = RCE(metric='class').fit(X)
    distances = rce.transform(X, y)
    labels = rce.predict(X, y).flatten()
    rce.evaluate(X,y,p, c=labels)
    logger.debug("evaluating using provided clusters")
    logger.info('rce test competed')


def test_subgroupnames():
    logger.info('testing utils module ...')
    np.random.seed(202)
    X = np.random.choice([0, 1], (100, 10))
    groups = np.random.choice(np.arange(8), 100)
    names = get_subgroup_names(X, groups, alpha=.2)
    names = get_subgroup_names(pd.DataFrame(X, columns=['col' + str(i) for i in range(10)]), groups, alpha=.2)
    assert len(names) == 8
    assert len(np.unique(names)) == 8
    logger.info('utils module testing completed')


def test_OptimalClusters():
    n_clusters = 10
    n_features = 5
    n_obs = 1000
    X, y = make_blobs(n_samples=n_obs, n_features=n_features, centers=n_clusters)
    opt = OptimalClusters(method=Birch)

    opt = opt.fit(X)
    print(find_optimal_clusters_size(X))


def test_main():
    logger.info('testing main in classification mode ...')
    X, y = make_classification(1000, 10, n_classes=2, n_informative=8)
    model = LogisticRegression().fit(X, y)
    p = model.predict_proba(X)[:, 1]
    df = pd.DataFrame(X)
    df.loc[:, 'p'] = p
    df.loc[:, 'y'] = y

    with tempfile.NamedTemporaryFile()as f:
        path = f.name
        df.to_csv(path, index=False)
        cmd = F'python3 ReduceClusterEval -f {path} -metric_type class -n_clusters=5'
        logger.info(F'running cmd {cmd}')
        #os.system(cmd)

    logger.info('main testing completed')


if __name__ == "__main__":
    logger.setLevel('INFO')
    logger.info('testing subgroup names ...')
    test_subgroupnames()
    logger.info('testing optimal clusters ...')
    test_OptimalClusters()
    logger.info('testing RCE Class ...')
    test_RCE_class()
    logger.info('testing RCE Reg ... ')
    test_RCE_reg()
    test_main()
    logger.info('All Tests Completed')