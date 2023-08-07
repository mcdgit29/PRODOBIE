import numpy as np
from scipy.stats import wilcoxon, ranksums
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, Birch, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.optimize import minimize_scalar
from sklearn.datasets import make_blobs
from tqdm import tqdm
from ReduceClusterEval.setup_logger import logger
import logging


def get_subgroup_names(X, groups, metric=ranksums, alpha=0.01, prefix='group:'):
    return list(_subgroup_name_gen(X, groups, metric, alpha, prefix))


def _subgroup_name_gen(X, groups, metric=ranksums, alpha=0.01, prefix=''):
    if isinstance(X, type(np.array([0]))):
        X = pd.DataFrame(X)
    elif isinstance(X, type(pd.DataFrame([]))):
        pass
    else:
        raise ValueError(F'unknown data type {type(X)} use numpy array or pandas dataframe')
    colnames = np.array(X.columns)
    group_set = np.unique(groups)
    for group in group_set:
        X_subset = X.loc[groups == group]
        X_remaining = X.loc[groups != group]
        vals = []
        index = []
        for i, col in enumerate(colnames):
            stat, p  = ranksums(X_subset.loc[:, col], X_remaining.loc[:, col])
            if p <= alpha:
                vals.append(stat)
                index.append(colnames[i])
        results = pd.Series(np.array(vals).astype(np.float64), index=index).sort_values()
        name =prefix + str(group) +":" + \
         'lower: (' + ','.join([v for v in results.loc[results < 0 ].index.astype(str)]) +  \
         ') higher: (' + ','.join([v for v in results.loc[results > 0 ].index.astype(str)]) + ')'
        yield name


class OptimalClusters:
    '''
    A class for detecting the optimal number of clusters using Silhouette Methods
    finds cluster my maximixing the silhouette score.
    '''
    def __init__(self, cluster_range=(2, 20), method=Birch, random_state=2022, maxiter=5, xatol=.001, n_passes=3,
                 new_pass_bounds=(-2,2),
                 **kwargs):
        '''
        param cluster_range: tuple of type  (int, int) range of number of clusters to search
        param method: sklearn.clearn method
        param random_state: int
        param maxiter:int max number of interations on the cluster search
        param xatol: float optiziation toterlance
        param n_passes: int number of optimization passes
        '''

        self.cluster_range = cluster_range
        self.method = method
        self.kwargs = kwargs
        self.scores = None
        self.X = None
        self.results = {}
        self.random_state = random_state
        self.optimize_obj = None
        self.best_number_of_clusters = None
        self.maxiter=maxiter
        self.xatol=xatol
        self.n_passes=n_passes
        self.new_pass_bounds = new_pass_bounds
        np.random.random(random_state)

    def silhouette_score_n_clusters(self, n):
        '''
        param n int number of clusters to calculate silhouette score
        returns: float 1-silhouette score
        '''

        n = np.round(n, 0)
        logger.debug(F'clustering method {type(self.method)} with {n} centers')
        model = self.method(n_clusters=int(n))
        model = model.fit(self.X)
        labels = model.predict(self.X)
        score = silhouette_score(self.X, labels)
        self.results.update({n: score})
        return 1-score

    def _optimize_n_clusters_search(self):
        '''
        sets optimize_obj object, runs an optimization search on the number of clusters
        returns: None
        '''

        n_passes = self.n_passes
        bounds = self.new_pass_bounds
        logger.debug('inital optimzation pass')
        self.optimize_obj = minimize_scalar(self.silhouette_score_n_clusters, method='bounded',
                                            bounds=self.cluster_range, options = {'maxiter': self.maxiter,
                                                                                  'xatol':self.xatol})
        for i in range(1, n_passes):
            logger.debug(F" additional optimzation pass {i} of {n_passes}")
            best_n = np.round(self.optimize_obj.x,0)
            new_range = np.max([best_n + bounds[0], 2]), best_n + bounds[1]
            self.optimize_obj = minimize_scalar(self.silhouette_score_n_clusters, method='bounded',
                                                bounds=new_range , options={'maxiter': self.maxiter,
                                                                                    'xatol': self.xatol})

    def fit(self, X):
        '''
        method for running optimization (fits multiple cluster models to find best silhouette score
        param X: numpy array of features
        returns: self
        '''

        self.X = X
        self._optimize_n_clusters_search()
        self.best_number_of_clusters = int(self.optimize_obj.x)
        return self

    def get_best_number_of_clusters(self):
        '''
        after fit method is run, returns the best number of clusters

        returns: int optimal number of clusters
        '''

        if self.best_number_of_clusters:
            return self.best_number_of_clusters
        else:
            raise ValueError('OptimalClusters is Not Fitted')


    def get_scores(self):
        '''
        returns: Pandas Series of silhouette scores with index number clusters tested
        '''
        results = pd.Series(self.results).sort_values()
        results.index = results.index.astype(int)
        return results


def find_optimal_clusters_size(X, **kwargs):
    '''
    param X: numpy array of features
    returns: Pandas Series of silhouette scores with index number clusters tested

    Additional kwargs:

        param cluster_range: tuple of type  (int, int) range of number of clusters to search
        param method: sklearn.clearn method
        param random_state: int
        param maxiter:int max number of interations on the cluster search
        param xatol: float optiziation toterlance
        param n_passes: int number of optimization passes

    '''

    opt = OptimalClusters(**kwargs)
    opt.fit(X)
    return opt.get_scores()



