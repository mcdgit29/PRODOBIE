import numpy as np
import pandas  as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.model_selection import KFold
from sklearn.datasets import load_diabetes
from scipy.stats import wilcoxon, entropy, gamma
from scipy.special import rel_entr
from plotly import express as px
from ReduceClusterEval.setup_logger import logger
from ReduceClusterEval.metrics import get_regression_metrics, get_binary_classification_metrics
from ReduceClusterEval.utils import get_subgroup_names, find_optimal_clusters_size
import plotly.graph_objects as go



class RCE:
    '''
    Recuce Cluster Evaluate Class



    '''
    def __init__(self, metric='reg', cluster_range=(2, 10), random_state=20, method=MiniBatchKMeans, **kwargs):
        '''
        param metric: str reg or class for regression of binary classification evaluation
        param cluster_range: tuple(int, int) range in witch to search clusters
        param random_state: int randomization state
        param method: Sklearn cluster class method
        param kwargs: key word arguments for the cluster method object

        '''
        self.binary_classification = False
        if str(metric).lower().__contains__('reg'):
            self.metrics = get_regression_metrics()
        elif str(metric).lower().__contains__('clas'):
            self.metrics = get_binary_classification_metrics()
            self.binary_classification = True
        else:
            raise ValueError(F'unknown metric {metric}, use classification or regression')
        self.cluster_names = None
        self.cluster_model = None
        self.decomp_model = FastICA(2,  whiten='unit-variance', random_state=random_state)
        self.scaler_model = StandardScaler()
        self.input_features = None
        self.cluster_range = cluster_range
        self.n_clusters = None
        self.cluster_scores = None
        self.set_cluster_model(method, **kwargs)

    def set_cluster_model(self, method=Birch, **kwargs):
        self.cluster_method = method
        self.cluster_kwargs = kwargs

    def _set_discrete_color_dict(self):

        vals = np.arange(self.n_clusters+1)
        self.discrete_colors = dict(zip(vals, px.colors.qualitative.T10*self.n_clusters))
        self.discrete_colors.update(dict(zip([str(v) for v in vals], px.colors.qualitative.T10*self.n_clusters)))

    def _get_discrete_color_dict(self):
        return self.discrete_colors

    def fit_cluster_model(self, X):
        if self.n_clusters is None:
            self.cluster_scores = find_optimal_clusters_size(X, cluster_range=self.cluster_range,  method=self.cluster_method)
            self.n_clusters = int(self.cluster_scores.index[-1])
            logger.info(F'optimal number of clusters detected {self.n_clusters}')
        else:
            logger.debug(F'using {self.n_clusters} number of clusters')
        self.cluster_model = self.cluster_method(n_clusters=self.n_clusters, **self.cluster_kwargs)
        self.cluster_model = self.cluster_model.fit(X)
        self._set_discrete_color_dict()

    def set_pca_model(self, model, **kwargs):
        self.decomp_model = model(**kwargs)

    def set_scaler_model(self, model, **kwargs):
        self.scaler_model = model(**kwargs)

    def get_cluster_names(self):
        return np.array(self.cluster_names)

    def fit(self, X, y=None):
        '''
        param X: data frame or 2d array
        param y: unuses (for sklearn compatability
        returns self
        '''
        logger.debug('fitting pipeline ...')
        self.scaler_model= self.scaler_model.fit(X)
        X_scaled = self.scaler_model.transform(X)
        self.fit_cluster_model(X_scaled)
        self.decomp_model = self.decomp_model.fit(X_scaled)
        X_pca = self.decomp_model.transform(X_scaled)

        distances = self.cluster_model.transform(X_scaled)
        self.distances_mean = np.min(distances, axis=1).mean()
        self.distances_std = np.min(distances, axis=1).std()
        labels =self.predict(X)
        logger.debug('fitting cluster names ...')
        self.cluster_names = get_subgroup_names(X,groups=labels, prefix='cluster')
        logger.debug('fitting discrete colors to cluster names ...')
        self._set_discrete_color_dict()
        return self

    def get_outlier_scores(self, X, dist_fun=gamma):
        '''
        param X: data frame or 2d array
        param dist_fun: distance function
        returns array of distances to the center of each cluster
        '''
        results = pd.Series(X.min(axis=1))\
        .apply(lambda x: dist_fun(a=x, scale=self.distances_std, loc=self.distances_mean ))
        return results.values

    def transform(self, X, y=None):
        '''
        param X: data frame or 2d array
        param y: unuses (for sklearn compatability
        return np.array sklearn cluster model transform results
        '''
        X_scaled = self.scaler_model.transform(X)
        return self.cluster_model.transform(X_scaled)

    def predict(self, X, y=None):
        '''
        param X: data frame or 2d array
        param y: unuses (for sklearn compatability
        return np.array sklearn cluster model predict results
        '''

        X_scaled = self.scaler_model.transform(X)
        return self.cluster_model.predict(X_scaled)

    def _evaluate_gen(self, X, y, p, c=None, threshold=None):
        '''
        internal evaluation method
        param X: data frame or 2d array of features
        param y: numpy array of labels
        param p: numpy array of predictions
        param c: numpy array of clusters (int)
        param threshold: float in  (0,1) prediction threshold

        '''


        if isinstance(c, type(None)):
            logger.debug('using clusters from fit cluster model')
            labels = self.predict(X)
        else:
            logger.debug('using clusters from input args for evaluation')
            labels = c
        logger.debug(F'F evaluate gen found labels {pd.Series(labels).value_counts()}')

        results_raw = pd.DataFrame({'actual':y, 'pred':p })
        if threshold is None:
            threshold=y.mean()
        else:
            pass
        results_raw.loc[:, 'pred_label'] = results_raw.pred.apply(lambda x: x > threshold).replace({True:1, False:0})
        for label in np.unique(labels):
            result_subset = results_raw.loc[labels==label, :]
            a_subset =result_subset.actual
            p_subset = result_subset.pred
            d = {}
            try:
                d['key'] = self.pipline.steps[-1][1].get_feature_names_out(label)
            except AttributeError:
                d['key'] = label
            d['weight'] = result_subset.shape[0]
            for key, metric in self.metrics.items():
                try:
                    d[key] = metric(a_subset, p_subset)
                except ValueError:
                    try:
                        p_label_subset = result_subset.pred_label
                        d[key] = metric(a_subset, p_label_subset)
                    except KeyError:
                        pass
                    except AttributeError:
                        pass
            stat, pval = wilcoxon(a_subset, p_subset)
            d['wilcoxon_signed_rank_stat'] = stat
            d['wilcoxon_signed_rank_pval'] = pval
            yield d

    def evaluate(self, X, y, p, c=None, threshold=None):

        '''
        Evaluation method (performance on each cluster
        param X: data frame or 2d array of features
        param y: numpy array of labels
        param p: numpy array of predictions
        param c: numpy array of clusters (int)
        param threshold: float in  (0,1) prediction threshold
        returns data frame
        '''
        results = pd.DataFrame(list(self._evaluate_gen(X, y, p, c, threshold=threshold)))
        results.loc[:, 'name'] = self.get_cluster_names()
        results = results.set_index('key')
        return results


    def get_clusters_silhouette_score(self, X):

        '''
        Cluster quality Evaluation method using silhouette score
        param X: data frame or 2d array of features\
        returns array
        '''
        distances = self.transform(X)
        labels = self.predict(X)
        return silhouette_score(X, labels)

    def get_labeled_tsne(self, X, y=None, p=None, c=None):
        '''
        Cluster quality Evaluation method using silhouette score
        param X: data frame or 2d array of features
        param y: numpy array of labels
        param p: numpy array of predictions
        param c:numpy array clusters (optional) if un used, method will call internal cluster model
        returns data frame
        '''

        X_scaled = self.scaler_model.transform(X)
        if isinstance(c, type(None)):
            labels = self.cluster_model.predict(X_scaled)
        else:
            labels = c
        results = pd.DataFrame(self.decomp_model.transform(X_scaled))
        results.columns = ['comp_' + str(i) for i in range(self.decomp_model.n_components)]
        results.loc[:, 'cluster_labels'] = labels
        results.loc[:, 'actuals'] = y
        results.loc[:, 'predictions'] = p
        return results


    def get_silouette_samples(self, X, c=None):
        '''
        Cluster quality Evaluation method using silhouette samples
        param X: data frame or 2d array of features\
        param c:numpy array clusters (optional) if un used, method will call internal cluster model
        returns array
        '''

        X_scaled = self.scaler_model.transform(X)
        if isinstance(c, type(None)):
            labels = self.cluster_model.predict(X_scaled)
        else:
            labels = c

        label_names = self.get_cluster_names()[labels]
        results = pd.DataFrame({'scores': silhouette_samples(  X_scaled , label_names)})
        results.index = label_names
        results.loc[:, 'cluster'] = labels
        return results.sort_values(by='scores')

    def plot_scatter(self, X, y, p, c=None,  metric=mean_absolute_error, metric_name='mae'):

        '''
        Scatter Plot
        param X: data frame or 2d array of features
        param y: numpy array of labels
        param p: numpy array of predictions
        param c:numpy array clusters (optional) if un used, method will call internal cluster model
        returns data frame
        '''

        df = self.get_labeled_tsne(X, y, p, c)
        df.cluster_labels = df.cluster_labels.astype(str)

        df.loc[:, metric_name] = metric(y, p)
        range_color = np.percentile(df.loc[:,  metric_name], [5, 95]).astype(int)
        fig = px.scatter(df,
        x='comp_0',
        y='comp_1',
        color='cluster_labels',
        hover_data=df.columns,
        range_color=range_color,
        size=metric_name,
        color_discrete_sequence=self._get_discrete_color_dict()
        )

        return fig

    def plot_cluster_silouette_samples(self, X):
        df = self.get_silouette_samples(X)
        df.index.name = 'cluster_name'
        df =df.reset_index()
        s = self.get_clusters_silhouette_score(X)
        df = df.sort_values(by=['cluster_name', 'scores']).reset_index()
        df.loc[:, 'cluster_label'] = df.loc[:, 'cluster'].astype(str)
        fig = px.bar(df,
        x = 'scores',
        color = 'cluster_label',
        hover_data = df.columns,
        color_discrete_sequence = self._get_discrete_color_dict())

        fig.update_layout(xaxis_title="Silouette Score", yaxis_title="Observation")
        fig.update_layout(title_text=F'Cluster Quality Silouette Analysis <br>  Ave Score: {np.round(s, 3)}')
        return fig

    def plot_performance_table(self, X, y, p, c=None):

        '''
        Scatter Plot
        param X: data frame or 2d array of features
        param y: numpy array of labels
        param p: numpy array of predictions
        param c:numpy array clusters (optional) if un used, method will call internal cluster model
        returns fig
        '''

        results = self.evaluate( X, y, p, c).round(3)
        results.index.name = 'cluster'
        results = results.transpose().reset_index()

        fig = go.Figure(data=[go.Table(
        header = dict(values=list(results.columns),
                    align='left'),
        cells = dict(values=[results.loc[:, col] for col in results.columns],
                   align='left'))])
        return fig

    def plot_performance_scatter(self, X,y, p, c=None):
        '''
        Scatter Plot performance of each cluster
        param X: data frame or 2d array of features
        param y: numpy array of labels
        param p: numpy array of predictions
        param c:numpy array clusters (optional) if un used, method will call internal cluster model
        returns fig
        '''
        X_scaled = self.scaler_model.transform(X)
        if isinstance(c, type(None)):
            labels = self.cluster_model.predict(X_scaled)
        else:
            labels = c
        to_plot = pd.DataFrame({"cluster": pd.Series(labels).astype(str), "actuals":y, "predictions":p })
        return px.scatter(to_plot,
        x = 'predictions',
        y = 'actuals',
        color = 'cluster',
        title = "Predicted vs Actual Values by Cluster",
        color_discrete_sequence = self._get_discrete_color_dict())

