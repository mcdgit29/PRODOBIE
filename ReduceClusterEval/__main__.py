import base64
import datetime
import io
import numpy as np
import plotly.graph_objs as go
from plotly import express as px
from dash import Dash, html, Input, Output, dash_table, dcc, State
import argparse
import pandas as pd
import sys
from ReduceClusterEval.transformers import RCE
import logging
from ReduceClusterEval.setup_logger import logger

parser = argparse.ArgumentParser(description='Arguments for Reduce Cluster Evaluate App (requires a -f filepath)')
parser.add_argument('-f', nargs="?",  help="File path to comma separated csv to be loaded with pandas \n \
must contain prediction_col_name: 'p',target_col_name: 'y', and features,  ")
parser.add_argument('-n_clusters', type=int, default=-1,  help='int number of kmeans clusters')
parser.add_argument('-n_comps', type=int, default=5,   help='int number of Principal Components')
parser.add_argument('-metric_type', type=str, default="reg",  help='type of metrics, either  "reg", or "class", defaulted to "reg"')
parser.add_argument('-prediction_col_name', type=str, default="p",  help='name of the prediction column')
parser.add_argument('-target_col_name', type=str,  default="y",  help='name of the target column')
parser.add_argument('-n_samples', type=int, default=20000,  help='The number of samples without replacement to take from csv')
parser.add_argument('-seed', type=int, default=20,  help='random seed')
args = parser.parse_args()

filename = args.f
n_clusters = int(args.n_clusters)
n_comps = int(args.n_comps)
metric_type = args.metric_type
target_col_name = args.target_col_name
prediction_col_name = args.prediction_col_name
n_samples = args.n_samples
seed = args.seed
np.random.seed(seed)

logger.info(F'loading file {filename}')
logger.info(F'n_clusters: {n_clusters }')
logger.info(F'n_comps: {n_comps }')
logger.info(F'metric_type: {metric_type}')
logger.info(F'target_col_name: {target_col_name}')
logger.info(F'prediction_col_name: {prediction_col_name}')

df = pd.read_csv(filename)
logger.info(F' csv loaded from : "{filename}" with shape {df.shape}')
if df.shape[0] > n_samples:
    logger.warn(F'Down sampling data to size {n_samples} without replacement')
    df = df.sample(n_samples, replace=False)
X = df.drop([target_col_name, prediction_col_name], axis=1)
p = df.loc[:, prediction_col_name]
y = df.loc[:, target_col_name]

logger.info(F'Using targeted (outcome) column: {target_col_name}  as target mean  {y.mean().round(2)}')
logger.info(F'Using prediction column: {prediction_col_name}  with mean  {p.mean().round(2)}')

## Data validity check
if y.isna().sum() > 0:
    logger.error(F' {target_col_name} contains null values')

if pd.Series(X.values.flatten()).isna().sum() > 0:
    logger.error(F' features contains null values')

if p.isna().sum() > 0:
    logger.error(F'  {prediction_col_name} contains null values')


assert X.shape[0] == y.shape[0]
assert y.shape[0] == p.shape[0]

## Fits  A Cluster Model ...
logger.debug('fitting rce ...')
if n_clusters == -1:
    rce = RCE(random_state=seed, metric=metric_type)
else:
    rce = RCE(cluster_range=(n_clusters,n_clusters), random_state=seed, metric=metric_type)
rce.fit(X=X)
labels = rce.predict(X)
df_w_clusters = df
df_w_clusters.loc[:, '--cluster'] = pd.Series(labels).astype(str)

## create plots
tsne_fig = rce.plot_scatter(X, y, p)
tsne_fig .update_layout(
    autosize=False,
    width=800,
    height=800)
performance_fig = rce.plot_performance_table(X, y, p)
performance_fig.update_layout(
    autosize=False,
    width=1800,
    height=800)

sillouette_fig = rce.plot_cluster_silouette_samples(X)
sillouette_fig.update_layout(
    autosize=False,
    width=800,
    height=800)


app = Dash(__name__)
colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

app.layout = html.Div([
    dcc.Markdown("""
        # Reduce Cluster Evaluate
        Predictive Model Analysis tool
        PCA => Clustering => Performance analysis of clusters
        """),
    dcc.Dropdown(df_w_clusters.columns, id='label_col', placeholder="Select label Column"),
    dcc.Dropdown(df_w_clusters.columns, id='prediction_col', placeholder="Select Prediction Column"),
    dcc.Dropdown(["Build A Cluster Model"] + list(df_w_clusters.columns), id='group', placeholder="Select Grouping Column"),
    dcc.Dropdown(['Regression', 'BinaryClassification'], id='metric', placeholder='Select Metric Type'),
    dcc.Input(id="n_clusters", type="text", placeholder="Enter Number of CLusters", debounce=True),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='params'),

    dcc.Graph(figure=performance_fig),
    html.H4(F'Feature Exploration, Select X and Y Axis'),
    dcc.Dropdown(df_w_clusters.columns, id='scatt_x', placeholder="Select x axis"),
    dcc.Dropdown(df_w_clusters.columns, id='scatt_y', placeholder="Select y axis"),
    dcc.Graph(id='scatter_fig'),

    html.H4(F'TSNE with 2 Components reduced from {n_comps} Principal Components'),
    dcc.Graph(figure=tsne_fig ),
    html.H4(F'Clustering Quality Analysis from {n_clusters} clusters of {n_comps} Principal Components'),
    dcc.Graph(figure=sillouette_fig ),
    html.H4(F'Cluster Performance'),


])

@app.callback(
    Output('params', 'children'),
    Input('label_col', "value"),
    Input('prediction_col',"value"),
    Input('group', "value"),
    Input('metric',"value"),
    Input('n_clusters', "value"),
    Input('submit-val',"value")
)
def update_params(label_col, prediction_col,group, metric, n_clusters, submit ):
    params = (label_col, prediction_col,group, metric, n_clusters, submit)
    return str(params)

@app.callback(
    Output('scatter_fig', 'figure'),
    Input('scatt_x', 'value'),
    Input('scatt_y', 'value'))
def update_scatter(scatt_x, scatt_y):
    fig = px.scatter(df_w_clusters, x=scatt_x, y=scatt_y, color=target_col_name, facet_col='--cluster', facet_col_wrap=5)
    fig.update_layout(
        autosize=False,
        width=1600,
        height=800)
    return fig

def main():
    app.run_server(debug=True)

if __name__ == '__main__':
    main()
