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
import logging
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
import logging
from pandas_profiling import ProfileReport
try:
    logger.debug('logger is up')
except:
# base logger setup, to standardize logging across classes
    name = 'DomainExplorer'
    formatter = logging.Formatter(fmt='%(asctime)s -  %(name)s - %(levelname)s  - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    logger.addHandler(handler)


def get_ohe_cols(data):
    results = []
    for c in data.select_dtypes(include=np.number).columns:
        x = data.loc[:, c]
        if all((x.max() == 1, x.min() == 0, len(np.unique(x))==2)):
            results.append(c)
        else:
            pass
    return results


def get_cont_columns_within_range(data, r=(0, 1), mip_n_unique=2):
    results = []
    for c in data.select_dtypes(include=np.number).columns:
        x = data.loc[:, c]
        if all((x.max() <= r[1],
                x.min() >= r[0],
                len(np.unique(x)) >= mip_n_unique)):
            results.append(c)
        else:
            pass
    return results


def get_acc(y_true, y_pred):
    if all((y_true==1, y_pred==1)):
        return 'TP'
    elif all((y_true == 1, y_pred == 0)):
        return 'FN'
    elif all((y_true == 0, y_pred == 1)):
        return 'FP'
    elif all((y_true == 0, y_pred == 0)):
        return 'TN'
    else:
        raise ValueError(F'unknown input y_true: {y_true}  y_pred: {y_pred}')

symbol_map = {'TP':'square', 'FN':'square-x', 'TN': 'circle', 'FP': 'circle-x'}


parser = argparse.ArgumentParser(description='Arguments for Reduce Cluster Evaluate App (requires a -f filepath)')
parser.add_argument('-f', nargs="?",  help="File path to comma separated csv to be loaded with pandas \n \
must contain prediction_col_name: 'p',target_col_name: 'y', and features,  ")
args = parser.parse_args()

filename = args.f

logger.info(F'loading file {filename}')



df = pd.read_csv(filename)
profile = ProfileReport(df, minimal=True)
profile_html = profile.to_html()

random_state = 2022
scaler = StandardScaler()
dcomp = FastICA(2,  whiten='unit-variance', random_state=random_state)

numeric_cols = df.select_dtypes(include=np.number).columns
ohe_cols = get_ohe_cols(df)
cont_in_0_1_cols = get_cont_columns_within_range(df, (0, 1), 2)

app = Dash(__name__)
colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

app.layout = html.Div([
    dcc.Markdown("""
        # Domain Explorer for Binary Classification 
        Predictive Model Analysis tool
        PCA => Clustering => Performance analysis of clusters
        """),
    html.Iframe(profile_html),
    html.Div(children=[html.H4(F'Select the Dependent Label (true value) column:  '),
    dcc.Dropdown(list(df.columns),list(df.columns)[-1], id='label_col', placeholder="Select label Column"),
                       ],style={'padding': 10, 'flex': 1}),

    html.Div(children=[
        html.H4(F'Select the predicted value column:  '),
        dcc.Dropdown(list(df.columns), list(df.columns)[-2], id='prediction_col', placeholder="Select Prediction Column"),
        ],style={'padding': 10, 'flex': 1},),

    html.Div(children=[ html.H4(F'Select the grouping column:  '),
                        dcc.Dropdown(list(df.columns), 'group',  id='group', placeholder="Select Grouping Column"),
                        ], style={'padding': 10, 'flex': 1},),

    html.H4(F'Check Feature Columns:'),
    dcc.Input(id='threshold', type='number', placeholder="Prediction Threshold between 0 and 1"),

    dcc.Checklist(list(numeric_cols), numeric_cols[:-2], id='features'),
    html.H4(F'Select Threshold'),
    html.Button('Submit', id='submit-val'),

    dcc.Graph(id='overall_performance_table'),
    dcc.Graph(id='performance_table'),
    dcc.Graph(id='silhouette'),
    dcc.Graph('dcomp_scatter'),
    dcc.Graph('performance_box'),
    dcc.Graph('roc_auc'),
    dcc.Graph('sun'),
    html.H4(F'Feature Exploration, Select X and Y Axis'),
    dcc.Dropdown(numeric_cols,  numeric_cols[0],  id='scatt_x', placeholder="Select Feature for Scatter Plot"),
    dcc.Graph(id='scatter_fig'),
    ])


@app.callback(
    Output('dcomp_scatter', 'figure'),
    Input('features','value'),
    Input('label_col', "value"),
    Input('prediction_col',"value"),
    Input('group', "value"),
    Input('threshold', 'value'),
    State('submit-val','value')
)
def update_params(features, label_col, prediction_col, group,threshold,  value ):
    p_lab_col = 'predicted_label_'
    acc_col = 'accuracy_'
    params = {'features':features,
              'label_col': label_col,
              'prediction_col': prediction_col,
              'grouping': group,
              'theshold': threshold,
              'predicted_label_col': p_lab_col,
              'accuracy_col': acc_col
               }
    logger.warning(F'params update to {params}')
    X_scaled = scaler.fit_transform(df.loc[:, features])
    y = df.loc[:, label_col]
    comps = dcomp.fit_transform(X_scaled)
    if threshold is None:
        t = y.mean()
        logger.info(F'using ave prob threshold {t}')
    else:
        t = threshold
        logger.info(F'using threshold input of {t}')
    y_predicted_labels = df.loc[:, prediction_col].apply(lambda x: x > float(t)).replace({True: 1, False: 0})
    df.loc[:, 'ica_comp1'] = comps[:, 0]
    df.loc[:, 'ica_comp2'] = comps[:, 1]
    df.loc[:, p_lab_col] = y_predicted_labels
    if isinstance(df.loc[:, [group]].iloc[0], str)==False:
        discrete_groups = df.loc[:, group]
    else:
        disc = KBinsDiscretizer(encode='ordinal', n_bins=5)
        discrete_groups = disc.fit_transform(df.loc[:, [group]])[:, 0]
    acc = list(map(lambda x: get_acc(*x), zip(df.loc[:, label_col], y_predicted_labels)))

    df.loc[:, acc_col] = pd.Series(acc)
    df.loc[:, group + '_disc'] = df.loc[:, group].astype(str)

    fig = px.scatter(df, x='ica_comp1', y='ica_comp2',
                     color= group + '_disc',
                     title=F'Decomposition grouped by {group}', width=1200, height=800)
    return fig



@app.callback(
    Output('silhouette', 'figure'),
    Input('features', 'value'),
    Input('label_col', "value"),
    Input('group', "value"),
    State('submit-val','value'))
def update_silloute(features, label_col,  group, submit):


    X_scaled = StandardScaler().fit_transform(df.loc[:, features])
    groups = df.loc[:, group]
    labels = LabelEncoder().fit_transform(groups)
    s = silhouette_samples(X_scaled, labels )
    y = df.loc[:, label_col]


    to_plot = pd.DataFrame({group: groups, 'scores': s, label_col:y })
    to_plot = to_plot.sort_values(by=[group, 'scores']).reset_index()
    to_plot.loc[:, group] = to_plot.loc[:, group] .astype(str)
    title = F'Sihlouette Score: {np.round(np.mean(s), 3)}'
    fig = px.bar(to_plot,
                 x='scores',
                 color=group,
                 hover_data=to_plot.columns,
                 title=title, range_x=(-1,1))
    fig.update_layout(
        autosize=False,
        width=1000,
        height=800,
    )
    return fig


@app.callback(
    Output('performance_box','figure'),
    Input('label_col', "value"),
    Input('prediction_col', "value"),
    Input('group', "value"),

    State('submit-val', 'value')
    )
def update_box(label_col, prediction_col, group, submit):

    fig = px.box(df, x=label_col,  y=prediction_col, color=group,  points="all", width=800, height=800, title="Predictions by Grouping")
    return fig


@app.callback(
    Output('sun','figure'),
    Input('label_col', "value"),
    Input('prediction_col', "value"),
    Input('group', "value"),
    Input('threshold', 'value'),
    State('submit-val', 'value')
    )
def update_sun(label_col, prediction_col, group, threshold, submit):
    y = df.loc[:, label_col]
    if threshold is None:
        t = y.mean()
        logger.info(F'using ave prob threshold {t}')
    else:
        t = threshold
        logger.info(F'using threshold input of {t}')

    y_predicted_labels = df.loc[:, prediction_col].apply(lambda x: x > float(t)).replace({True: 1, False: 0})
    acc = list(map(lambda x: get_acc(*x), zip(df.loc[:, label_col], y_predicted_labels)))
    df.loc[:, 'accuracy'] = acc
    df.loc[:, 'predicted_label'] = y_predicted_labels
    to_plot = df.groupby([ group,  'accuracy']).count().reset_index()
    fig = px.sunburst(to_plot,
                      path=[group,   'accuracy'],
                      values='pid')
    return fig



@app.callback(
    Output('performance_table','figure'),
    Input('label_col', "value"),
    Input('prediction_col', "value"),
    Input('group', "value"),
    Input('threshold', 'value'),
    State('submit-val', 'value')
    )
def update_performance_table(label_col, prediction_col, group, threshold, submit):
    results = []
    y_true = df.loc[:, label_col]
    y_pred = df.loc[:, prediction_col]
    y_pred_label = pd.Series(y_pred> threshold).replace({True:1, False:0})
    groups =  np.unique(df.loc[:, group])
    groups_evaluated = []
    for g in groups:
        index = df.loc[:, group] == g
        y_true_temp = y_true.loc[index]
        y_pred_temp = y_pred.loc[index]
        y_pred_label_temp = y_pred_label.loc[index]
        try:
            tn, fp, fn, tp = confusion_matrix(y_true_temp, y_pred_label_temp).ravel()

            d = {'auc':  roc_auc_score(y_true_temp, y_pred_temp),
                 'f1' : f1_score(y_true_temp,y_pred_label_temp),
                 'precision' : precision_score(y_true_temp,y_pred_label_temp),
                 'recall': recall_score(y_true_temp, y_pred_label_temp),
                 'accuracy' : accuracy_score(y_true_temp, y_pred_label_temp),
                 'balanced_acc' : balanced_accuracy_score(y_true_temp, y_pred_label_temp),
                 'TP': tp,
                 'FP': fp,
                 'TN': tn,
                 'FN': fn,
                 'weight': y_true_temp.shape[0],

                 }
            groups_evaluated.append(g)
            results.append(d)
        except:
            pass

    output = pd.DataFrame(results, index=groups_evaluated).transpose().round(3)
    output.index.name = 'metric'
    output = output.reset_index()
    print(output)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(output.columns),
                    align='left'),
        cells=dict(values=[output.loc[:, col] for col in output.columns],
                   align='left'))])
    return fig


@app.callback(
    Output('overall_performance_table','figure'),
    Input('label_col', "value"),
    Input('prediction_col', "value"),
    Input('group', "value"),
    Input('threshold', 'value'),
    State('submit-val', 'value')
    )
def update_performance_table(label_col, prediction_col, group, threshold, submit):
    results = []
    y_true = df.loc[:, label_col]
    y_pred = df.loc[:, prediction_col]
    y_pred_label = pd.Series(y_pred> threshold).replace({True:1, False:0})
    groups =  np.unique(df.loc[:, group])
    groups_evaluated = []
    for g in groups:
        index = df.loc[:, group] == g
        y_true_temp = y_true.loc[index]
        y_pred_temp = y_pred.loc[index]
        y_pred_label_temp = y_pred_label.loc[index]
        try:
            tn, fp, fn, tp = confusion_matrix(y_true_temp, y_pred_label_temp).ravel()

            d = {'auc':  roc_auc_score(y_true_temp, y_pred_temp),
                 'f1' : f1_score(y_true_temp,y_pred_label_temp),
                 'precision' : precision_score(y_true_temp,y_pred_label_temp),
                 'recall': recall_score(y_true_temp, y_pred_label_temp),
                 'accuracy' : accuracy_score(y_true_temp, y_pred_label_temp),
                 'balanced_acc': balanced_accuracy_score(y_true_temp, y_pred_label_temp),
                 'TP': tp,
                 'FP': fp,
                 'TN': tn,
                 'FN': fn,
                 'weight': y_true_temp.shape[0],

                 }
            groups_evaluated.append(g)
            results.append(d)
        except:
            pass

    macro_avg = pd.DataFrame(results, index=groups_evaluated).mean(axis=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()
    d = {'auc': roc_auc_score(y_true, y_pred),
         'f1': f1_score(y_true, y_pred_label),
         'precision': precision_score(y_true, y_pred_label),
         'recall': recall_score(y_true, y_pred_label),
         'accuracy': accuracy_score(y_true, y_pred_label),
         'balanced_acc': balanced_accuracy_score(y_true, y_pred_label),
         'TP': tp,
         'FP': fp,
         'TN': tn,
         'FN': fn,
         'weight': y_true.shape[0],

         }
    output = pd.DataFrame([d,  macro_avg.to_dict()], index=['MicroAverage', 'MacroAverage'
                           ]).round(3).reset_index()

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(output.columns),
                    align='left'),
        cells=dict(values=[output.loc[:, col] for col in output.columns],
                   align='left'))])
    return fig


@app.callback(
    Output('roc_auc','figure'),
    Input('label_col', "value"),
    Input('prediction_col', "value"),
    Input('group', "value"),
    Input('threshold', 'value'),
    State('submit-val', 'value')
    ,
    )
def update_roc_auc(label_col, prediction_col, group, threshold, submit):
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    y_true = df.loc[:, label_col]
    y_pred = df.loc[:, prediction_col]
    groups =  np.unique(df.loc[:, group])
    for g in groups:
        index = df.loc[:, group] == g
        y_true_temp = y_true.loc[index]
        y_pred_temp = y_pred.loc[index]
        try:
            fpr, tpr, thresholds = roc_curve(y_true_temp, y_pred_temp)
            auc_score = roc_auc_score(y_true_temp, y_pred_temp)
            name = f"{g} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
        except:
            pass
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    return fig

@app.callback(
    Output('scatter_fig', 'figure'),
    Input('scatt_x', 'value'),
    Input('group', 'value'),
    Input('label_col', "value"),
    State('submit-val', 'value')
)
def update_marginal_distribution(scatt_x, group, label_col, submit):
    fig = px.histogram(df, x=scatt_x, color=label_col, facet_col=group, marginal="box")
    return fig



def main():
    app.run_server(debug=False)


if __name__ == '__main__':
    main()

