import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State, MATCH
from dash import html
import dash
import ast
import pandas as pd
import numpy as np
import pickle

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import time
import pickle
import plotly.graph_objects as go
import shap   # 1. argparseをインポート
from dash_shap_components import ForcePlot, ForceArrayPlot
from shap.plots._force_matplotlib import draw_additive_plot
import io
import base64
# parser = argparse.ArgumentParser(description='')

# # 3. parser.add_argumentで受け取る引数を追加していく
# parser.add_argument('video_id', help='')

def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

exp_name = 'prevent_leak'
processed_data = load_pkl(f'/home/yamanishi/project/airport/src/data/processed_data_{exp_name}.pkl')
X_train, X_val, X_test = processed_data['X']['train'], processed_data['X']['val'], processed_data['X']['test']
y_train, y_val, y_test = processed_data['y']['train'], processed_data['y']['val'], processed_data['y']['test']
X_test_shap = X_test.copy().reset_index(drop=True)
print(X_train.columns)
explainer = load_pkl(f'/home/yamanishi/project/airport/src/data/explainer_optuna_{exp_name}.pkl')
shap_values = load_pkl(f'/home/yamanishi/project/airport/src/data/shap_values_optuna_{exp_name}.pkl')
bst = load_pkl(f'/home/yamanishi/project/airport/src/data/model/best_lgbm_optuna_{exp_name}.pkl')
y_pred = bst.predict(X_test).argmax(1)
print('predict end')
X_orig, X_orig_encoded= processed_data['orig_X'], processed_data['orig_X_encoded']
home_prefecture_d = dict(zip(X_orig_encoded['home_prefecture'].unique(), X_orig['home_prefecture'].unique()))
home_prefecture_d = dict(sorted(home_prefecture_d.items(), key=lambda x:x[0]))
home_prefecture_region_d = dict(zip(X_orig_encoded['home_prefecture_region'].unique(), X_orig['home_prefecture_region'].unique()))
home_prefecture_region_d = dict(sorted(home_prefecture_region_d.items(), key=lambda x:x[0]))
ap21_d = dict(zip(X_orig_encoded['ap21'].unique(), X_orig['ap21'].unique()))
ap21_d = dict(sorted(ap21_d.items(), key=lambda x:x[0]))
ap22_d = dict(zip(X_orig_encoded['ap22'].unique(), X_orig['ap22'].unique()))
ap22_d = dict(sorted(ap22_d.items(), key=lambda x:x[0]))
#y_pred = np.load('../data/y_pred.npy')
y_test = y_test.values

PIE_WIDTH=150
PIE_HEIGHT=150
HEATMAP_WIDTH=600
HEATMAP_HEIGHT=270
HEATMAP_PERS_WIDTH=270
HEATMAP_PERS_HEIGHT=270
TRAJECTORY_WIDTH=400
TRAJECTORY_HEIGHT=225
PIE_ROW_HEIGHT='25vh'
classes = ['中部', '小豆島', '東部', '西部', '高松']
def figure_to_html_img(figure):
    """ figure to html base64 png image """ 
    try:
        tmpfile = io.BytesIO()
        figure.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        shap_html = html.Img(src=f"data:image/png;base64, {encoded}")
        return shap_html
    except AttributeError:
        return ""
    
def generate_image():
    i, id=0, 0
    force_plot = shap.force_plot(explainer.expected_value[i], shap_values[i][id,:], X_test_shap.iloc[id,:], show=True)
    force_plot_mpl = draw_additive_plot(force_plot.data, (30, 7), show=False)
    return figure_to_html_img(force_plot_mpl)

def make_feature(id, shap_values, X_test_shap):
    features = {}
    for i,col in enumerate(X_test_shap.columns):
        features[str(i)] = {'effect': shap_values[id, i],
                            'value': X_test_shap.loc[id, col]}
    return features



def initial_shap_value(n_clicks, id):
    print('id is: ',id)
    id = int(id)
    for j in range(5):
       #force_plot = shap.force_plot(explainer.expected_value[i], shap_values[i][id,:], X_test_shap.iloc[id,:], show=True)
        #force_plot_mpl = draw_additive_plot(force_plot.data, (30, 7), show=False)
        #return figure_to_html_img(force_plot_mpl)
        featureNames = {str(i): X_test.columns[i] for i in range(len(X_test.columns))}
        print('featureNames', featureNames)
        print('feature', make_feature(id, shap_values[j], X_test_shap))
        return ForcePlot(
            id=f'sample{id}',
            className=j,
            title=id,
            baseValue = explainer.expected_value[j],
            link='identity',
            featureNames=featureNames,
            outNames='Output Value',
            features=make_feature(id, shap_values[j], X_test_shap),
            hideBaseValueLabel=False,
            hideBars=False,
            labelMargin=0,
            plot_cmap=['#DB0011', '#000FFF'],
            )

app =dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
sidebar = html.Div(
    [
        dbc.Row(
            [
                html.H5('Settings',
                        style={'margin-top': '12px', 'margin-left': '24px'})
                ],
            style={"height": "5vh"},
            className='bg-primary text-white font-italic'
            ),
        dbc.Row(
            [
                html.Div([
                        html.P('Prediction', id='prediction', className='font-weight-bold'),
                        dcc.Dropdown(id='pred', multi=False, value='0',
                                 options=[{'label': str(x), 'value': str(x)}
                                          for x in range(5)],
                                 style={'width': '150px'}
                                 ),
                        html.P('Ground Truth', id='ground truth', className='font-weight-bold'),
                        dcc.Dropdown(id='gt', multi=False, value='0',
                                 options=[{'label': str(x), 'value': str(x)}
                                          for x in range(5)],
                                 style={'width': '150px'}
                                 ),
                        html.Button(id='pred-gt-button',
                                children='apply',
                                n_clicks=0, 
                                style={'margin-top': '16px'},
                                className='bg-dark text-white',
                                ),
                        html.P(id='test-id', className='font-weight-bold'),
                        dcc.Dropdown(id='id', multi=False, value='0',
                                 options=[{'label': str(x), 'value': str(x)}
                                          for x in range(len(X_test))],
                                 style={'width': '150px'}
                                 ),
                        html.Button(id='id-button',
                                children='apply',
                                n_clicks=0, 
                                style={'margin-top': '16px'},
                                className='bg-dark text-white',
                                ),
                    html.Hr()
                ]
                )
            ],
            style={'height': '50vh', 'margin': '8px'}),
        ]
    )


content = html.Div(
    [
         dbc.Row(
            [
                html.H5('Takamatsu Airport Dashboard',
                        style={'margin-top': '12px', 'margin-left': '24px'})
                ],
            style={"height": "5vh"},
            className='bg-secondary text-white font-italic'
            ),
        dbc.Row(
            [
                dbc.Col([
                    html.Div([html.P(id='shap-for-each-sample', className='font-weight-bold'),
                        #html.Img(src=generate_image())],
                        #dcc.Graph(id='shap', figure=generate_image())],
                        #dcc.Graph(id='shap', 
                        #          className='font-weight-bold',
                        #          )],
                        html.Div(id='shap', className='row'),],
                        style={'width': HEATMAP_WIDTH+200, 'height':HEATMAP_HEIGHT+40,'margin-left': 'auto', 'margin-right': 'auto'})
                    ]),
                dbc.Col([
                    html.Table(
        # 辞書の各要素を行として表示する
                        [html.Tr([html.Th(col), html.Td(home_prefecture_d[col])]) for col in home_prefecture_d],
                        style={'width': 80, 'font-size': '12px'}
                    )], style={'width': 100, 'margin': 10, 'padding': 10}),
                dbc.Col([
                    html.Table(
        # 辞書の各要素を行として表示する
                        [html.Tr([html.Th(col), html.Td(home_prefecture_region_d[col])]) for col in home_prefecture_region_d],
                        style={'width': 80, 'font-size': '12px'}
                    )], style={'width': 100, 'margin': 10, 'padding': 10}),
                dbc.Col([
                    html.Table(
        # 辞書の各要素を行として表示する
                        [html.Tr([html.Th(col), html.Td(ap21_d[col])]) for col in ap21_d],
                        style={'width': 80, 'font-size': '12px'}
                    )],style={'width': 100, 'margin': 10, 'padding': 10}),
                dbc.Col([
                    html.Table(
        # 辞書の各要素を行として表示する
                        [html.Tr([html.Th(col), html.Td(ap22_d[col])]) for col in ap22_d],
                        style={'width': 80, 'font-size': '12px'}
                    )], style={'width': 100, 'margin': 10, 'padding': 10}),
    ], style={'margin': '0px', 'padding': '0px'})
    ])

    

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=2, className='bg-light'),
                dbc.Col(content, width=10),
                ]
            ),
        ],
    fluid=True
    )

    
@app.callback(
    Output('id', 'options'),
    [State('pred', 'value'),
     State('gt', 'value'),
     Input('pred-gt-button', 'n_clicks')]
)
def update_dropdown_id_options(value_pred, value_gt, n_clicks):
    id = ((y_pred==int(value_pred))&(y_test==int(value_gt))).nonzero()[0]
    options =[{'label': str(i), 'value': str(i)} for i in id]
    return options
    

@app.callback(
    Output('shap', 'children'),
    [Input('id-button', 'n_clicks'),
     State('id', 'value'),]
)
def update_shap_value(n_clicks, id):
    print('id is: ',id)
    id = int(id)
    plots = []
    for j in range(5):
       #force_plot = shap.force_plot(explainer.expected_value[i], shap_values[i][id,:], X_test_shap.iloc[id,:], show=True)
        #force_plot_mpl = draw_additive_plot(force_plot.data, (30, 7), show=False)
        #return figure_to_html_img(force_plot_mpl)
        featureNames = {str(i): X_test.columns[i] for i in range(len(X_test.columns))}
        print('featureNames', featureNames)
        print('feature', make_feature(id, shap_values[j], X_test_shap))
        
        plots.append(html.Div(ForcePlot(
            id=f'sample{id}',
            className=str(j),
            title=classes[j],
            baseValue = explainer.expected_value[j],
            link='identity',
            featureNames=featureNames,
            outNames=['Output Value'],
            features=make_feature(id, shap_values[j], X_test_shap),
            hideBaseValueLabel=False,
            hideBars=False,
            labelMargin=0,
            plot_cmap=['#DB0011', '#000FFF'],
            )))
    return plots
    #for i in range(5):
        #shap.force_plot(explainer.expected_value[i], shap_values[i][id,:], X_test_shap.iloc[id,:], show=True,matplotlib=True)


if __name__ == "__main__":
    app.run_server(debug=True, port=1274)