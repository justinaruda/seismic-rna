import pandas as pd
import numpy as np

from  dreem.draw.util import *

import plotly.graph_objects as go
from plotly.offline import plot, iplot

from dreem.draw import  util
from itertools import cycle
from typing import Tuple, List
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from plotly.subplots import make_subplots


LIST_COLORS = ['red','green','blue','orange','purple','black','yellow','pink','brown','grey','cyan','magenta']

def mutation_fraction(df, show_ci:bool=True)->dict:
    assert len(df) == 1, "df must have only one row"
    mh = df.iloc[0]
    cmap = {"A": "red", "T": "green", "G": "orange", "C": "blue"}  # Color map
    
    traces, layouts = [], []
    mh.index_selected = [i + 1 for i in mh.index_selected] # index starts at 1
    mh_unrolled = pd.DataFrame({'mut_rate':list(mh.mut_rates), 'base':list(mh.sequence), 'index_reset':list(range(len(mh.index_selected))),'index_selected':mh.index_selected, 'paired':list(mh.structure)})

    for bt in set(mh['sequence']):
        df_loc = mh_unrolled[mh_unrolled['base'] == bt]
        if len(df_loc) == 0:
            continue

        hover_attr = pd.DataFrame({'mut_rate':list(df_loc.mut_rate),
                                        'base':list(df_loc.base), 
                                        'index': df_loc['index_selected'],
                                        'paired':[{'.':True, '(':False,')':False}[s] for s in df_loc.paired]})
        traces.append(go.Bar(
            x= np.array(df_loc['index_reset']),
            y= np.array(df_loc['mut_rate']),
            name=bt,
            marker_color=cmap[bt],
            text = hover_attr,
            hovertemplate = ''.join(["<b>"+ha+": %{text["+str(i)+"]}<br>" for i, ha in enumerate(hover_attr)]),
            ))
        if show_ci:
            traces[-1].update(
                        error_y=dict(
                        type='data',
                        symmetric=False,
                        ))

    

    fig = go.Figure(data=traces)

    fig.update_layout(title=f"{mh['sample']} - {mh['reference']} - {mh['section']} - {mh['cluster']} - {mh['num_aligned']} reads",
                        xaxis=dict(title="Sequence"),
                        yaxis=dict(title="Mutation rate", range=[0, 0.1]))
   
    fig.update_yaxes(
            gridcolor='lightgray',
            linewidth=1,
            linecolor='black',
            mirror=True,
            autorange=True
    )
    fig.update_xaxes(
            linewidth=1,
            linecolor='black',
            mirror=True,
            autorange=True
    )

    fig.update_xaxes(
            tickvals=mh_unrolled['index_reset'],
            ticktext=["%s %s" % ({'.':'(U)','(':'(P)',')':'(P)'}[x], str(y)) for (x,y) in zip(mh['structure'],mh['index_selected'])],
            tickangle=90,
            autorange=True
    )

    return {'fig':fig, 'df':mh}


def mutations_in_barcodes(data):
    
    fig = go.Figure()

    len_barcode = len(data['sequence'].iloc[2])
    for sample in data['sample'].unique():
        hist = np.sum(np.stack(data[data['sample']==sample]['num_of_mutations'].values), axis=0)
        bin_edges = np.arange(0, len_barcode, 1)
        
        fig.add_trace(
            go.Bar(
                x=bin_edges[:-1],
                y=hist,
                name=sample,
                visible=False,
                hovertemplate='Number of mutations: %{x}<br>Number of reads: %{y}<extra></extra>',
                ))
        
    fig.data[0].visible = True
    
    fig.update_layout(barmode='stack', title='Number of mutations in barcodes - {}'.format(data['sample'].unique()[0]))
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=list([
                    dict(label = sample,
                            method = "update",
                            args = [{"visible": [sample == s for s in data['sample'].unique()]},
                                    {"title": 'Number of mutations in barcodes - {}'.format(sample)}])
                    for sample in data['sample'].unique()
                ]), 
                x = 0.7,
                xanchor = 'left',
                y = 1.1,
                yanchor = 'top'
                )
            ])
    
    return {'fig': fig, 'data': data[['sample','reference','num_of_mutations']]}


def deltaG_vs_mut_rates(df:pd.DataFrame, models:List[str]=[],  savefile=None, auto_open=False, use_iplot=True, title=None)->dict:

    df_temp = pd.DataFrame()
    for _, row in df.iterrows():
        df_temp = pd.concat([df_temp, pd.DataFrame({'reference':row.reference, 'index':row.index_selected, 'mut_rates':row.mut_rates, 'num_aligned':row.num_aligned, 'deltaG':row['deltaG'],'base':list(row.sequence), 'paired':[s !='.' for s in row.structure]}, index= [i+1 for i in range(len(row.sequence))])])
    
    assert len(df_temp) > 0, "No data to plot"
    df = df_temp.reset_index()

    hover_attr = ['num_aligned','mut_rates','base','index','reference','deltaG']
    tra = {}
    for is_paired, prefix in zip([True,False], ['Paired ','Unpaired ']):
        markers = cycle(list(range(153)))
        x=np.array(df[df.paired == is_paired]['deltaG'])
        y=np.array(df[df.paired == is_paired]['mut_rates'])
        tra[prefix] = go.Scatter(
            x=x,
            y=y,
            text = df[df.paired == is_paired][hover_attr],
            marker_size= df[df.paired == is_paired]['num_aligned']/200,
            mode='markers',
            name=prefix,
            hovertemplate = ''.join(["<b>"+ha+": %{text["+str(i)+"]}<br>" for i, ha in enumerate(hover_attr)]),
            line=dict(color='green' if is_paired else 'red'))
            
        
        for m in models:
            if len(y) > 0:
                fit = util.Fit()
                x_sorted, pred_y_sorted = fit.predict(x,y,m, prefix)
                tra[fit.get_legend()] = go.Scatter(
                    x=x_sorted,
                    y=pred_y_sorted,
                    mode='lines+markers',
                    name=fit.get_legend(), 
                    marker=dict(symbol=next(markers)),
                    line=dict(color='darkseagreen' if is_paired else 'crimson', dash='dash'))

    layout = dict(title = 'Mutation rates of paired / unpaired residues vs the expected energy of the molecule',
            xaxis= dict(title= 'DeltaG',ticklen= 5,zeroline= False),
            yaxis= dict(title= 'Mutation rate ',ticklen= 5,zeroline= False),
            )

    fig = go.Figure(data=list(tra.values()), layout=layout)

    return {'fig':fig, 'df':df}

    
def exp_variable_across_samples(df:pd.DataFrame, experimental_variable:str, models:List[str]=[],  savefile=None, auto_open=False, use_iplot=True, title=None)->dict:

    colors = cycle(LIST_COLORS)
    data = pd.DataFrame()
    for _, row in df.iterrows():
        data = pd.concat([data, pd.DataFrame({'sample':row['sample'],experimental_variable:getattr(row,experimental_variable), 'index':list(row.index_selected), 'base':list(row.sequence), 'mut_rates':list(row.mut_rates), 'paired':[s !='.' for s in row.structure_selected]}, index=list(range(len(row.index_selected))))])
    data = data.reset_index().rename(columns={'level_0':'index_subsequence'})
    data = data.sort_values(by='index')
    data['Marker'] = data['paired'].apply(lambda x: {True: 0, False:1}[x])
    hover_attr = ['base','mut_rates','sample',experimental_variable, 'paired']

    tra = {}
    for idx, row in data.groupby('index_subsequence'):
        color = next(colors)
        markers = cycle(list(range(153)))
        name = f"({row['base'].iloc[0]},{row['index'].iloc[0]})"
        tra[row['index'].iloc[0]] = go.Scatter(
            x= row[experimental_variable], 
            y= row['mut_rates'], 
            text = data[hover_attr],
            mode='lines+markers',
            marker = dict(symbol = list(map(util.Setshape, data['Marker']))),
            name= name,
            hovertemplate = ''.join(["<b>"+ha+": %{text["+str(i)+"]}<br>" for i, ha in enumerate(hover_attr)]),
            line=dict(color=color))

        my_dash = cycle(['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'])
        for m in models:
            x= row[experimental_variable]
            y= row['mut_rates']
            fit = util.Fit()
            x_sorted, pred_y_sorted = fit.predict(x,y,m, name)
            tra[fit.get_legend()] = go.Scatter(
                x=x_sorted,
                y=pred_y_sorted,
                mode='lines',
                line=dict(dash= next(my_dash), color=color),
               # line=dict(color=color),
                name=fit.get_legend())

    layout = dict(title = 'Mutation rates of paired / unpaired residues vs '+experimental_variable,
            xaxis= dict(title= experimental_variable,ticklen= 5,zeroline= False),
            yaxis= dict(title= 'Mutation rate ',ticklen= 5,zeroline= False),
            )

    #tra = {arg:tra[list(tra.keys()[arg])] for arg in np.argsort(np.array([int(k[k.index('(')+3:k.index(')')]) for k in tra.keys()]))}

    fig = dict(data = list(tra.values()), layout = layout)

    return {'fig':fig, 'df':data}



def auc(df:pd.DataFrame,  savefile=None, auto_open=False, use_iplot=True, title=None)->dict:
    def make_roc_curve(X, y, y_pred, fig, title):
        fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
                            mode='lines',
                            name=title,
                            line=dict(width=2)))
        return fig


    fig = go.Figure()
    for row in df.iterrows():
        X = row[1]['mut_rates'].reshape(-1, 1)
        y = np.array([1 if c == '.' else 0 for c in row[1]['structure_selected']]).reshape(-1, 1)
        y_pred = LogisticRegression().fit(X, y.ravel()).predict_proba(X)[:,1]
        fig = make_roc_curve(X, y, y_pred, fig, row[1]['unique_id'])


    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], line=dict(color='black', width=2, dash='dash'), showlegend=False)
    )
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate')

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')

    return {'fig':fig, 'df':df}


def mutation_fraction_delta(df, savefile=None, auto_open=False, use_iplot=True, title=None)->dict:
    assert len(df) == 2, "df must have 2 row"
    mp_attr = ['sample', 'reference', 'section', 'cluster']
    df['unique_id'] = df.apply(lambda row: ' - '.join([str(row[attr]) for attr in mp_attr]), axis=1)

    mh = pd.Series(
        {
            'mut_rates': df['mut_rates'].values[0] - df['mut_rates'].values[1],
            'sequence': ''.join([c1 if c1 == c2 else '-' for c1,c2 in zip(df['sequence'].values[0],df['sequence'].values[1])]),
            'title': "{} - {} reads vs {} - {} reads".format(df['unique_id'].values[0], df['num_aligned'].values[0], df['unique_id'].values[1], df['num_aligned'].values[1])
        }
    )
    cmap = {"A": "red", "T": "green", "G": "orange", "C": "blue", '-':'grey'}  # Color map
    
    traces, layouts = [], []
    mh_unrolled = pd.DataFrame({'mut_rate':list(mh.mut_rates), 'base':list(mh.sequence), 'index_reset':list(range(len(mh.sequence)))})

    for bt in set(mh['sequence']):
        df_loc = mh_unrolled[mh_unrolled['base'] == bt]
        if len(df_loc) == 0:
            continue

        hover_attr = pd.DataFrame({'mut_rate':list(df_loc.mut_rate),
                                        'base':list(df_loc.base)})
        traces.append(go.Bar(
            x= np.array(df_loc['index_reset']),
            y= np.array(df_loc['mut_rate']),
            name=bt,
            marker_color=cmap[bt],
            text = hover_attr,
            hovertemplate = ''.join(["<b>"+ha+": %{text["+str(i)+"]}<br>" for i, ha in enumerate(hover_attr)]),
            ))
    

    fig = go.Figure(data=traces, 
                    layout=go.Layout(
                        title=go.layout.Title(text=mh['title']),
                        xaxis=dict(title="Sequence"),
                        yaxis=dict(title="Mutation rate", range=[0, 0.1])))

    fig.update_yaxes(
            gridcolor='lightgray',
            linewidth=1,
            linecolor='black',
            mirror=True,
            autorange=True
    )
    fig.update_xaxes(
            linewidth=1,
            linecolor='black',
            mirror=True,
            autorange=True
    )
    
    return {'fig':fig, 'df':mh}

               
def _mutations_per_read_subplot(data):
    hist = np.sum(np.stack(data.values), axis=0)
    bin_edges = np.arange(0, max(np.argwhere(hist != 0)), 1)
    return go.Bar( x=bin_edges, y=hist, showlegend=False, marker_color='indianred')

def mutations_per_read_per_sample(data):

    unique_samples = data['sample'].unique()
    fig = make_subplots(rows=len(unique_samples), cols=1, vertical_spacing=0.4/len(unique_samples),
                        subplot_titles=['Number of mutations per read - {}'.format(sample) for sample in unique_samples])
    for i_s, sample in enumerate(unique_samples):
        
        fig.add_trace(_mutations_per_read_subplot(data[data['sample']==sample]['num_of_mutations'].reset_index(drop=True)),
                      row=i_s+1, col=1 )
        fig.update_yaxes(title='Count')
        fig.update_xaxes(dtick=10)

    fig.update_layout(autosize=True, height=len(unique_samples)*500, title='Number of mutation per read across samples')
    return {
        'fig':fig,
        'data':data
        }
    
def num_aligned_reads_per_reference_frequency_distribution(data):
    return {
            'fig':go.Figure(
                go.Histogram(
                    x=data, 
                    showlegend=False, 
                    marker_color='indianred',
                    ),
                layout=go.Layout(
                    title=go.layout.Title(text='Number of aligned reads per reference frequency'),
                    xaxis=dict(title="Number of aligned reads"),
                    yaxis=dict(title="Count")
                    )
                
                
                
                
                ),
            'data':data
            }