# medical_data_visualizer.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("medical_examination.csv")

df = df.rename(columns={'gender': 'sex'})

df['overweight'] = (df['weight'] / ((df['height']/100) ** 2) > 25).astype(int)


df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


def draw_cat_plot():
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    )


    df_cat = df_cat.value_counts().reset_index(name='total')


    df_cat['variable'] = pd.Categorical(
        df_cat['variable'],
        categories=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'],
        ordered=True
    )


    fig = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    )

    fig.fig.tight_layout()
    fig.savefig('catplot.png')
    return fig.fig


def draw_heat_map():
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    corr = df_heat.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", square=True, cbar_kws={'shrink':0.5})
    
    fig.savefig('heatmap.png')
    return fig
