# medical_data_visualizer.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data
df = pd.read_csv("medical_examination.csv")

# Rename 'gender' to 'sex' to match FreeCodeCamp tests
df = df.rename(columns={'gender': 'sex'})

# 2. Add overweight column
df['overweight'] = (df['weight'] / ((df['height']/100) ** 2) > 25).astype(int)

# 3. Normalize data: 0 = good, 1 = bad
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Draw Categorical Plot
def draw_cat_plot():
    # 4a. Melt dataframe with correct feature order
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    )

    # 4b. Group and count
    df_cat = df_cat.value_counts().reset_index(name='total')

    # 4c. Ensure the correct x-axis order
    df_cat['variable'] = pd.Categorical(
        df_cat['variable'],
        categories=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'],
        ordered=True
    )

    # 4d. Draw the catplot
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

# 5. Draw Heat Map
def draw_heat_map():
    # 5a. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    # 5b. Calculate the correlation matrix
    corr = df_heat.corr()
    
    # 5c. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # 5d. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 5e. Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", square=True, cbar_kws={'shrink':0.5})
    
    fig.savefig('heatmap.png')
    return fig
