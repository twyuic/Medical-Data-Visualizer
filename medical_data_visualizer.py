# medical_data_visualizer.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 讀取資料（請確保 medical_examination.csv 在同一資料夾）
df = pd.read_csv('medical_examination.csv')

# 1) 新增 overweight 欄位（BMI > 25 => 1，否則 0）
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 2) 正規化 cholesterol 與 gluc （1 -> 0 (正常)， >1 -> 1 (高)）
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


def draw_cat_plot():
    """
    繪製 categorical plot（分 cardio=0/1 兩欄）
    回傳 matplotlib.figure.Figure（並儲存成 catplot.png）
    """
    # 將需要的欄位轉為長格式
    df_cat = pd.melt(df,
                     id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 計算每組的數量
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 使用 seaborn 的 catplot（bar）
    catplot = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar',
        data=df_cat
    )

    fig = catplot.fig
    fig.tight_layout()
    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    """
    清理資料並繪製相關係數熱圖（上三角遮罩）
    回傳 matplotlib.figure.Figure（並儲存成 heatmap.png）
    """
    # 清理條件：
    #  - 收縮壓 <= 舒張壓 (ap_lo <= ap_hi)
    #  - height 與 weight 去掉上下 2.5% 的極端值
    df_heat = df.copy()
    df_heat = df_heat[df_heat['ap_lo'] <= df_heat['ap_hi']]

    # 去除 height / weight 的極端值（2.5% - 97.5%）
    h_low = df_heat['height'].quantile(0.025)
    h_high = df_heat['height'].quantile(0.975)
    w_low = df_heat['weight'].quantile(0.025)
    w_high = df_heat['weight'].quantile(0.975)

    df_heat = df_heat[(df_heat['height'] >= h_low) & (df_heat['height'] <= h_high)]
    df_heat = df_heat[(df_heat['weight'] >= w_low) & (df_heat['weight'] <= w_high)]

    # 計算相關係數矩陣
    corr = df_heat.corr()

    # 建立上三角遮罩
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 畫圖
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        vmax=0.3,
        center=0,
        square=True,
        linewidths=.5,
        ax=ax
    )

    fig.tight_layout()
    fig.savefig('heatmap.png')
    return fig


# 若直接用 python 執行，會產生兩張圖
if __name__ == "__main__":
    draw_cat_plot()
    draw_heat_map()
    print("Saved catplot.png and heatmap.png")
