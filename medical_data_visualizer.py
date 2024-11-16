import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
# BMI Calculation
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
# Creating feature overweight
df['overweight'] = (df['BMI'] > 25).astype(int)
# Delete tempory column 'BMI' if it not requested anymore
df.drop('BMI', axis=1, inplace=True)

# 3
# Normolize 'cholesterol'
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
# Normolize 'gluc'
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5
    # prepare data
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])


    # 6
    # Group data and calculate sum of each category
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    

    # 7
    # develop graph
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig


    # 8
    


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.copy()
    df_heat = df_heat[
        (df_heat['ap_lo'] <= df_heat['ap_hi']) &
        (df_heat['height'] >= df_heat['height'].quantile(0.025)) &
        (df_heat['height'] <= df_heat['height'].quantile(0.975)) &
        (df_heat['weight'] >= df_heat['weight'].quantile(0.025)) &
        (df_heat['weight'] <= df_heat['weight'].quantile(0.975)) 
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, vmax=0.3, vmin=-0.1,
                square=True, cbar_kws={'shrink': 0.5} )


    # 16
    fig.savefig('heatmap.png')
    return fig
