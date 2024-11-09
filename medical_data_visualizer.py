import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')


# 2
# adding an overweight col to the dataframe

# processing function
def is_overweight(row):
  """
  Checks if a person is overweight based on their BMI.
  """
  
  # calculating BMI
  BMI = row["weight"] / ((row["height"] * 0.01) ** 2)

  if BMI > 25:
    return 1
  else:
    return 0

df['overweight'] = df.apply(is_overweight, axis = 1)


# 3
# Normalizing the cholesterol and gluc column
def normalize_cholesterol_data(row):
  """
  normalizing cholesterol data.
  """
  
  if row["cholesterol"] == 1:
    return 0
  else:
    return 1

def normalize_gluc_data(row):
  """
  normalizing gluc data.
  """
  
  if row["gluc"] == 1:
    return 0
  else:
    return 1

df["cholesterol"] = df.apply(normalize_cholesterol_data, axis = 1)
df["gluc"] = df.apply(normalize_gluc_data, axis = 1)


# 4
def draw_cat_plot():
    # 5
    # actually we pivot around cardio because 
    # we are exploring the relationship between cardiac disease, 
    # body measurements, blood markers, and lifestyle choices.
    df_cat = df.melt(
        id_vars = ["cardio"], 
        value_vars = ["cholesterol", "gluc", "smoke", "alco", "active", "overweight"]
    )


    # 6
    # getting all the possibles relationship between having a cardio disease and
    # and the truth value of cholesterol, gluc, smoke, alco, active, overweight counts.
    df_cat = df_cat.groupby(["cardio", "variable", "value"]).size().reset_index(name = "total")
    df_cat.rename(columns = {"value": "feature_value"}, inplace = True)

    # 7
    categorical_plot = sns.catplot(
        data = df_cat, 
        y = "total", 
        x = "variable", 
        hue = "feature_value",
        kind = "bar", 
        col = "cardio", 
        height = 5, 
        aspect = 1
    );


    # 8
    fig = categorical_plot.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    filter_1 = df['ap_lo'] <= df['ap_hi']
    filter_2 = df['height'] >= df['height'].quantile(0.025)
    filter_3 = df["height"] <= df["height"].quantile(0.975)
    filter_4 = df["weight"] >= df["weight"].quantile(0.025)
    filter_5 = df["weight"] <= df["weight"].quantile(0.975)

    df_heat = df[filter_1 & filter_2 & filter_3 & filter_4 & filter_5]


    # 12
    corr = df_heat.corr()


    # 13
    # help me visualizing a heatmap without redundant information on both sides of the diagonal
    # creates a matrix of the same shape as matrix filled with True and False values.
    mask = np.triu(np.ones_like(corr, dtype=bool))


    # 14
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15
    custom_ticks = [0.24, 0.16, 0.08, 0.0, -0.08]

    sns.heatmap(
      corr,
      mask = mask,
      annot = True,
      fmt = ".1f",
      center = 0.00,
      square = True,
      linewidths = 0.5,
      cbar_kws = {"shrink": 0.5, "ticks": custom_ticks},
      vmin = -0.16,                 
      vmax = 0.32,
      ax = ax
    )


    # 16
    fig.savefig('heatmap.png')
    return fig
