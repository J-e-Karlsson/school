import pandas as pd
#df = pd.read_csv("dataset.csv")
df = pd.read_csv("user_behavior_dataset.csv")
"""
pd.set_option("display.max_columns", None)
pd.options.display.float_format = "{:.0f}"

df = pd.read_csv("dataset.csv").drop_duplicates()
df.shape
"""
print(df)
