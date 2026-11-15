import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

filename = os.path.join(os.path.dirname(__file__), "data", "iris.csv")

df = pd.read_csv(filename)
table = df.to_markdown(index=False)
print(table)

