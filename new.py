from dummy_one import dummy_one
import pandas as pd

data = pd.read_csv('test.csv')

dummy_one(data, thresh=30, drop=True).to_csv('testing.csv', index_label=False, index=False)