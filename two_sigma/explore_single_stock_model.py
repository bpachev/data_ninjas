import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import sklearn.metrics as metrics

with pd.HDFStore("train.h5", 'r') as train:
    df = train.get("train")

g = df.groupby('id')

keys = groups.keys()
low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (df.y > high_y_cut)
y_is_below_cut = (df.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

def train(sub, cols_to_use, model):
    cutoff = sub.timestamp.median()
    mean_vals = sub.mean(axis=0)
    cols_to_use = [col for col in cols_to_use if not np.isnan(mean_vals[col])]
    sub.fillna(mean_vals, inplace=True)
    train_mask = (sub.timestamp <= cutoff) & y_is_within_cut
    test_mask = sub.timestamp > cutoff
    y = sub.y
    data = sub.loc[:,cols_to_use]
    model.fit(np.array(data.loc[train_mask].values), np.array(y.loc[train_mask]))
    return metrics.r2_score(y.loc[test_mask], model.predict(np.array(data.loc[test_mask].values)))
    
groups = dict(iter(g))

models = {}
rvals = {}
cols_to_use = ['technical_30', 'technical_20', 'fundamental_11']
for k in keys:
    model = Ridge()
    sub = groups[k]    
    rval = train(sub,cols_to_use, model)
    rvals[k] = rval
    models[k] = model
    print k, rval
    