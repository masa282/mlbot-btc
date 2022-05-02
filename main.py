import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import catboost as cb
import features
import advanced_ml
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


#data BTC-USD converted to dollarbar
df = pd.read_csv("BTCUSD201702-202203_dollarbar.csv").set_index("date")
df.index = pd.to_datetime(df.index)


########################################################################################################
# feature engineering
########################################################################################################
periods = [3, 5, 10, 25, 50, 100, 200, 300]
window = [1, 5, 10, 25, 50, 100, 200, 300]   #rolling(window_size).ewm/mean
features_list = []

#Ratio of x to open
ratio = features.get_ratio_x_open(df, window)
df = pd.concat([df, ratio], axis=1)
for i in ratio.columns:
    features_list.append(i)


#OrderFlow
for p in periods:
    oflow = features.get_OrderFlow(df["volume_ask"], df["volume_bit"], p, window)
    df = pd.concat([df, oflow], axis=1)
    for i in oflow.columns:
        features_list.append(i)

#volatility
v = features.get_VOLA(df["FFD_close"], window)
df = pd.concat([df, v], axis=1)
for i in v.columns:
    features_list.append(i)


# Spread
s = features.get_Spread(df[["FFD_high", "FFD_low"]], window=[3, 5, 7])
df = pd.concat([df, s], axis=1)
for i in s.columns:
    features_list.append(i)


#diff between close and open price
diff = features.get_diff_price(df["FFD_open"],df["FFD_close"], window=[1, 7, 15, 30, 50])
df = pd.concat([df, diff], axis=1)
for i in diff.columns:
    features_list.append(i)


#VPIN
vpin, vpin_cdf = features.get_VPIN(df["close"], df["dollar_vol_ask"], df["dollar_vol_bit"], window)
df = pd.concat([df, vpin_cdf], axis=1)
for i in vpin_cdf.columns:
    features_list.append(i)


#Amihud's Lambda
for p in periods:
    amihud = features.get_Amihud_Lamda(df["Log_close"], df["close"], df["dollar_vol_ask"], df["dollar_vol_bit"], p, window)
    df = pd.concat([df, amihud], axis=1)
    f = amihud.columns
    for i in f:
        features_list.append(i)

#Kyle's Lambda
for p in periods:
    kyle = features.get_Kyle_Lambda(df["close"], df["volume"], p, window)
    df = pd.concat([df, kyle], axis=1)
    f = kyle.columns
    for i in f:
        features_list.append(i)



"""
Feature-selection should be implemented
"""


########################################################################################################
# Machine Learning
########################################################################################################
params = {
    "task_type": "GPU",
    'objective': 'MultiClass',
    'random_seed':77,
    'verbose': 0,
    'learning_rate': .01,
    'l2_leaf_reg': 5, 
    'iterations': 750,
    'max_depth': 6,
    'max_bin': None,
    'auto_class_weights': 'Balanced',
    "early_stopping_rounds": 5,
}
clf = cb.CatBoostClassifier(**params)

#Labeling
label = "bin"
df["ret"] = df['close'].shift(-1) / df["close"] -1
bin_edges = pd.qcut(df["ret"], q=11, labels=range(11))
df[label] = bin_edges.to_numpy()
df.loc[df[label]<4, [label]] = -1
df.loc[(4<=df[label]) & (df[label]<=6), [label]] = 0
df.loc[6<df[label], [label]] = 1
df["t1"] = np.nan
idx = df.iloc[:-1].index
df.loc[idx, ["t1"]] = df.index[1:]
df.dropna(inplace=True)
#get weights of samples
advanced_ml.get_uniqueness(df["close"], pd.DataFrame(df["t1"]), df)
advanced_ml.weighten_sample(df["close"], pd.DataFrame(df["t1"]), df, df["numCoEvents"])


#train: 2017-2020
#val:2021
#test: 2022
train, val, test = df[:164910], df[163000:252785], df[257930:]

#train data from 2017-2020 separately and first because I got better results 
clf.fit(train[features_list], train[label], sample_weight=train["w"])

#cross validation
n_splits=5
groups = advanced_ml.cpcv_pathmap(n_splits,2)
score = advanced_ml.cvScore(clf, val[features_list], val[label], sample_weight=val["w"], scoring='neg_log_loss', t1=val["t1"], cv=n_splits, pctEmbargo=.0, groups=groups)
print(score)

#clf.score(train[features_list_new], train[label])
#clf.score(val[features_list_new], val[label])
#clf.score(test[features_list_new], test[label])


#calcurate cumsum return
pred_test = clf.predict(test[features_list])
out = pd.DataFrame(index=test.index)
out["ret"] = test["close"].shift(-1)/test['close']-1
out["pred_cb"] = pred_test
out["ret_cb"] = out["ret"] * out["pred_cb"]
out["ret_cb"].cumsum().plot(), plt.show()
"""
prob_test = clf.predict_proba(test[features_list_new])
tmp = pd.DataFrame(prob_test, index=test.index ,columns=[-1, 0, 1])
out = pd.concat([out, tmp], axis=1)
"""

#Evaluate model
print("SR : ", out["ret_cb"].mean()/out["ret_cb"].std())
print("SR+ : ", out["ret_cb"][out["ret_cb"]>0].mean()/out["ret_cb"][out["ret_cb"]>0].std())
print("SR- : ", out["ret_cb"][out["ret_cb"]<0].mean()/out["ret_cb"][out["ret_cb"]<0].std())
out["pred_cb"].hist(), plt.show()

