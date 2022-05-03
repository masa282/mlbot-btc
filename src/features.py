from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import advanced_ml


##############################################################################################
## ratio of x to open_price
##############################################################################################
def get_ratio_x_open(df, window):
    df_ = pd.DataFrame(index=df.index)
    f = ["Log_high", "Log_low", "Log_close", "volume"]
    for w in window:
        for i in f:
            df_[f"Ratio_{i}{w}"] = (df[i] / df["Log_open"] ).ewm(w).mean()
    return df_


##############################################################################################
## diff between close and open price
##############################################################################################
def get_diff_price(open, close, window):
    df_ = pd.DataFrame(index=close.index)
    for w in window:
        df_[f"oc_diff{w}"] = (open - close).rolling(w).mean()
    return df_


##############################################################################################
## Order-flow
##############################################################################################
def get_OrderFlow(vols_ask, vols_bit, period, window):
    df_ = pd.DataFrame(index=vols_ask.index)
    oflow = []
    for i in range(period, df_.shape[0]+1):
        oflow.append(vols_ask.iloc[i-period:i].sum() - vols_bit.iloc[i-period:i].sum())
    for w in window:
        df_[f"OrderFlow{period}_{w}"] = pd.DataFrame(oflow, index=vols_ask.iloc[period-1:].index).ewm(w).mean()
    return df_ 
    

##############################################################################################
## Volatility
##############################################################################################
def get_VOLA(close, window):
    df_ = pd.DataFrame(index=close.index)
    for w in window:
        s = (close.diff() / close).ewm(w).std()
        s.name = f"VOLA{w}"
        df_ = pd.concat([df_, s], axis=1)
    return df_


##############################################################################################
## Estimated Spread
##############################################################################################
def get_Spread(df, window):
    df_ = df.copy()
    df_.columns = ["high", "low"]
    for w in window:
        corwin = advanced_ml.corwinSchultz(df_[["high", "low"]], s1=1)
        s = corwin["Spread"].rolling(w).mean()
        s.name = f"Spread{w}"
        df_ = pd.concat([df_, s], axis=1)
    return df_.drop(["high", "low"], axis=1)


##############################################################################################
## Kyle’s Lambda: 
##############################################################################################
def get_Kyle_Lambda(close, volume, period, window): #volume → dollarVol
    df_ = pd.DataFrame(index=close.index)
    price_diff = close.diff().dropna().abs()
    #clipping
    p01, p99 = price_diff.quantile(0.01), price_diff.quantile(0.99)
    price_diff = np.array(price_diff.clip(p01, p99), dtype="float32")
    dollarVol = np.array((close*volume)[1:], dtype="float32")
    reg = []
    for i in range(period, len(close)+1):
        reg.append(LinearRegression().fit(dollarVol[i-period:i].reshape(-1, 1), price_diff[i-period:i].reshape(-1, 1)).coef_)
    for w in window:
        df_[f"Kyle{period}_{w}"] = pd.DataFrame(np.array(reg).reshape(-1,1), index=df_.iloc[period-1:].index).ewm(w).mean()
    return df_


##############################################################################################
## Amihud’s Lambda
## - I changed price*volume to price**2*volume
##############################################################################################
def get_Amihud_Lamda(log_close, close, dollarVol_ask, dollarVol_bit, period, window):
    df_ = pd.DataFrame(index=log_close.index)
    price_diff = log_close.diff().dropna().abs()
    p01, p99 = price_diff.quantile(0.01), price_diff.quantile(0.99)
    price_diff = np.array(price_diff.clip(p01, p99), dtype="float32")
    dollarVol_diff = (close*(dollarVol_ask - dollarVol_bit))[1:]
    reg = []
    for i in range(period, len(log_close)+1):
        reg.append(LinearRegression().fit(dollarVol_diff[i-period:i].values.reshape(-1, 1), price_diff[i-period:i].reshape(-1, 1)).coef_)
    for w in window:
        df_[f"Amihud{period}_{w}"] = pd.DataFrame(np.array(reg).reshape(-1,1), index=df_.iloc[period-1:].index).ewm(w).mean()
    return df_


##############################################################################################
## VPIN: Volume-synchronized probability of informed trading
##############################################################################################
def get_VPIN(close, dollarVol_ask, dollarVol_bit, window):
    df_ = pd.DataFrame(index=close.index)
    df_cdf_ = pd.DataFrame(index=close.index)
    vpin = []
    dollarVol_total = dollarVol_ask + dollarVol_bit
    for w in window:
        order_imbalance = np.abs(dollarVol_ask - dollarVol_bit).rolling(w).mean()
        vpin = order_imbalance / dollarVol_total.rolling(w).mean()
        df_[f"VPIN{w}"] = pd.DataFrame(vpin, index=close.iloc[w-1:].index)
        df_cdf_[f"VPIN_CDF{w}"] = df_[f"VPIN{w}"].rank(pct=True)
    return df_, df_cdf_
