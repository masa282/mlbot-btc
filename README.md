# BTC/USD Trading Bot using Machine Learning(ver.1)

"This is my first trading Bot!"
I made a trading bot which predicts the next movement of BTC/USD, and traded with my actual money for about 10 days(20220410-20220421). This is the result!

![actual_reaturn.jpeg](/img/actual_result.jpg)

After running my bot, I figured out "execution strategy" was more important than "how we train a model" because my limit orders did not executed sometimes even if my bot predicted correctly. That can be big opportunity loss! After few days I started to trading, I tuned a few parameters, and it got better. I can't show my strategy and some of my feature enginnerings, especially feature-selection, because I'm currently using it as well. But, I uploaded most of my code! 


## What I did

### 1.) Collect 1min candle data
I used a data(BTC-USD) from 2017 to 2021 as training, and from 2022/01 to 2022/03 as test. In my model, I divided to training and validation data and fit them separately because of performance.

### 2.) Pre-process data
I converted 1min time-bar to dollar-bar. I tried some threshold values, and then I decided to set 5e+5 finally.

### 3.) Find better features
It is difficult to find the best features, which improve the accuracy of prediction. First of all, I picked various features and its timeperiods for example, ratio of ohlcv and features based on microstructure. Secondly I extracted some features from them, checking features importance, dependency, and so on. Finally, I confirmed which ones are effective, testing a model over and over again.

### 4.) Build a model and validate it
Financial data are very noisy, therefore overfitting happens easily. To avoid it, I used bagging model(ET) firstly, but I could not get good performance. I tried a lot of model, such as LightGBM, LSTM, MLP, RF, and I decided to use Catboost in terms of performance and calculation cost. The most effective method was CPCV(Combinatorial Purged Cross-Validation). This is my final results(cumulative return) of my backtest.

![backtest.png](/img/backtest.png)

### 5.) Find better execution strategy
As I mentioned before, this part is one of most important because limit-order is not executed sometimes. First few days, my bot lost in low because order price was not good enough, and stopped loss happened. After tuning a few parameters such as coefficient and formula of order price, my strategy got better! 


Of course, there is still great room for improvement! To be continued... (ver.2)


Thank you for reading!

