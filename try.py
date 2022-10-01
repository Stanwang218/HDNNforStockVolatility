# acquire dataset
import numpy as np
import pandas as pd
import datetime

start = datetime.datetime(2010,1,1)
end = datetime.datetime(2017,9,22)
data = pd.read_csv("google_stock_train.csv")
data['log_ret'] = np.log(data['Close']/data['Close'].shift(1))
# 2265
data['volatility'] = data['log_ret'].rolling(window=252,center=False).std()*np.sqrt(252)
# 2265
a = data[['log_ret','volatility']].values.T
# [2,2265]


datalist = []
for i in range(252,a.shape[1]-41):
    b = a[:,i:i+40]
    # break
    b = b.flatten()  # r + v
    b = np.append(b,a[1,i+40])  # r + v + rv
    datalist.append(b)
datalist = np.array(datalist)
# 1972 * 81
print(datalist.shape)
np.save('final_dataset.npy',datalist)