from torch.utils.data import DataLoader
from read_file import *
import matplotlib.pyplot as plt
from HDNN import *
import torch


data = stock_dataset(validate=True)
dataloader = DataLoader(data,16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = hdnn().to(device)
net.eval()
net.load_state_dict(torch.load('./HDNN.pth'))
x_list = [i for i in range(len(data))]
y_list = []
pred_list = []
with torch.no_grad():
    for x, y, r, rv in dataloader:
        x = x.type(torch.cuda.FloatTensor).to(device)
        r = r.type(torch.cuda.FloatTensor).to(device)
        rv = rv.type(torch.cuda.FloatTensor).to(device)
        y = y.to(device)
        pred = net(x, r, rv)
        y_list.extend(y.cpu().numpy().tolist())
        pred_list.extend(pred.cpu().numpy().tolist())

y_list = np.array(y_list)
y_list = y_list[:,0]
# print(y_list.shape)
# print(y_list)
# print(len(y_list))
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.title('Volatility for Google Stock')
plt.plot(x_list,y_list,color='blue')
# plt.plot(x_list,pred_list,)
plt.legend(labels=('Original Volatility','Predicted Volatility'))
plt.show()
print(((np.array(pred_list) - y_list) ** 2).sum())