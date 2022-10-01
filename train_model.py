from HDNN import *
from read_file import *
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = hdnn().to(device)
dataset = stock_dataset()
# print(len(dataset))
dataloader = DataLoader(dataset=dataset,batch_size=16)
optim = torch.optim.Adam(net.parameters(),lr=0.00001,weight_decay=0.00001)
criterion = mymse()
epochs = 50
x_list = []
y_list = []

for epoch in range(epochs):
    step = 0
    for x,y,r,rv in dataloader:
        step += 1
        x = x.type(torch.cuda.FloatTensor).to(device)
        r = r.type(torch.cuda.FloatTensor).to(device)
        rv = rv.type(torch.cuda.FloatTensor).to(device)
        y = y.to(device)
        pred = net(x,r,rv)
        loss = criterion(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # print(step)
        # if step % 10 == 0:
        #     print("Epoch: {}, {} batch loss = {}".format(epoch,step,loss.item()))

    net.eval()
    with torch.no_grad():
        test_loss = 0
        dataset = stock_dataset(training=False)
        test_dataloader = DataLoader(dataset=dataset,batch_size=16)
        for x, y, r, rv in test_dataloader:
            x = x.type(torch.cuda.FloatTensor).to(device)
            r = r.type(torch.cuda.FloatTensor).to(device)
            rv = rv.type(torch.cuda.FloatTensor).to(device)
            y = y.to(device)
            y_pred = net(x,r,rv)
            test_loss += criterion(y_pred, y).item()

        test_loss /= len(dataloader)
        x_list.append(epoch)
        y_list.append(test_loss)
        print("average loss:{}".format(test_loss))

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function')
plt.plot(x_list,y_list)
plt.show()

model_name = './HDNN.pth'
torch.save(net.state_dict(),model_name)