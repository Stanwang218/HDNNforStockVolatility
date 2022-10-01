import numpy as np
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch


def normalization(data):
    maxi = data.max()
    mini = data.min()
    return (data - mini) / (maxi - mini) * 2 - 1


class stock_dataset(Dataset):
    def __init__(self,training=True,validate=False):
        super(stock_dataset, self).__init__()
        file_name = './final_dataset.npy'
        raw = np.load(file_name)
        matrix_list = []
        label_list = []
        r_list = []
        rv_list = []
        # print(raw.shape)
        # print(raw[0][:40])
        for each in raw:
            label = each[-1]
            r_list.append(np.array(each[:40]))
            rv_list.append(np.array(each[40:-1]))
            cos_r = normalization(each[:40]).reshape(40, 1)
            cos_rv = normalization(each[40:-1]).reshape(40, 1)
            sin_r = np.sqrt(1 - cos_r ** 2)
            sin_rv = np.sqrt(1 - cos_rv ** 2)
            st_r = np.matmul(cos_r, cos_r.transpose()) - np.matmul(sin_r, sin_r.transpose())
            dt_r = np.matmul(sin_r, cos_r.transpose()) + np.matmul(cos_r, sin_r.transpose())
            st_rv = np.matmul(cos_rv, cos_rv.transpose()) - np.matmul(sin_rv, sin_rv.transpose())
            dt_rv = np.matmul(sin_rv, cos_rv.transpose()) + np.matmul(cos_rv, sin_rv.transpose())
            row1 = np.hstack((st_r, st_rv))
            row2 = np.hstack((dt_r, dt_rv))
            matrix = np.vstack((row1, row2))
            matrix_list.append(matrix.reshape(1, 80, 80))
            label_list.append(label)
        self.r_list = r_list
        self.rv_list = rv_list
        self.data = []
        self.label = []
        for i in range(0,len(label_list)-3, 1):
            x = matrix_list[i]
            y = matrix_list[i + 1]
            z = matrix_list[i + 2]
            mat = np.concatenate([x, y, z], axis=0)
            self.data.append(mat)
            self.label.append(np.array([label_list[i], label_list[i + 1], label_list[i + 2]]))
        self.length = len(self.data)
        if not validate:
            if training:
                self.data = self.data[:int(0.7 * self.length)]
                self.label = self.label[:int(0.7 * self.length)]
                self.r_list = self.r_list[:int(0.7 * self.length)]
                self.rv_list = self.rv_list[:int(0.7 * self.length)]
            else:
                self.data = self.data[int(0.7 * self.length):]
                self.label = self.label[int(0.7 * self.length):]
                self.r_list = self.r_list[int(0.7 * self.length):]
                self.rv_list = self.rv_list[int(0.7 * self.length):]

    def __getitem__(self, item):
        return self.data[item], self.label[item], self.r_list[item], self.rv_list[item]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from HDNN import *
    net = hdnn()
    data = stock_dataset()
    print(len(data))
    print(len(stock_dataset(False)))
    dataloader = DataLoader(data,16)
    crt = mymse()
    for x,y,r,rv in dataloader:
        # print(r.shape)
        # print(rv.shape)
        # x = x.type(torch.FloatTensor)
        x = x.to(torch.float32)
        r = r.to(torch.float32)
        rv = rv.to(torch.float32)
        y = y.to(torch.long)
        print(x.shape)
        break

