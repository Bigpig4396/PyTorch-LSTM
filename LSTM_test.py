import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MyLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = num_layers)
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc1(out)
        return out, hidden

class MyLSTM_trainer(object):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, epoch=500):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.epoch = epoch
        print('input size:', self.input_dim)
        print('hidden size:', self.hidden_dim)
        print('number of layers:', self.num_layers)
        print('output size:', self.output_dim)
        self.model = MyLSTM(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        self.loss_fn = torch.nn.MSELoss()
    
    def input_list_to_batch(self, x):
        new_list = []
        for i in range(len(x)):
            temp = torch.from_numpy(x[i].reshape((1, 1, self.input_dim)))
            new_list.append(temp.float())
        return new_list

    def output_list_to_batch(self, x):
        new_list = []
        for i in range(len(x)):
            temp = torch.from_numpy(x[i].reshape((1, 1, self.output_dim)))
            new_list.append(temp.float())
        return new_list

    def train(self, x, y, init_hidden):
        data_len = len(x)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-3)
        train_x = self.input_list_to_batch(x)
        train_y = self.output_list_to_batch(y)
        for epoch in range(self.epoch):
            loss = 0
            hidden = init_hidden
            for i in range(data_len):
                out, hidden = self.model(train_x[i], hidden)
                loss = loss + self.loss_fn(out, train_y[i])
            print('epoch', epoch, 'loss', loss.data.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, x, init_hidden):
        test_x = self.input_list_to_batch(x)
        hidden = init_hidden
        output_list = []
        for i in range(len(test_x)):
            out, hidden = self.model(test_x[i], hidden)
            output_list.append(out.detach().numpy())
        return output_list

if __name__ == '__main__':
    input_dim=3
    hidden_dim=5
    num_layers=4
    output_dim =2
    rnn = MyLSTM_trainer(input_dim, hidden_dim, num_layers, output_dim, epoch=200)
    x = [np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 0, 0])]
    y = [np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
    h0 = Variable(torch.zeros(num_layers, 1, hidden_dim).float())
    c0 = Variable(torch.zeros(num_layers, 1, hidden_dim).float())
    rnn.train(x, y, (h0, c0))
    print(rnn.predict(x, (h0, c0)))
