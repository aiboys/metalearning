import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import torch
from optimizee import optim
# from meta import goal

def Himmelblau(x):

    return (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2

def goal2(w,x,y):

    return torch.mean(torch.sum((torch.matmul(w,x.unsqueeze(-1)).squeeze()-y)**2,dim=1),dim=0)

def train(ff,length,batch_size, w, x_torch, y, iters, lr=0.001, arg=''):
    state1 = None
    state2 = None
    state = None
    x_record = []
    losses = []
    x_torch = x_torch.detach()
    if arg =='adamoff':
        # x_torch.retain_grad()
        optimizee = torch.optim.Adam([x_torch], lr=0.1)
        for i in range(iters):
            a = x_torch.detach_().cpu().data.numpy().copy()
            x_record.append(a)
            x_torch.requires_grad = True
            optimizee.zero_grad()
            loss = goal2(w, x_torch,y)
            loss.backward()
            optimizee.step()
            losses.append(loss.detach_())
            x_torch = x_torch.detach_()
    else:
         # x_torch.retain_grad()
         for t in range(iters):
            # print(t)

            a = x_torch.detach_().cpu().data.numpy().copy()
            x_record.append(a)
            x_torch.requires_grad = True
            loss = goal2(w,x_torch,y)
            loss.backward()
            losses.append(loss.detach_())
            optimize = optim(batch_size, x_torch.grad, state, length, decay=0.99, learning_rate=lr)
            if arg == 'sgd':
                update = optimize.SGD()
            if arg == 'adam':
                update, state1, state2 = optimize.Adam(iters=t, tho1=0.9, tho2=0.999, state1=state1, state2=state2)
            if arg == 'adagrad':
                update, state = optimize.Adagrad()
            if arg == 'sgdm':
                update, state = optimize.SGDM()
            if arg == 'nes':
                update, state = optimize.Nesterov_momentum(x_torch,tho=0.01)
            if arg == 'rms':
                update, state = optimize.RMS()
            x_torch = x_torch + update
            x_torch = x_torch.detach_()

    return x_record, losses

def plot():
    x = torch.tensor([1.0, -4.0], requires_grad=True)
    length = 2
    iters = 210
    x_sgd, loss_sgd = train(Himmelblau,length, x, iters, lr=0.01, arg='sgd')
    x_sgdm, loss_sgdm = train(Himmelblau,length, x, iters, lr=0.01, arg='sgdm')
    x_adagrad, loss_adgrad= train(Himmelblau,length, x, iters, lr=0.1, arg='adagrad')
    # x_nes, loss_nes= train(length,x,iters,lr=0.001, arg='nes')
    x_rms, loss_rms = train(Himmelblau,length, x, iters, lr=0.01, arg='rms')
    x_adam, loss_adam = train(Himmelblau, length, x, iters, lr=0.1, arg='adam')
    x_adam_off, losses_adam_off   = train(Himmelblau, length,x,iters,lr=0.1, arg='adamoff')
    x_adagradoff, losses_adagardoff = [], []

    x = torch.tensor([1.0, -4.0], requires_grad=True)
    optimizee2 = torch.optim.Adagrad([x], lr=0.1)
    for i in range(iters):
            a = x.detach_().data.numpy().copy()
            x_adagradoff.append(a)
            x.requires_grad = True
            optimizee2.zero_grad()
            loss = Himmelblau(x)
            loss.backward()
            optimizee2.step()
            losses_adagardoff.append(loss.detach_())
            x_torch = x.detach_()


    fig = plt.figure()
    t = range(iters)
    plt.plot(t, loss_sgd, label='sgd')
    plt.plot(t, loss_sgdm, label='sgdm')
    plt.plot(t, loss_adgrad, label='adagrad')
    plt.plot(t, losses_adagardoff, label='adgrad_off')
    plt.plot(t, loss_rms, label='rms')
    # plt.plot(t, loss_nes, label='nes')
    plt.plot(t, loss_adam, label='adam')
    plt.plot(t, losses_adam_off,label='adma_off')
    plt.legend()
    # plt.savefig('loss_2dimen.jpg')
    plt.show()


    u = np.arange(-6, 6, 0.1)
    x, y = np.meshgrid(u, u)
    zz = Himmelblau([x, y])
    fig, ax = plt.subplots()

    xdata, ydata = [], []
    plt.contourf(x, y, zz, 100)
    # plt.contour(x, y, zz, 10, colors='black', linewidth=.6)
    C = plt.contour(x, y, zz, 10, colors='black', linewidth=.2)
    plt.clabel(C, inline=True, fontsize=6)
    plt.xticks(())
    plt.yticks(())

    plt.ion()
    plt.show()
    for t in range(iters):
        plt.scatter(x_sgd[t][0], x_sgd[t][1], c='purple', s=8)
        plt.scatter(x_rms[t][0], x_rms[t][1], c='pink', s=8)
        plt.scatter(x_adagradoff[t][0], x_adagradoff[t][1], c='white', s=8)
        plt.scatter(x_sgdm[t][0],x_sgdm[t][1],c='red',s = 8)
        plt.scatter(x_adagrad[t][0], x_adagrad[t][1],c='blue',s=8)
        # plt.scatter(x_nes[t][0], x_nes[t][1],c='green',s=8)
        plt.scatter(x_adam[t][0], x_adam[t][1],c='black',s=8)
        plt.scatter(x_adam_off[t][0], x_adam_off[t][1], c='orange', s=8)
        # plt.text(-5.,-5, 't= %.1f' % t, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.000000002)
        # print(t)

    plt.ioff()
    plt.show()



if __name__ == '__main__':
     plot()




import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# from gif_plot import *

def goal(w,x,y):

    return torch.mean(torch.sum((torch.matmul(w,x.unsqueeze(-1)).squeeze()-y)**2,dim=1),dim=0)

class LSTM_FORMULA(nn.Module):
    def __init__(self,input_size, hidden_num, layers, output_size,p,batch_size,process_flag=True):
        super(LSTM_FORMULA, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_num = hidden_num
        self.layers = layers
        self.batch_size = batch_size
        self.p = p
        self.process_flag = process_flag
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_num,
            num_layers=layers,
            batch_first=True)
        self.linear = nn.Linear(hidden_num, output_size)

    def lstm_optim(self, gradient, state):

        update,state = self.rnn(gradient,state)
        update = self.linear(update)
        return update.squeeze(), state

    def process(self, x):
            p  = self.p
            log = torch.log(torch.abs(x))
            clamp_log = torch.clamp(log/p , min = -1.0,max = 1.0)
            clamp_sign = torch.clamp(torch.tensor(np.exp(p))*x, min = -1.0, max =1.0)
            return torch.cat((clamp_log,clamp_sign), dim = -1)
    def forward(self, x, state):
        x = x.unsqueeze(0)
        if torch.cuda.is_available():
            if state is None:

                state = (torch.zeros((self.layers, self.batch_size, self.hidden_num)).cuda(),
                         torch.zeros((self.layers, self.batch_size, self.hidden_num)).cuda()
                         )
            x = x.cuda()
        else:
                state = (torch.zeros((self.layers, self.batch_size, self.hidden_num)),
                         torch.zeros((self.layers, self.batch_size, self.hidden_num))
                         )
        if self.process_flag:
            x = self.process(x)
        update, state = self.lstm_optim(x, state)

        return update, state

class meta_learning():
    def __init__(self, optimizer, input_size, output_size, layers, batch_size, hidden_num, global_iters, inner_iters,test_iters):
        self.layers = layers
        self.batch_size = batch_size
        self.hidden_num = hidden_num
        self.input_size = input_size
        self.output_size = output_size
        self.inner_iters = inner_iters
        self.global_iters = global_iters
        self.test_iters = test_iters
        self.w = None
        self.x = None
        self.y = None
        self.LSTM = optimizer
    def init_parameters(self):
        self.w = torch.randn(self.batch_size, 10, 10, requires_grad=False)
        self.x = torch.zeros(self.batch_size, 10, requires_grad=True)
        self.y = torch.randn(self.batch_size, 10, requires_grad=False)
        if torch.cuda.is_available():
            self.w = self.w.cuda()
            self.x = self.x.cuda()
            self.y = self.y.cuda()


    def reset_parameters(self):

        self.w = torch.randn(self.batch_size, 10,10,requires_grad=False)
        self.y = torch.randn(self.batch_size, 10,requires_grad=False)
        self.x = torch.zeros(self.batch_size, 10,requires_grad=True)
        if torch.cuda.is_available():
            self.w = self.w.cuda()
            self.x = self.x.cuda()
            self.y = self.y.cuda()

    def train_optim(self, optim, iters, reset_data=False):
        # if reset_data:
        #     length = 2
        #     tmp = torch.empty(length)
        #     torch.nn.init.uniform_(tmp, a=-1, b=1)
        #     w = torch.tensor(tmp, dtype=torch.float32, requires_grad=True)
        #     self.x = w
        #     # x = torch.tensor([1.0, -4.0], requires_grad=True)
        # else:
        #     x = x
        loss_sum = 0
        state = None
        losses = []
        x = self.x
        for t in range(iters):
            x.retain_grad()
            loss = goal(self.w, x, self.y)

            loss_sum += loss
            losses.append(loss)

            loss.backward(retain_graph=True)

            update, state = optim(x.grad, state)
            x = x + update
        return losses, loss_sum

    def lstm_optim(self, gradient, state):
        if state is None:
            state = (torch.zeros((self.layers, self.batch_size, self.hidden_num)).cuda(),
                     torch.zeros((self.layers, self.batch_size, self.hidden_num)).cuda()
                     )
        update, state = self.LSTM(gradient, state)
        return update, state

    def total_train(self):
        losses_sum = []
        self.init_parameters()
        _, loss_sum = self.train_optim(self.lstm_optim, self.inner_iters, reset_data=False)
        print(loss_sum)
        optimizer = torch.optim.Adam([{'params': self.LSTM.parameters()}],
                                     lr=0.01)
        for i in range(self.global_iters):
            if i == 0:
                self.init_parameters()
            else:
                self.reset_parameters()

            optimizer.zero_grad()
            _, loss_sum = self.train_optim(self.lstm_optim, self.inner_iters, reset_data=False)
            loss_sum.backward()
            optimizer.step()
            losses_sum.append(loss_sum)
            print(i, loss_sum)
        torch.save(self.LSTM, 'model/lstm_final.pkl')
        return losses_sum

    def meta_test(self):
        self.init_parameters()
        losses, loss_sum = self.train_optim(self.lstm_optim, self.test_iters, reset_data=False)
        return losses


def train_(input_size, output_size, layers, batch_size, hidden_num, global_iters, inner_iters, test_iters):
    optimizer = LSTM_FORMULA(input_size=input_size * 2,
                 hidden_num=hidden_num,
                 layers=layers,
                 output_size=output_size,
                 p=10, batch_size=batch_size)
    if torch.cuda.is_available():
        optimizer = optimizer.cuda()
    Meta = meta_learning(optimizer, input_size, output_size, layers, batch_size, hidden_num, global_iters, inner_iters,test_iters)
    losses = Meta.total_train()
    fig = plt.figure()
    plt.plot(range(global_iters), losses, label='lstm_train_losssum')
    plt.legend(loc='best')
    plt.savefig('result/lstm_train_losssum_{}_iters.jpg'.format(global_iters))
    plt.show()

def test_(input_size, output_size, layers, batch_size, hidden_num, global_iters, inner_iters, test_iters):
    optimizer = torch.load('model/lstm_final.pkl')
    if torch.cuda.is_available():
        optimizer = optimizer.cuda()
    meta = meta_learning(optimizer, input_size, output_size, layers, batch_size, hidden_num, global_iters, inner_iters,test_iters)
    losses = meta.meta_test()
    fig = plt.figure()
    plt.plot(range(test_iters), losses, label='lstm_train_losssum')
    plt.legend(loc='best')
    plt.savefig('result/lstm_train_losssum_{}_iters.jpg'.format(test_iters))
    plt.show()


def main():
    input_size = 10
    output_size = 10
    layers = 2
    batch_size = 128
    hidden_num = 20
    global_iters = 200
    inner_iters = 20
    test_iters = 50
    train_(input_size, output_size, layers, batch_size, hidden_num, global_iters, inner_iters, test_iters)

    test_(input_size, output_size, layers, batch_size, hidden_num, global_iters, inner_iters, test_iters)


if __name__ == '__main__':
    main()


