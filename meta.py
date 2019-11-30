import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gif_plot import *
import os
import json

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
            input_size = input_size,
            hidden_size = hidden_num,
            num_layers = layers)
        self.linear = nn.Linear(hidden_num, output_size)

    def lstm_optim(self, gradient, state):
        gradient = gradient.unsqueeze(0)
        update,state = self.rnn(gradient,state)
        update = self.linear(update)
        return update.squeeze().squeeze(), state

    def process(self, x):
            if np.exp(-self.p)<= torch.max(torch.sum(torch.abs(x),dim=-1)):
                gradient_sgn = torch.where(x<0, torch.full_like(x,-1), torch.full_like(x,1))
                gradient_log = torch.clamp(torch.log(torch.abs(x))/self.p,min=-1,max=1)
                return torch.cat((gradient_log,gradient_sgn), dim=-1)
            else:
                gradient_exp = torch.clamp(np.exp(self.p)*self.gradient,min=-1,max=1)
                return torch.cat((-1*torch.from_numpy(np.ones_like(x)), gradient_exp),dim=-1)
        # p  = self.p
        # log = torch.log(torch.abs(x))
        # clamp_log = torch.clamp(log/p , min = -1.0,max = 1.0)
        # clamp_sign = torch.clamp(np.exp(p)*x, min = -1.0, max =1.0)
        # return torch.cat((clamp_log,clamp_sign),dim = -1) #在gradients的最后一维input_dims拼接

    def forward(self, x, state):
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
    def __init__(self, optimizer, input_size, output_size, layers, batch_size, hidden_num, global_iters, outer_iters, inner_iters,test_iters):
        self.layers = layers
        self.batch_size = batch_size
        self.hidden_num = hidden_num
        self.input_size = input_size
        self.output_size = output_size
        self.inner_iters = inner_iters
        self.outer_iters = outer_iters
        self.global_iters = global_iters
        self.test_iters = test_iters
        self.w = None
        self.x = None
        self.y = None
        self.state =None
        self.LSTM = optimizer

    def init_parameters(self):
        self.w = torch.randn(self.batch_size, self.input_size, self.input_size, requires_grad=False)
        self.x = torch.zeros(self.batch_size, self.input_size, requires_grad=True)
        self.y = torch.randn(self.batch_size, self.output_size, requires_grad=False)
        self.state = None
        if torch.cuda.is_available():
            self.w = self.w.cuda()
            self.x = self.x.cuda()
            self.y = self.y.cuda()


    def reset_parameters(self):
        self.w = torch.randn(self.batch_size, self.input_size, self.input_size, requires_grad=False)
        self.x = torch.zeros(self.batch_size, self.input_size, requires_grad=True)
        self.y = torch.randn(self.batch_size, self.output_size, requires_grad=False)
        # self.state = None
        if torch.cuda.is_available():
            self.w = self.w.cuda()
            self.x = self.x.cuda()
            self.x.retain_grad()
            self.y = self.y.cuda()

    def train_optim(self,optim, iters):
        loss_sum = 0

        # state = None
        state = self.state
        losses = []

        x = self.x
        for t in range(iters):   # 20

            x.retain_grad()
            loss = goal(self.w, x, self.y)

            loss_sum += loss
            losses.append(loss)

            loss.backward(retain_graph=True)

            update, state = optim(x.grad, state)
            x = x + update
            x.retain_grad()

        if state is not None:
                self.state = (state[0].detach(), state[1].detach())

        return losses, loss_sum

    def lstm_optim(self, gradient, state):
        if state is None:
            state = (torch.zeros((self.layers, self.batch_size, self.hidden_num)).cuda(),
                     torch.zeros((self.layers, self.batch_size, self.hidden_num)).cuda()
                     )
        update, state = self.LSTM(gradient, state)
        return update, state

    def evaluate(self, optimizer_val, pre_iter):
        with open('result/best_loss_sum.json', 'r') as f:
            loss_sum_json = json.load(f)
        f.close()
        val_w = torch.randn(self.batch_size, self.input_size, self.input_size, requires_grad=False)
        val_x = torch.zeros(self.batch_size, self.input_size, requires_grad=True)
        val_y = torch.randn(self.batch_size, self.output_size, requires_grad=False)
        if torch.cuda.is_available():
            val_w = val_w.cuda()
            val_x = val_x.cuda()
            val_y = val_y.cuda()
        loss_sum_val = 0
        state = None

        if state is None:
            if torch.cuda.is_available():
                state = (torch.zeros((self.layers, self.batch_size, self.hidden_num)).cuda(),
                     torch.zeros((self.layers, self.batch_size, self.hidden_num)).cuda()
                     )
            else:
                state = (torch.zeros((self.layers, self.batch_size, self.hidden_num)),
                         torch.zeros((self.layers, self.batch_size, self.hidden_num))
                         )
        losses = []
        import copy
        optimizer_val = copy.deepcopy(optimizer_val)
        for t in range(self.inner_iters):  # 20
            val_x.retain_grad()
            loss = goal(val_w, val_x, val_y)
            loss_sum_val += loss
            losses.append(loss)
            loss.backward(retain_graph=True)
            update, state = optimizer_val(val_x.grad, state)
            val_x = val_x + update

        if loss_sum_val < loss_sum_json['best_loss_sum']:
            best_loss_sum = float(loss_sum_val.detach_().cpu())
            best_global_iter = pre_iter
            loss_sum_json['best_loss_sum'] = best_loss_sum
            loss_sum_json['best_global_iter'] = best_global_iter
            new_record = json.dumps(loss_sum_json, indent=4)
            with open('result/best_loss_sum.json', 'w') as json_file:
                json_file.write(new_record)
                path = 'model/lstm_final_'+str(loss_sum_json['best_global_iter'])+'.pkl'
                torch.save(optimizer_val, path)

        return loss_sum_json['best_loss_sum'], loss_sum_json['best_global_iter']


    def total_train(self):
        with open('result/best_loss_sum.json', 'w+') as json_file:
            global_loss_record = {}
            global_loss_record['best_loss_sum'] = 9999999
            global_loss_record['best_global_iter'] = 0
            new_best_record = json.dumps(global_loss_record, indent=4)
            json_file.write(new_best_record)


        losses_sum = []
        self.init_parameters()
        _, loss_sum = self.train_optim(self.lstm_optim, self.inner_iters)
        print(loss_sum)

        optimizer = torch.optim.Adam([{'params': self.LSTM.parameters()}],
                                     lr=0.01)
        iter_for_reset = self.outer_iters // self.inner_iters
        loss_sum = 999999
        for i in range(self.global_iters):
              print("global inters:\t{} loss_sum:\t{}".format(i,loss_sum))
              self.init_parameters()
              for m in range(iter_for_reset):  # 5
                    if m == 0:
                        self.reset_parameters()
                    if self.state is None:
                        print('state reset...')

                    optimizer.zero_grad()
                    _, loss_sum = self.train_optim(self.lstm_optim, self.inner_iters)
                    loss_sum.backward()
                    optimizer.step()
                    losses_sum.append(loss_sum)
                    print("reset iter:{}\t loss:{}".format(m, loss_sum))
              if i% 20 == 0:  # evaluate:
                  best_loss_sum, best_global_iter = self.evaluate(self.LSTM, i)
              print('best_loss_sum:\t{}, best_global_iter:{}'.format(best_loss_sum,  best_global_iter))
        # torch.save(self.LSTM,'model/ok_my200.pkl')
        return losses_sum

    def meta_test(self,x,w,y):
        # self.init_parameters()
        self.x = x
        self.w = w
        self.y = y
        losses, loss_sum = self.train_optim(self.lstm_optim, self.test_iters)
        return losses,loss_sum

def train_(input_size, output_size, layers, batch_size, hidden_num, global_iters, outer_iters, inner_iters, test_iters):
    optimizer = LSTM_FORMULA(input_size=input_size * 2,
                 hidden_num=hidden_num,
                 layers=layers,
                 output_size=output_size,
                 p=10, batch_size=batch_size)
    if torch.cuda.is_available():
        optimizer = optimizer.cuda()
    Meta = meta_learning(optimizer, input_size, output_size, layers, batch_size, hidden_num, global_iters, outer_iters, inner_iters,test_iters)
    losses = Meta.total_train()
    # fig = plt.figure()
    # plt.plot(range(global_iters*(outer_iters//inner_iters)), losses, label='lstm_train_losssum')
    # plt.legend(loc='best')
    # plt.savefig('result/lstm_train_losssum_{}_iters.jpg'.format(global_iters))
    # plt.show()

def test_(input_size, output_size, layers, batch_size, hidden_num, global_iters, outer_iters,inner_iters, test_iters):
    with open('result/best_loss_sum.json', 'r') as json_file:
        model_info = json.load(json_file)
    optimizer = torch.load('model/lstm_final_{}.pkl'.format(model_info['best_global_iter']))
    # optimizer = LSTM_FORMULA(input_size*2,hidden_num,layers,output_size,10,batch_size)   # 随机初始化模型
    optimizer = torch.load('model/ok_my200.pkl').cuda()          # 1000 iters trained model
    x = torch.zeros(batch_size, input_size, requires_grad=True)
    w = torch.randn(batch_size, input_size, input_size, requires_grad=False)
    y = torch.randn(batch_size, output_size, requires_grad=False)
    if torch.cuda.is_available():
        optimizer = optimizer.cuda()
        x = x.cuda()
        w = w.cuda()
        y = y.cuda()

    meta = meta_learning(optimizer, input_size, output_size, layers, batch_size, hidden_num, global_iters, outer_iters, inner_iters,test_iters)
    losses, loss_sum = meta.meta_test(x,w,y)

    x_adam_off, losses_adam_off = train(goal, input_size, batch_size, w.clone(),x.clone(),y.clone(), test_iters, lr=0.01, arg='adamoff')

    x_sgd, loss_sgd = train(goal, input_size, batch_size,w.clone(),x.clone(),y.clone(), test_iters, lr=0.01, arg='sgd')

    x_sgdm, loss_sgdm = train(goal, input_size, batch_size, w.clone(),x.clone(),y.clone(), test_iters, lr=0.01, arg='sgdm')

    x_adagrad, loss_adgrad = train(goal, input_size, batch_size, w.clone(),x.clone(),y.clone(), test_iters, lr=0.01, arg='adagrad')

    x_rms, loss_rms = train(goal, input_size, batch_size, w.clone(),x.clone(),y.clone(), test_iters, lr=0.01, arg='rms')

    x_adam, loss_adam = train(goal, input_size, batch_size,w.clone(),x.clone(),y.clone(), test_iters, lr=0.01, arg='adam')


    fig = plt.figure()
    plt.plot(range(test_iters), losses, label='lstm')
    plt.plot(range(test_iters), losses_adam_off, label='adam_off')
    plt.plot(range(test_iters), loss_sgd, label='sgd')
    plt.plot(range(test_iters), loss_sgdm, label='agdm')
    plt.plot(range(test_iters), loss_adgrad, label='adagrad')
    plt.plot(range(test_iters), loss_rms, label='rms')
    plt.plot(range(test_iters), loss_adam, label='adam')
    plt.legend(loc='best')
    plt.savefig('result/test_losssum_{}_itersmy.jpg'.format(test_iters))
    plt.show()


def main():
    input_size = 10
    output_size = 10
    layers = 2
    batch_size = 128
    hidden_num = 20
    global_iters = 6
    outer_iters = 100
    inner_iters = 20
    test_iters = 50

    train_(input_size, output_size, layers, batch_size, hidden_num, global_iters, outer_iters, inner_iters, test_iters)
    test_(input_size, output_size, layers, batch_size, hidden_num, global_iters, outer_iters, inner_iters, test_iters)


if __name__ == '__main__':
    main()

