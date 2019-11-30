import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
def f(x):

    return (((x+1)*(x+0.5)*x*(x-1))**2).sum()
class optim():
    def __init__(self, batch_size, gradients, state, length, decay=0.90, learning_rate=0.01):

        self.grad = gradients
        self.lr = learning_rate
        self.decay = decay
        self.state = state
        self.len  = length
        self.batch_size = batch_size

    def SGD(self):
        return  -self.grad * self.lr

    def SGDM(self, tho=0.90):

        if self.state is None:
            if torch.cuda.is_available():
                self.state = torch.zeros(self.batch_size, self.len).cuda()
            else:
                self.state = torch.zeros(self.batch_size, self.len)

        self.state = tho*self.state - self.lr*self.grad
        update = self.state
        return  update, self.state

    def Nesterov_momentum(self,x,tho=0.7):
        if self.state is None:
            if torch.cuda.is_available():
                self.state = torch.zeros(self.batch_size, self.len).cuda()
            else:
                self.state = torch.zeros(self.batch_size, self.len)



        t =  Variable(x + tho*self.state, requires_grad =True)

        loss2 = f(t)
        loss2.backward()
        state2 = tho * self.state - self.lr * t.grad
        update = (1+tho)*state2 - tho*self.state
        t=t.detach()
        return  update, state2

    def Adagrad(self):
        if self.state is None:
            if torch.cuda.is_available():
                self.state = torch.zeros(self.batch_size, self.len).cuda()
            else:
                self.state = torch.zeros(self.batch_size, self.len)

        self.state+= torch.pow(self.grad,2)
        update = -self.lr*self.grad / (torch.sqrt(self.state)+ 1e-7)
        return update, self.state

    def RMS(self):
        if self.state is None:
            if torch.cuda.is_available():
                self.state = torch.zeros(self.batch_size, self.len).cuda()
            else:
                self.state = torch.zeros(self.batch_size, self.len)

        self.state = self.decay * self.state + (1-self.decay)* torch.pow(self.grad,2)
        update = -self.lr * self.grad / (torch.sqrt(self.state)+ 1e-7)
        return update,self.state

    def Adam(self, iters, tho1, tho2, state1, state2):
        if state1 is None:
            if torch.cuda.is_available():
                state1 = torch.zeros(self.batch_size, self.len).cuda()
            else:
                state1 = torch.zeros(self.batch_size, self.len)
        if state2 is None:
            if torch.cuda.is_available():
                state2 = torch.zeros(self.batch_size, self.len).cuda()
            else:
                state2 = torch.zeros(self.batch_size, self.len)

        state1 = tho1 * state1 + (1- tho1)* self.grad
        state2 = tho2 * state2 + (1- tho2)*torch.pow(self.grad, 2)

        state1_bias = state1 / (1- tho1**(iters+1))
        state2_bias = state2 / (1- tho2**(iters+1))

        update = -self.lr * state1_bias / (torch.sqrt(state2_bias)+1e-7)
        return update, state1, state2

