import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import torch
from optimizee import optim

def Himmelblau(x):
    '''
    Himmelblau function for optim-process show
    :return: loss
    '''
    return (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2

def goal2(w,x,y):
    '''
    a more complex loss function to optimize
    :return: ||w*x-y||^2_2
    '''
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
        '''
        adam optimizer in pytorch 
        '''
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
         '''
           self-defined optimizers 
         '''
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



