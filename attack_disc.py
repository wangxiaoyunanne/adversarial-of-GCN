import sys

import argparse

import torch

import torch.nn as nn

from torch.autograd import Variable

import torch.optim as optim

import torchvision.transforms as tfs

import torchvision.datasets as dst

from torch.utils.data import DataLoader

import numpy as np

from models import GCN

import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, accuracy
import time

def train(epoch):
    t = time.time()
    net.train()
    optimizer.zero_grad()
    output = net(features, adj)
    #print (output)
    #output = Variable(output)
    output = F.log_softmax(output, dim =1)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
    #net.eval()
    #output = net(features, adj)
    #output = F.log_softmax(output, dim =1)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data[0]),
          'acc_train: {:.4f}'.format(acc_train.data[0]),
          'loss_val: {:.4f}'.format(loss_val.data[0]),
          'acc_val: {:.4f}'.format(acc_val.data[0]),
          'time: {:.4f}s'.format(time.time() - t))


def get_index ( grad,w_v ) :
    # change all grad to 0 with adj matrix element = 1
    # in adj matrix change 0 to 1 and 1 to 0
    adj_rev = 1 - torch.ge(w_v, 0.5) 
    adj_rev = adj_rev.float().cuda()
    grad = grad * adj_rev 
    # get index of max element in tensor
    val_col, ind_col = torch.max(grad, 1)
    val_row, ind_row = torch.max(val_col,0)
    max_row = ind_row
    max_col = ind_col[ind_row]
    return max_row, max_col

def change_adj (max_row,max_col,w_v):
    # change the largest vaule to 1
    a = np.zeros ((w_v.size()[0], w_v.size()[1]))
    a[max_row][max_col] = 1
    a = torch.from_numpy(a)
    a = a.float().cuda()
    w_v = w_v + a
    return w_v
          

def attack_cw(feat_v,input_v, label_v, net, c, untarget=True, n_class=7):

    net.eval()
    #output = net(feat_v, input_v)
    #output = F.log_softmax(output, dim =1)
    #loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    #acc_test = accuracy(output[idx_test], labels[idx_test])
    #print("cwTest set results:",
    #      "loss= {:.4f}".format(loss_test.data[0]),
    #      "accuracy= {:.4f}".format(acc_test.data[0]))


    index = label_v.data.cpu().view(-1, 1)

    label_onehot = torch.FloatTensor(input_v.size()[0], n_class)

    label_onehot.zero_()

    label_onehot.scatter_(1, index, 1)

    label_onehot_v = Variable(label_onehot, requires_grad = False).cuda()
   # print (label_onehot_v.data)
    w = input_v.data.clone()
    w_v = Variable(w.cuda(), requires_grad=True)
    #w_v = input_v
    optimizer = optim.Adam([w_v], lr=1.0e-3)

    zero_v = Variable(torch.FloatTensor([0]).cuda(), requires_grad=False)

    for _ in range(20):

        #net.zero_grad()

        optimizer.zero_grad()
        #print 'w_v grad'
        #print(w_v.grad)
        #adverse_v = 0.5 * (torch.tanh(w_v) + 1.0)
        adverse_v = w_v
        diff = adverse_v - input_v
        #print torch.sum((diff))
        output = net(feat_v,adverse_v)
        output1 = F.log_softmax(output, dim =1)
        loss_test = F.nll_loss(output1[idx_test], labels[idx_test])
        acc_test = accuracy(output1[idx_test], labels[idx_test])
        print("cwTest set results:", "loss= {:.4f}".format(loss_test.data[0]),
        "acc = {:.4f}".format(acc_test.data[0]))
    
        real = (torch.max(torch.mul(output, label_onehot_v), 1)[0])
     #   print real
        other = (torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0])
        
        error = torch.sum(diff * diff)
        print error 
        if untarget:

            error += c * torch.sum(torch.max(real - other, zero_v))
            
        else:

            error += c * torch.sum(torch.max(other - real, zero_v))

        error.backward()
        print error 
        #optimizer.step()
        print w_v.grad
        # get the biggest grad then change to 1
        ind_i, ind_j = get_index(w_v_grad, w_v)
        w_v = change_adj(ind_i, ind_j, w_v)
    return adverse_v, diff

def acc_under_attack(feat_v,input_v,label_v, net, c, attack_f):

    correct = 0

    tot = 0

    distort = 0.0
    #input_v = adj_v
    #for k, (input, output) in enumerate(dataloader):

    #    input_v, label_v = Variable(input.cuda()), Variable(output.cuda())

        # attack

    if c != 0:

        adverse_v, diff = attack_f(feat_v,input_v, label_v, net, c)

    else:

        adverse_v = input_v

        diff = Variable(torch.FloatTensor([0]))

        # defense

    net.eval()

    adverse_v = Variable(adverse_v.data, volatile=True)

    _, idx = torch.max(net(feat_v,adverse_v), 1)

    correct += torch.sum(label_v.eq(idx)).data[0]

    tot += label_v.numel()

    distort += torch.sum(diff.data * diff.data)
    print 'distort' + str(distort)
    return correct / tot, np.sqrt(distort / tot)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

    parser.add_argument('--dataset', type=str, required=True)

    parser.add_argument('--net', type=str, required=True)

   # parser.add_argument('--defense', type=str, required=True)

    parser.add_argument('--modelIn', type=str, required=True)

    parser.add_argument('--c', type=str, default='1.0')

#    parser.add_argument('--root', type=str, required=True)

    opt = parser.parse_args()

    # parse c

    opt.c = [float(c) for c in opt.c.split(',')]

    if opt.dataset == 'cora':
        adj, features, labels, idx_train, idx_val, idx_test = load_data()

    else:

        print("Invalid dataset")

        exit(-1)

    print("#c, test accuracy")


    if opt.net == "GCN" or opt.net == "gcn":

        #print (labels.max() , features.shape[1])
        net = GCN(nfeat=features.shape[1],
            nhid=16,
            nclass=labels.max() + 1,
            dropout=0.5
         )

        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        features, adj, labels = Variable(features), Variable(adj), Variable(labels)


    #net = nn.DataParallel(net, device_ids=range(1))

    #net.load_state_dict(torch.load(opt.modelIn))
    optimizer = optim.Adam(net.parameters(),
                       lr=0.01, weight_decay=5e-4)
    net.cuda()
    for epoch_n in range(200):
        train(epoch_n)
    loss_f = nn.CrossEntropyLoss()

    output = net(features, adj)
    output = F.log_softmax(output, dim =1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))

    for c in opt.c:
        #print (type(adj), type(features))
        acc, avg_distort = acc_under_attack(features,adj,labels, net, c, attack_cw)

        print("{}, {}, {}".format(c, acc, avg_distort))

        sys.stdout.flush()




