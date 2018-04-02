# attack feature matrix by dropping/adding element 
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
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data[0]),
          'acc_train: {:.4f}'.format(acc_train.data[0]),
          'loss_val: {:.4f}'.format(loss_val.data[0]),
          'acc_val: {:.4f}'.format(acc_val.data[0]),
          'time: {:.4f}s'.format(time.time() - t))


# using new method to change the edge to be added
          
def get_index ( grad,w_v, modi ) :
    
    #grad_abs = torch.abs(grad)  
    modi = Variable(modi)
    grad = grad * modi
    # get adjust max element 
    grad_max = grad * w_v
    # get index of max element in tensor
    val_col, ind_col = torch.max(grad_max, 1)
    val_row, ind_row = torch.max(val_col,0)
    max_row = ind_row.data
    max_col = ind_col[ind_row].data
    # get adjusted min element
    nnz_row = torch.eye (w_v.size()[0] )
    for row in range (w_v.size() [0] ) :
        ind_nonz = torch.nonzero (w_v[row])  
        sum_row = torch.sum(w_v[row]) 
        if sum_row.data[0]  == 1.0 :
            nnz_row [row] [row] = 1.0
        else : 
            dim = ind_nonz.size()[0]
            nnz_row [row] [row] = 1.0 / dim

    nnz_row = nnz_row.cuda()
    nnz_row = Variable(nnz_row)
    adj_ge = torch.ge(w_v, 0.0001)
    adj_ge = adj_ge.float().cuda()
    #print torch.diag(adj_ge)
    adj_rev = 1.0 - adj_ge
    adj_rev = adj_rev.float().cuda()
    grad_min = grad * adj_rev
    grad_min = torch.mm ( nnz_row, grad_min )
    # get index of max element in tensor
    val_col, ind_col = torch.min(grad_min, 1)
    val_row, ind_row = torch.min(val_col,0)
    min_row = ind_row.data
    min_col = ind_col[ind_row].data
    ratio = torch.min (grad_min) / torch.max(grad_max)
    if  ratio.data[0] < -5.1 :
    # add element 
        return min_row[0], min_col[0]
    else :
    #delete element
        return max_row[0], max_col[0]

def get_index_random (grad,w_v):
    row, col = np.random.randint(w_v.size()[0],size = 2)
    return row,col

def add_edge (row, col, w_v):
    a = np.zeros ((w_v.size()[0], w_v.size()[1]))
    ind_nonz = torch.nonzero (w_v[row]) 
    dim = ind_nonz.size()[0]
    for j in range(dim):
        ind_col = ind_nonz[j].data
        #print w_v[row , ind_col]
        a [row, ind_col] = 1.0/(dim+1) - 1.0/dim
    print "dim", dim
    a [row,col] = 1.0/(dim+1)
    
    a = torch.from_numpy(a)
    a = Variable (a.float().cuda())
   # print (dim ,torch.sum(a))
    return a 

def drop_edge (row, col, w_v):
    a = np.zeros ((w_v.size()[0], w_v.size()[1]))
    ind_nonz = torch.nonzero (w_v[row])
    dim = ind_nonz.size()[0]
    print "dim " + str(dim)
    if dim == 1 :
        a [row, col] = 1.0/dim
    else :
        for j in range(dim):
            ind_col = ind_nonz[j].data
        #print w_v[row , ind_col]
            a [row, ind_col] = -1.0/(dim-1) + 1.0/dim
            a[row,col] = 1.0/dim

    a = torch.from_numpy(a)
    a = Variable (a.float().cuda())
   # print (dim ,torch.sum(a))
    return a

          

def attack_cw(feat_v,input_v, label_v, net, c, untarget=True, n_class=7):

    net.eval()
    
    index = label_v.data.cpu().view(-1, 1)

    label_onehot = torch.FloatTensor(input_v.size()[0], n_class)

    label_onehot.zero_()

    label_onehot.scatter_(1, index, 1)

    label_onehot_v = Variable(label_onehot, requires_grad = False).cuda()
   # print (label_onehot_v.data)
    w = feat_v.data.clone()
    w_v = Variable(w.cuda(), requires_grad=True)
    #print w_v[2][3]
     
    #optimizer = optim.Adam([w_v], lr=1.0e-3)

    zero_v = Variable(torch.FloatTensor([0]).cuda(), requires_grad=False)
    num_edges = 0.0
    # get a matrix to note the localtion of changed element
    adj_mod = torch.FloatTensor (w_v.size()[0] ,w_v.size()[1] ).zero_()
    adj_mod += 1.0
    adj_mod = adj_mod.cuda()
    print w_v.size()
    for _ in range(100):

        net.zero_grad()
        num_edges += 1
        #optimizer.zero_grad()
        
        #adverse_v = 0.5 * (torch.tanh(w_v) + 1.0)
        adverse_v = w_v
        diff = adverse_v - feat_v
        #print torch.sum((diff))
        output = net(adverse_v, input_v)
        output1 = F.log_softmax(output, dim =1)
        loss_test = F.nll_loss(output1[idx_test], labels[idx_test])
        acc_test = accuracy(output1[idx_test], labels[idx_test])
        print("cwTest set results:", "loss= {:.4f}".format(loss_test.data[0]),
        "acc = {:.4f}".format(acc_test.data[0]))
    
        real = (torch.max(torch.mul(output, label_onehot_v), 1)[0])
     #   print real
        other = (torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0])
        
        #error = torch.sum(diff * diff)
        error = 0
        #print 'error 1' + str(error)
        if untarget:

            error += c * torch.sum(torch.max(real - other, zero_v))
            
        else:

            error += c * torch.sum(torch.max(other - real, zero_v))

        error.backward()
        
        ind_i, ind_j = get_index(w_v.grad, w_v, adj_mod)
        print ind_i, ind_j
        adj_mod [ind_i][ind_j] = 0
        #a = add_edge (ind_i, ind_j, w_v)
        val =  w_v.grad[ind_i][ ind_j].data        
        if val[0] > 0 :
            print "drop"
            a = drop_edge(ind_i,ind_j, w_v)
            w_v.data -= a.data
        else :
            print "add"
            a = add_edge (ind_i, ind_j, w_v)  
            w_v.data += a.data 
        #w_v.data += b.data

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

        adverse_v = feat_v

        diff = Variable(torch.FloatTensor([0]))

        # defense

    net.eval()

    adverse_v = Variable(adverse_v.data, volatile=True)

    _, idx = torch.max(net(adverse_v, input_v), 1)

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
    net.eval()
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




