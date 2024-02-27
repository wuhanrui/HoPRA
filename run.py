from torch import nn, optim
from model import HoPRA
from utils import seed_everything, dotdict, calculate_f1, load_target, load_source, load_H
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import pandas as pd

def recon_loss(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

def training(data, args):

    seed_everything(2023)

    s_norm = data.s_norm
    t_norm = data.t_norm
    sx = data.sx
    sy = data.sy
    sPPMI_ori = data.sPPMI_ori
    sPPMI_norm = data.sPPMI_norm
    s_pos_weight = data.s_pos_weight
    H_ori = data.H_ori
    H_norm = data.H_norm
    Ht_norm = data.Ht_norm
    tx = data.tx
    ty_all = data.ty
    tx_train = data.tx_train
    t_train_idx = data.t_train_idx
    t_test_idx = data.t_test_idx
    ty_trian = ty_all[data.t_train_idx]
    ty_test = ty_all[data.t_test_idx]
    tPPMI_ori = data.tPPMI_ori
    tPPMI_norm = data.tPPMI_norm
    t_pos_weight = data.t_pos_weight
    domain_y = torch.cat([torch.ones(sx.shape[0]), torch.zeros(tx_train.shape[0])])
    nclass = int((torch.max(data.sy) + 1).numpy())
    dim_l2 = int(args.hidden_dim / 2)

    model = HoPRA(sx.shape[1], tx.shape[1], args.hidden_dim, dim_l2, nclass)
    if cuda:
        sx = sx.cuda()
        sy = sy.cuda()
        sPPMI_ori = sPPMI_ori.cuda()
        sPPMI_norm = sPPMI_norm.cuda()
        s_pos_weight = s_pos_weight.cuda()
        H_ori = H_ori.cuda()
        H_norm = H_norm.cuda()
        Ht_norm = Ht_norm.cuda()
        tx = tx.cuda()
        tx_train = tx_train.cuda()
        t_test_idx = t_test_idx.cuda()
        ty_trian = ty_trian.cuda()
        ty_test = ty_test.cuda()
        tPPMI_ori = tPPMI_ori.cuda()
        tPPMI_norm = tPPMI_norm.cuda()
        t_pos_weight = t_pos_weight.cuda()
        domain_y = domain_y.cuda()
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    floss = nn.MSELoss()
    hco_loss = nn.BCELoss()


    for epoch in range(args.epochs):

        model.train()

        H_reco, x_output, xs2, xtl2, sxp2, sxp3, sp_reco, txp2, txp3, tp_reco, disc = model(H_norm, Ht_norm, sx, tx_train, tx, sPPMI_norm, tPPMI_norm, t_test_idx)

        # node classification loss
        l_label = torch.cat((sy, ty_trian), 0)
        loss1 = F.nll_loss(x_output[0:(sx.shape[0] + tx_train.shape[0]), :], l_label)
        # hypergraph reconstruction loss
        loss2 = hco_loss(H_reco, H_ori)
        # source network reconstruction loss
        loss3 = recon_loss(sp_reco, sPPMI_ori, sxp2, sxp3, sPPMI_ori.shape[0], s_norm, s_pos_weight)
        # target network reconstruction loss
        loss4 = recon_loss(tp_reco, tPPMI_ori, txp2, txp3, tPPMI_ori.shape[0], t_norm, t_pos_weight)
        # feature matching loss
        loss5 = floss(xs2, sxp2)
        loss6 = floss(xtl2, txp2[t_train_idx])
        # Domain classification loss
        loss7 = F.binary_cross_entropy_with_logits(disc.squeeze(), domain_y)

        loss = loss1 + args.alpha*loss2 + args.beta*(loss3 + loss4) + args.eta*(loss5 + loss6) + args.gamma*loss7

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.verb == 1:
            print("Epoch:", '%04d|' % (epoch + 1),
                  "loss1: ", '%f6|' % loss1.item(),
                  "loss2: ", '%f6|' % loss2.item(),
                  "loss3: ", '%f6|' % loss3.item(),
                  "loss4: ", '%f6|' % loss4.item(),
                  "loss5: ", '%f6|' % loss5.item(),
                  "loss6: ", '%f6|' % loss6.item(),
                  "loss7: ", '%f6|' % loss7.item(),
                  "loss: ", '%f6|' % loss.item())

    with torch.no_grad():
        model.eval()
        H_reco, x_output, xs2, xtl2, sxp2, sxp3, sp_reco, txp2, txp3, tp_reco, disc = model(H_norm, Ht_norm, sx, tx_train, tx, sPPMI_norm, tPPMI_norm, t_test_idx)
        pred = x_output[(sx.shape[0] + tx_train.shape[0]):,:]
        tf1_micro, tf1_macro = calculate_f1(pred.detach().cpu().numpy(), ty_test.detach().cpu().numpy())

    return tf1_micro, tf1_macro


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CNNC')
    parser.add_argument('--gpu', type=int, nargs='?', default=-1, help="gpu id")
    parser.add_argument('--source', type=str, nargs='?', default='acmv9', help="source network")
    parser.add_argument('--target', type=str, nargs='?', default='dblpv7', help="target network")
    parser.add_argument('--verb', type=int, nargs='?', default=1)
    setting = parser.parse_args()

    if setting.gpu >= 0:
        print('Training on GPU')
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(setting.gpu)
        cuda = True
    else:
        print('Training on CPU')
        device = torch.device('cpu')
        cuda = False

    print(setting)

    # fixed
    dim_hidden = 512
    learning_rate = 0.001
    weight_decay = 0
    epochs = 200
    if setting.source == 'acmv9' and setting.target == 'dblpv7':
        alpha = 0.01
        beta = 0.01
        eta = 0.01
        gamma = 1

    micro_list = []
    macro_list = []
    sx, sy, sPPMI_ori, sPPMI_norm, s_norm, s_pos_weight = load_source(setting.source)
    tx, ty, tPPMI_ori, tPPMI_norm, t_norm, t_pos_weight, train_list, test_list = load_target(setting.target)

    for trial in range(train_list.shape[0]):
        print(trial)
        t_train_idx = train_list[trial]
        t_test_idx = test_list[trial]
        H_ori, H_norm, Ht_norm = load_H(sy, ty[t_train_idx])
        data = dotdict()
        args = dotdict()
        data.sx = sx
        data.sy = torch.LongTensor(sy)
        data.sPPMI_ori = sPPMI_ori
        data.sPPMI_norm = sPPMI_norm
        data.s_norm = s_norm
        data.s_pos_weight = s_pos_weight
        data.H_ori = H_ori
        data.H_norm = H_norm
        data.Ht_norm = Ht_norm
        data.tx = tx
        data.ty = torch.LongTensor(ty)
        data.tx_train = tx[t_train_idx]
        data.t_train_idx = torch.LongTensor(t_train_idx)
        data.t_test_idx = torch.LongTensor(t_test_idx)
        data.tPPMI_ori = tPPMI_ori
        data.tPPMI_norm = tPPMI_norm
        data.t_norm = t_norm
        data.t_pos_weight = t_pos_weight
        args.hidden_dim = dim_hidden
        args.learning_rate = learning_rate
        args.weight_decay = weight_decay
        args.alpha = alpha
        args.beta = beta
        args.eta = eta
        args.gamma = gamma
        args.epochs = epochs
        args.verb = setting.verb
        tf1_micro, tf1_macro = training(data, args)
        micro_list.append(tf1_micro)
        macro_list.append(tf1_macro)

    mean_micro = np.mean(micro_list)
    std_micro = np.std(macro_list)

    mean_macro = np.mean(macro_list)
    std_macro = np.std(macro_list)

    print("F1-Micro: {:.4f}({:.4f})".format(mean_micro, std_micro))
    print("F1-Macro: {:.4f}({:.4f})".format(mean_macro, std_macro))





