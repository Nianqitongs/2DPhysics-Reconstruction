# 院校：南京信息工程大学
# 院系：自动化学院
# 开发时间：2023/11/14 13:07
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
import pickle
from test import show_result,show_train_hist
from utils.utils import seed_torch,d_loss_func,g_loss_func
from model.model import Generator,Discriminator
from data_proprecess.data_proprecess import KeyDataset,read_npy_data,create_data

path_data = 'E:/github/my_projects/2DPhysics_reconstruction/data/reconstruct.npy'
path_label = 'E:/github/my_projects/2DPhysics_reconstruction/data/ut.npy'
device = torch.device('cuda:0')
seed_torch()

def init_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--epochs", default=200)
    parser.add_argument('--lr_G', default=0.0005, type=float)
    parser.add_argument('--lr_D', default=0.00005, type=float)
    parser.add_argument("--log_dir", default="/data1/zzd/TransBTS/logdir_TransBTS")
    return parser.parse_args()


def trainer(G, D, train_data, valid_data, args):
    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=0.0005)
    D_optimizer = optim.Adam(D.parameters(), lr=0.00005)

    # results save folder
    root = 'output/'
    weights = 'weights/'
    predicate = 'predicate/'
    label = 'label/'
    hist = 'hist/'
    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(root + 'generate_results'):
        os.mkdir(root + 'generate_results')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    print('training start!')
    start_time = time.time()

    G.train()
    D.train()
    for epoch in range(args.epochs):
        D_losses = []
        G_losses = []

        epoch_start_time = time.time()

        t = DataLoader(KeyDataset(train_data), batch_size=args.batch_size, shuffle=False, drop_last=True)
        for i, data in enumerate(t):
            inp, target = data["input"].float().to(device), data["target"].float().to(device)  # (128,1,64,64),(128,2)
            # train discriminator D
            D_optimizer.zero_grad()
            mini_batch = inp.size()[0]
            inp, target = inp.to(device), target.to(device)
            target_D = target.view([inp.size(0), 2, 1, 1])
            D_input_real = torch.cat([inp, torch.ones_like(inp) * target_D], dim=1).to(device)
            D_result_real = D(D_input_real)  # (128)

            z_fake = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).to(device)
            G_input_fake = torch.cat([z_fake[:inp.size(0)], target_D], dim=1).to(device)
            G_result_fake = G(G_input_fake)
            D_input_fake = torch.cat([G_result_fake, torch.ones_like(G_result_fake) * target_D], dim=1)
            D_result_fake = D(D_input_fake)
            D_train_loss = d_loss_func(D_result_real, D_result_fake)

            D_train_loss.backward(retain_graph=True)
            D_optimizer.step()
            D_losses.append(D_train_loss.item())

            # train generator G
            G_optimizer.zero_grad()
            input_fake_G = torch.cat((G_result_fake, torch.ones_like(G_result_fake) * target_D), dim=1)
            D_result = D(input_fake_G)
            G_train_loss = g_loss_func(D_result)

            G_train_loss.backward(retain_graph=False)
            G_optimizer.step()
            G_losses.append(G_train_loss.item())

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), args.epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                     torch.mean(torch.FloatTensor(G_losses))))
        fixed_p = root + predicate + str(epoch + 1) + '.png'
        fixed_gt = root + label + str(epoch + 1) + '.png'
        show_result(valid_data, G, (epoch + 1),args, save=True, path=fixed_p,path2 = fixed_gt)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), args.epochs, total_ptime))
    print("Training finish!... save training results")
    torch.save(G.state_dict(), root + weights + 'generator_param.pkl')
    torch.save(D.state_dict(), root + weights + 'discriminator_param.pkl')
    with open(root + weights + 'train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path=root + hist + 'train_hist.png')

def main():
    args = init_argument()
    train_data_all, train_label_all = read_npy_data(path_data, path_label)

    train_data = create_data(
        {
            "input": train_data_all[:5700, ...],
            "target": train_label_all[:5700, ...]
        }
    )

    valid_data = create_data(
        {
            "input": train_data_all[5701:, ...],
            "target": train_label_all[5701:, ...]
        }
    )

    # network
    G = Generator()
    D = Discriminator()
    G.to(device)
    D.to(device)

    trainer(G, D, train_data, valid_data, args)


if __name__ == '__main__':
    main()

