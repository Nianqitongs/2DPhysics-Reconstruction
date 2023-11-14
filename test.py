# 院校：南京信息工程大学
# 院系：自动化学院
# 开发时间：2023/11/14 13:07
import torch
import numpy as np
import matplotlib.pyplot as plt
# fixed noise & label
temp_z_ = torch.randn(64, 100)
fixed_z_ = temp_z_
fixed_z_ = fixed_z_.view(-1, 100).cuda()
def show_result(valid_data,G,num_epoch, show = False, save = False, path = 'result.png'):

    G.eval()
    target = []
    for i in range(64):
        temp = valid_data[i]['target']
        target.append(temp)
    target = torch.tensor(np.vstack(target)).float().cuda()
    input_test = torch.cat((fixed_z_,target),dim=1)
    test_images = G(input_test.view(-1,102,1,1))
    G.train()

    plt.figure(figsize=(8, 8))
    for i in range(8):
        for j in range(8):
            plt.subplot(8, 8, i * 8 + j + 1)
            plt.imshow(test_images[i * 8 + j, 0, :, :].detach().cpu().numpy(), cmap='gray')
            plt.axis(False)
    plt.tight_layout()
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()