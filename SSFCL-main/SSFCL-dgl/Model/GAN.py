"""
Time: 2024.1.5
Author: Yiran Shi
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import GD
import warnings
warnings.filterwarnings("ignore")

def select_negative_items(realData, num_pm, num_zr):
    '''
    realData : n-dimensional indicator vector specifying whether u has purchased each item i
    num_pm : num of negative items (partial-masking) sampled on the t-th iteration
    num_zr : num of negative items (zeroreconstruction regularization) sampled on the t-th iteration
    '''
    data = np.array(realData)
    n_items_pm = np.zeros_like(data)
    n_items_zr = np.zeros_like(data)
    for i in range(data.shape[0]):
        p_items = np.where(data[i] != 0)[0]
        all_item_index = random.sample(range(data.shape[1]), 32)
        random.shuffle(all_item_index)  # random.shuffle()用于将一个列表中的元素打乱顺序
        n_item_index_pm = all_item_index[0: num_pm]  # 生成list
        n_item_index_zr = all_item_index[num_pm: (num_pm + num_zr)]
        n_items_pm[i][n_item_index_pm] = 1
        n_items_zr[i][n_item_index_zr] = 1
    return n_items_pm, n_items_zr

# G = GAN.main(len(tmp1[0]), tmp1, epochs, pro_ZR, pro_PM, alpha)
def main(len, Vector, epochCount, pro_ZR, pro_PM, alpha):
    # Build the generator and discriminator
    G = GD.generator(len)
    D = GD.discriminator(len)
    regularization = nn.MSELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
    G_step = 5
    D_step = 2
    batchSize_G = 16
    batchSize_D = 16

    for epoch in range(epochCount):
        #  Train Generator
        for step in range(G_step):
            # Select a random batch of purchased vector
            leftIndex = random.randint(1, len - batchSize_G - 1)
            realData = Variable(Vector[leftIndex:leftIndex + batchSize_G])
            eu = Variable(Vector[leftIndex:leftIndex + batchSize_G])

            n_items_pm, n_items_zr = select_negative_items(realData, pro_PM, pro_ZR)
            ku_zp = Variable(torch.tensor(n_items_pm + n_items_zr))
            realData_zp = Variable(torch.ones_like(realData)) * eu + Variable(
                torch.zeros_like(realData)) * ku_zp

            # Generate a batch of new purchased vector
            fakeData = G(realData)  # G=cfgan.generator(itemCount, info_shape)
            fakeData_ZP = fakeData * (eu + ku_zp)
            fakeData_result = D(fakeData_ZP)  # D=cfgan.discriminator(itemCount, info_shape)

            # Train the discriminator
            g_loss = np.mean(np.log(1. - fakeData_result.detach().numpy() + 10e-5)) + alpha * regularization(
                fakeData_ZP, realData_zp)
            g_optimizer.zero_grad()  # g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

        #  Train Discriminator
        for step in range(D_step):
            # Select a random batch of purchased vector
            leftIndex = random.randint(1, len - batchSize_D - 1)
            realData = Variable(Vector[leftIndex:leftIndex + batchSize_D])
            eu = Variable(Vector[leftIndex:leftIndex + batchSize_G])
            # useInfo_batch = Variable(copy.deepcopy(UseInfo_pre[leftIndex:leftIndex + batchSize_G]))

            # Select a random batch of negative items for every user
            n_items_pm, _ = select_negative_items(realData, pro_PM, pro_ZR)
            ku = Variable(torch.tensor(n_items_pm))

            # Generate a batch of new purchased vector
            fakeData = G(realData)
            fakeData_ZP = fakeData * (eu + ku)

            # Train the discriminator
            fakeData_result = D(fakeData_ZP)
            realData_result = D(realData)
            d_loss = -np.mean(np.log(realData_result.detach().numpy() + 10e-5) +
                              np.log(1. - fakeData_result.detach().numpy() + 10e-5)) + 0 * regularization(fakeData_ZP,
                                                                                                          realData_zp)
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

    return G
