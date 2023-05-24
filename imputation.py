import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import time
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import numpy as np
import pandas as pd
# Imputer
from NRN_MLP import NRN_m
from NRN_Attention import NRN_a
from utils import load_data, generateMissing

import argparse
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='asf')
    parser.add_argument('--random_seed', type=int, default=933)
    parser.add_argument('--missing_rate', type=float, default=0.1)
    parser.add_argument('--missing_mechanism', type=str, default='MCAR')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--mixer_depth', type=int, default=4)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--denoising_rate', type=float, default=0.4)
    parser.add_argument('--method', type=str, default='mlp')
    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    use_cuda = True

    data_path = './data/' + args.dataset + '.csv'
    X = pd.read_csv(data_path, sep=',', index_col=None, dtype=float).to_numpy()
    clean_data = pd.read_csv(data_path, sep=',', index_col=None, dtype=float).to_numpy()
    rows, cols = X.shape
    print(rows, cols)
    
    scaler = MinMaxScaler(feature_range=(0.1, 1.1))
    scaler.fit(clean_data)
    clean_data = scaler.transform(clean_data)
    missed_data = generateMissing(X, args.missing_rate, args.random_seed)
    mask = np.isnan(missed_data)
    # the imputer's parameters
    num_epochs = args.epochs
    K = args.K
    mixer_depth = args.mixer_depth
    theta = 4
    denoising_rate = args.denoising_rate
    batch_size  = args.batch_size
    learning_rate = args.lr
    start = time.time()
    missed_data_K = load_data(missed_data, K, scaler)
    
    #print(missed_data_K)
    missed_data_K = np.nan_to_num(missed_data_K)
    train_data = torch.from_numpy(missed_data_K).float()
    test_data = torch.from_numpy(missed_data_K).float()

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=batch_size,
                                            shuffle=True)
                                            
    print(train_data.shape)
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.method == 'mlp':
        model = NRN_m(features=cols, K_neighbor=K+1, depth=mixer_depth, 
                                d_model=128, theta=theta, drop_out=denoising_rate).to(device)
    else :
        model = NRN_a(features=cols, K_neighbor=K+1, depth=mixer_depth,
                                d_model=128, n_heads=theta, drop_out=denoising_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    cost_list = []

    # train
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            _, loss = model(batch_data)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            cost_list.append(loss)
        if epoch % 20 == 0:
            print('Epoch [%d/%d], Loss: %.6f'%(epoch+1, num_epochs, epoch_loss))
    print("Learning Finished!")
    model.eval()
    finish = time.time()
    print('time consumed: {:.2f}s'.format(finish - start))

    # test
    rmse = 0
    with torch.no_grad():
        filled_data, _ = model(test_data.to(device))
        reconst_data = filled_data.cpu().detach().numpy()
        predicts = reconst_data[mask]
        targets = clean_data[mask]
        rmse = np.sqrt(((np.array(targets) - np.array(predicts)) ** 2).mean())
        print("RMSE:", rmse)

if __name__ == "__main__":
    main()
