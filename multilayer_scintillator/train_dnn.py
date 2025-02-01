import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-24])

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

def train(procID,
          train_x,
          test_x,
          train_y,
          test_y,
          N_in,
          N_hidden,
          N_out,
          ):
          
    device = "cuda:" + str(procID) if torch.cuda.is_available() else "cpu"
    
    x_train = torch.FloatTensor(train_x).to(device)
    y_train = torch.FloatTensor(train_y).to(device)
    x_test = torch.FloatTensor(test_x).to(device)
    y_test = torch.FloatTensor(test_y).to(device)
    
    lr = np.geomspace(1e-5, 1e0, 6)
    N_hidden = np.linspace(N_in, N_out, N_out-N_in+1)
    model_dict = {}
    loss_test_all = np.zeros((lr.size, N_hidden.size))
    
    for i in range(lr.size):
        for j in range(N_hidden.size):
            model = torch.nn.Sequential(torch.nn.Linear(N_in, N_hidden[j]),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(N_hidden[j], N_out)).to(device)
        
            optimizer = torch.optim.ADAM(model.parameters(), lr=lr[i])
        
            y_pred = model(x_train)
            loss = MSE(y_pred, y_train)
            loss_prev = loss.detach().numpy()
            
            y_pred = model(x_test)
            loss_test = MSE(y_pred, y_test)
            loss_test_min = loss_test.detach().numpy()
            
            iteration = 0
            while True:
                optimizer.zero_grad() # gradient initialization
                loss.backward() # gradients w.r.t. weights & biases
                optimizer.step() # update parameters
                
                y_pred = model(x_train)
                loss = MSE(y_pred, y_train)
                
                y_pred = model(x_test)
                loss = MSE(y_pred, y_test)
                
                if iteration > 2e4:
                    break
                elif np.abs(loss_prev - loss.detach().numpy()) < 1e-6:
                    break
                elif loss_test_min < 0.99*loss_test.detach().numpy():
                    break
                elif loss_test_min > loss_test.detach().numpy():
                    loss_test_min = loss_test.detach().numpy()
                    
                loss_prev = loss.detach().numpy()
                iteration += 1
            
            model_dict[i,j] = model
            loss_test_all[i,j] = loss_test_min

    best_ind = np.argwhere(loss_test_all==np.min(loss_test_all))

    return model_dict, loss_test_all, best_ind

def MSE(y_pred, y_true):
    return torch.sum(torch.sum((y_pred - y_true)**2, axis=1))/y_true.shape[0]

def predict(i, index, index_disc, model, statusfile=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    x = torch.FloatTensor(np.hstack((index, index_disc))).to(device)
    R = model(x)
    R = R.detach().numpy()
    
    lam = np.linspace(300, 2500, 2201, endpoint=True)
    I_am1_5 = rc.load_solar('AM1.5_SMARTS295', 1, wavelength=lam)
    fom = np.dot(1-R, I_am1_5)

    if statusfile != None:
        with open(statusfile, 'a') as f:
            f.write('Cost %d (DNN): %f\n' %(i, fom))
    print('Index ' + str(int(i)) + ' (DNN): ' + str(fom))

    return fom

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"