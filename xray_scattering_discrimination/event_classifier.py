import os
directory = os.path.dirname(os.path.realpath(__file__))

import numpy as np
#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.utils.data import DataLoader, TensorDataset
#from sklearn.model_selection import train_test_split
#from kan import KAN

class simple_probabilistic_classifier:
    def __init__(self, identifier):
        with np.load(directory + "/data/" + identifier + "/geant4_dist_merged.npz") as data:
            self.dist = data['dist']
            self.list1 = data['list1']
            self.list2 = data['list2']
            self.list3 = data['list3']
            self.list4 = data['list4']
    
    def confusion_matrix(self):
        dist_reshape = self.dist.reshape(-1,
                                         self.list1.size,
                                         self.list2.size,
                                         self.list3.size,
                                         self.list4.size)
        dist_reshape += 1e-8
        Pscint = dist_reshape/np.sum(dist_reshape, axis=(1,2,3,4))[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
        Pdet = dist_reshape/np.sum(dist_reshape, axis=0)[np.newaxis,:,:,:,:]
        
        Mconf = np.tensordot(Pscint, P_det.transpose(1,2,3,4,0), axes=4)
        np.savez(directory + "/data/" + identifier + "/confusion_matrix", list1=self.list1, list2=self.list2, list3=self.list3, list4=self.list4,
                                                                          dist_reshape=dist_reshape,
                                                                          Pscint=Pscint,
                                                                          Pdet=Pdet,
                                                                          Mconf=Mconf)

#class DNN_classifier(nn.Module):
#    def __init__(self, input_dim, hidden_sizes):
#        super().__init__()
#        
#        layers = []
#        prev_size = input_dim
#        
#        for size in hidden_sizes:
#            layers.append(nn.Linear(prev_size, size))
#            layers.append(nn.ReLU())
#            prev_size = size
#        
#        self.shared = nn.Sequential(*layers)
#        self.rayleigh_head = nn.Linear(prev_size, 1)
#        self.photoelectric_head = nn.Sequential(nn.Linear(prev_size, 1),
#                                                nn.Sigmoid(),
#                                                )
#        self.compton_head = nn.Linear(prev_size, 1)
#        
#    def forward(self, x):
#        shared_out = self.shared(x)
#        
#        return self.rayleigh_head(shared_out), self.photoelectric_head(shared_out), self.compton_head(shared_out)
#
#class KAN_classifier:
#    def __init__(self, input_dim, hidden_layers):
#        self.            

if __name__ == '__main__':
    identifier = 'YAGCe_200keV'
    classifer = simple_probabilistic_classifier(identifier)
    classifier.confusion_matrix()