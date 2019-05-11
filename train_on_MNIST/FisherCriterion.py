#coding=utf-8
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class FisherLoss(nn.Module):
    def __init__(self, margin=1.0, hard_mining=True):
        super(FisherLoss, self).__init__()
        self.margin = margin
        self.hard_mining = hard_mining
    
    def expanded_pairwise_distances(self,x, y=None):
        '''
        Input: x is a Nxd matrix
            y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        if y is not None:
            differences = torch.unsqueeze(x,1) - torch.unsqueeze(y,0)
        else:
            differences = torch.unsqueeze(x,1) - torch.unsqueeze(x,0)
        distances = torch.sum(differences * differences, -1)
        return distances

    def forward(self, inputs, targets):
        inputs=inputs.cpu().float()
        targets=targets.cpu().long()
        one_hot_targets=torch.zeros(targets.shape[0], targets.max()+1).scatter_(1,torch.unsqueeze(targets,1), 1)#转成one_hot编码
        num_of_instances_per_appeared_class=torch.unsqueeze(one_hot_targets.sum(0),1)
        # num_of_instances_per_appeared_class[num_of_instances_per_appeared_class==0]=1e-6#避免除零操作

        transposedTargets=one_hot_targets.t()
        sum_of_embeddings_per_appeared_class=torch.mm(transposedTargets,inputs)
        mean_embedding_per_appeared_class=sum_of_embeddings_per_appeared_class/(num_of_instances_per_appeared_class+1e-6)

        mean_embedding_all=inputs.mean(0)
        temp=mean_embedding_per_appeared_class-mean_embedding_all
        S_b=torch.mm((temp*num_of_instances_per_appeared_class).t(),temp)

        temp2=inputs-mean_embedding_all
        S_t=torch.mm(temp2.t(),temp2)
        
        loss=1.0/torch.trace(torch.mm(S_b,torch.inverse(S_t+1e-6*torch.eye(S_t.shape[0]))))
        return loss
    

def main():
    inputs=torch.from_numpy(np.load('inputs.npy'))
    targets=torch.from_numpy(np.load('targets.npy'))


    print(FisherLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')

