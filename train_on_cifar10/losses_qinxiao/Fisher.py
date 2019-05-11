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
        num_of_instances_per_appeared_class[num_of_instances_per_appeared_class==0]=1e-6#避免除零操作

        transposedTargets=one_hot_targets.t()
        sum_of_embeddings_per_appeared_class=torch.mm(transposedTargets,inputs)
        mean_embedding_per_appeared_class=sum_of_embeddings_per_appeared_class/num_of_instances_per_appeared_class

        square_diff_per_instance=(inputs-torch.index_select(input=mean_embedding_per_appeared_class,dim=0,index=targets))**2
        sum_of_square_diff_per_appeared_class=torch.mm(transposedTargets,square_diff_per_instance).sum(1)
        sum_of_square_diff_per_appeared_class=torch.unsqueeze(sum_of_square_diff_per_appeared_class,1)
        var_of_embeddings_per_appeared_class=sum_of_square_diff_per_appeared_class/num_of_instances_per_appeared_class

        inter_loss_matrix=self.expanded_pairwise_distances(mean_embedding_per_appeared_class)
        intra_loss_matrix=var_of_embeddings_per_appeared_class+var_of_embeddings_per_appeared_class.t()

        loss_matrix=intra_loss_matrix-inter_loss_matrix+self.margin
        loss_matrix[loss_matrix<0]=0
        loss_matrix=loss_matrix-torch.eye(loss_matrix.shape[0])*loss_matrix
        return torch.max(loss_matrix)
    

def main():
    inputs=torch.from_numpy(np.load('inputs.npy'))
    targets=torch.from_numpy(np.load('targets.npy'))


    print(FisherLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')

