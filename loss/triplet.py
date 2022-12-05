#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
import random

class LossFunction(nn.Module):

    def __init__(self, hard_rank=0, hard_prob=0, margin=0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.hard_rank  = hard_rank
        self.hard_prob  = hard_prob
        self.margin     = margin

        print('Initialised Triplet Loss')

    def forward(self, x, label=None):

        assert x.size()[1] == 2
        
        # Normalize anchor and positive
        out_anchor      = F.normalize(x[:,0,:], p=2, dim=1)
        out_positive    = F.normalize(x[:,1,:], p=2, dim=1)

        # Choose appropriate negative indices
        negidx      = self.choose_negative(out_anchor.detach(),out_positive.detach(),type='any')

        # Get negative pairs
        out_negative = out_positive[negidx,:]

        ## Calculate positive and negative pair distances
        pos_dist    = F.pairwise_distance(out_anchor,out_positive)
        neg_dist    = F.pairwise_distance(out_anchor,out_negative)

        ## Triplet loss function
        nloss   = torch.mean(F.relu(torch.pow(pos_dist, 2) - torch.pow(neg_dist, 2) + self.margin))

        return nloss

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Negative mining
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def choose_negative(self, embed_a, embed_p, type=None):
      
        # Get batch size
        batch_size = embed_a.size(0)

        # Positive and negative indices
        negidx = [] # empty list to fill
        allidx = range(0,batch_size)

        for idx in allidx:
            
            excidx = list(allidx)
            excidx.pop(idx)

            if type == 'any':
                # Random negative mining
                negidx.append(random.choice(excidx))

            elif type == 'semihard':
                # Compute pairwise distance between the anchor and all positives - use F.pairwise_distance
                distance = F.pairwise_distance(embed_a[idx, :], embed_p)

                # Semi-hard negative mining - criteria (distance greater than the positive pair, and less than positive + margin )
                hardidx = torch.IntTensor(allidx)[(distance > distance[idx]) & (distance < distance[idx] + self.margin)]
                #not sure about distance[idx] part

                if len(hardidx) == 0:
                    # append random index if no index matches the criteria
                    # (your code)
                    negidx.append(random.choice(excidx))
                else:
                    # append a random index that meets the criteria
                    # (your code)
                    negidx.append(int(random.choice(hardidx)))

            elif type == 'hard':
                # Compute pairwise distance between the anchor and all positives - use F.pairwise_distance
                #distance = # (your code)

                # Hard negative mining - criteria (distance less than the positive pair)
                # (your code)
                distance = F.pairwise_distance(embed_a[idx,:], embed_p)
                hardidx = torch.IntTensor(allidx)[(distance < distance[idx])]
                
                if len(hardidx) == 0:
                    # append random index if no index matches the criteria
                    # (your code)
                    negidx.append(random.choice(excidx))
                else:
                    # append a random index that meets the criteria
                    # (your code)
                    negidx.append(int(random.choice(hardidx)))
                 

            else:
                ValueError('Undefined type of mining.')            
        return negidx