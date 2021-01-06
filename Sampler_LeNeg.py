# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:56:12 2020

@author: leona
"""

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

#mat = scipy.io.loadmat(os.path.join(args.dataset_folder, args.dataset_name, "clusters.mat"))

#clusters=mat['centroids']
    
class Sampler_LeNeg(nn.Module):

    '''
    This class is pretty straightforward:
    from the 1000 dim output for each input,
    it samples num_samples elements(which are the clusters)
    and return the positions linked to those clusters
    and as attribute the probabilities of the selected clusters
    (this last one is needed for the dpp loss and has name 'zs')

    '''
# =============================================================================
#num_samples is how many samples we take from the future step
#out_TF if the **first prob vector** of the Transformer output
# =============================================================================
    
    def __init__(self, clusters, out_TF, num_samples = 10):
        super(Sampler_LeNeg, self).__init__()
        self.out_TF = out_TF
       # print('self.tf_out shape', self.out_TF.size()) #this is only 1 vector(or batch) of dimension 1000 in this notation
        self.num_samples = num_samples
        self.clusters = clusters
        self.zs = torch.empty(num_samples, 1)
        self.clusters_id = torch.empty(num_samples, 1)

    
    def forward(self, bool = True):
        #1) sample num_samples cluster_id
        if bool == True:
          #we opted for multinomial since is clearly a good way
          clusters_id = torch.multinomial(self.out_TF, self.num_samples).cpu() #e.g [1, 5, 980]
          #print('clusters_id', clusters_id.size()) everything checks out
          #print(clusters_id)
          self.clusters_id = clusters_id

          #We need to create a vector of dim 1024, 10
          #where we have <<prob of 1st selected cluster>>, <<prob of 2nd selected cluster>>...
          temp = (self.out_TF[0, clusters_id[0]].reshape(-1,1)).clone().detach() #you can take out clone()
          #for each of the 1000 rows I convert the 10 clusters into the previous 
          #probs (this will be the zs of out DPP process)

          for i in range(1, clusters_id.size()[0]):
              temp = torch.cat((temp, self.out_TF[i, clusters_id[i]].reshape(-1, 1)), dim = 1)
          #store it for access in quantized_TFsamples file
          self.zs = temp
        #2) Convert the cluster_id in position
          positions = torch.from_numpy(self.clusters[clusters_id]) #e.g. [(1,2), (0.1, 0.3), (10, 1)]
        
          return positions
    
    