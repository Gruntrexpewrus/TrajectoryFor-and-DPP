# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:41:53 2020

@author: leona
"""


'''
  Here we implement the central Loss of the paper 
  @article{DBLP:journals/corr/abs-1907-04967,
  author    = {Ye Yuan and
               Kris Kitani},
  title     = {Diverse Trajectory Forecasting with Determinantal Point Processes},
  journal   = {CoRR},
  volume    = {abs/1907.04967},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.04967},
  archivePrefix = {arXiv},
  eprint    = {1907.04967},
  timestamp = {Wed, 17 Jul 2019 10:27:36 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1907-04967.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
'''

import torch
import torch.nn as nn

class LeNeg_loss(nn.Module):

  '''
  
  The paper does not deal with Transformers, then we had to think about an adaptation
  and we decided to assume that:
  1) our z_i are the probs associated with the sampled positions at each step e.g. [0.32]
  2) our x_i are indeed the positions e.g. [[0,1, 0.3]]

  The development follows the algorithm 2 (then 1 in the new print of the paper)
  and we describe it here:
  1) Identify the sets of z_i and x_i
  2) Since we did not have a network for sampling, we did not have the parameters Gamma. We talked about it with Dr. Prof Galasso
  and Dr. Franco, and we decided that a good solution was to put the DPP loss in the transformer itself
  then the final Gamma are indeed all the parameters of the Transformer. 
  3) Compute similarity matrix 'similarity' and the quality vector r (called DiagR_mat)
  4) compute the DPP_kernel
  5) Compute the loss with the formula = trace(I − (L(γ) + I)^−1) where L(γ) is indeed the DPP_kernel
  6) here since the loss was negative (this could be discussed) we decided to change the sign of it,
  anyway torch would optimize to 0 so the result should not vary
  7) backprop is automatic in torch nn module.




  We must mention that, since the TF alone (without DPP) was trained, we decided to 'retrain it' starting from those parameters.
  The reason is that we believe that training the tf from random parameters with DPP losse, would result in 'unlikely' 
  in matter of probability measure, future samples, but only 'diverse'. 
  In brief we tried a '2 block method' where we took a trained TF (for likely paths) to diversify the paths (with DPP),
  starting the training from the optimized tf.
  '''



  def __init__(self, R, k = 0.10, omega = 0.1):
    super(LeNeg_loss, self).__init__()
    self.k = k
    self.R = R
    self.omega = omega


  def forward(self, vector_of_zs, vector_of_xs, num_samples):
    #print('loss xs shape',  vector_of_xs.shape) #expected shape 1024, 10, 2
    dim = vector_of_zs.shape[1]
    #print('SHAPEEEES')
    #print(vector_of_zs.shape[0], vector_of_zs.shape[1])
    vector_of_zs = torch.reshape(vector_of_zs, (dim, -1))
    #print('loss zs.shape', vector_of_zs.shape)  #expected 1024, 10
    #mtrxA= (vector_of_xs)
    dist = torch.cdist(vector_of_xs,vector_of_xs)
    similarity = torch.exp(-self.k* dist)
    DiagR_mat = torch.empty(size = vector_of_zs.size())
    number_elements = list(vector_of_zs.size())[0]
    for i in range(vector_of_zs.size()[0]):
      if torch.linalg.norm(vector_of_zs[i]) <= self.R :
        DiagR_mat[i] = self.omega
      else :
        DiagR_mat[i] = (self.omega*torch.exp(-vector_of_zs[i]**2+self.R**2))
    DiagR_matrix = torch.diag_embed(DiagR_mat)
    #print('DiagR.shape', DiagR_mat.shape) 
    #we want now to pass from torch.Size([1024, 10])
    #to torch.Size([1024, 10, 10])
    #print('similarity shape', similarity.shape)
    #print('DiagR eye shape', DiagR_matrix.shape)
    #print('ENTERING THE DPP KERNEL')
    DPP_kernel = torch.matmul(torch.matmul(DiagR_matrix, similarity), DiagR_matrix)
    #seek how to do 1024 eye matrixes 10x10  (do 1024 eye matrixes)
    #seek how to get the trace how 1024x10x10 matrixes resulting in sum(1024x10)
    eye = torch.stack([torch.eye(num_samples) for i in range(dim)], dim=0) #10 is num of samples
    intermediate = eye-torch.inverse(DPP_kernel + eye)
    tr = 0 
    for i in range(intermediate.shape[0]):
      #print(torch.trace(intermediate[i,:,:]))
      tr = tr + torch.trace(intermediate[i,:,:])
    #print('##############################################')
    #print('TRACE', tr/dim)
    #print('\n')
    #print('###############################################')
    #i need the minus from paper
    #print(dim)
    return -tr/dim #vector_of_zs.shape[1]

