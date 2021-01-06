import torch
import torch.nn as nn
import torch.nn.functional as F
from Sampler_LeNeg import Sampler_LeNeg
from transformer.decoder import Decoder
from transformer.multihead_attention import MultiHeadAttention
from transformer.positional_encoding import PositionalEncoding
from transformer.pointerwise_feedforward import PointerwiseFeedforward
from transformer.encoder_decoder import EncoderDecoder
from transformer.encoder import Encoder
from transformer.encoder_layer import EncoderLayer
from transformer.decoder_layer import DecoderLayer
from transformer.batch import subsequent_mask
from transformer.embeddings import Embeddings
from transformer.generator import Generator
import numpy as np
import scipy.io
import os

import copy
import math
'''
This is a modification of the original Quantized TF, here instead of returning, for every element, 
a 1000 dim vector, we sample #num_samples of element from that vector, giving them as result
both the output clusters and (just for completeness) the final positions relative to those clusters
'''
class QuantizedTF(nn.Module):
    def __init__(self, enc_inp_size, dec_inp_size, clusters, dec_out_size, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1, num_samples = 20):
        super(QuantizedTF, self).__init__()
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model,enc_inp_size), c(position)),
            nn.Sequential(Embeddings(d_model,dec_inp_size), c(position)),
            Generator(d_model, dec_out_size))
        #declare stuff we will us in predict
        self.clusters = clusters
        self.num_samples = num_samples
        self.zs = 0
        self.clusters_id = 0
        #here fun to sample and return coordinatesof future
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, *input):
        #here next a list of 12 prob vectors for each point
        after_tf = F.softmax(self.model.generator(self.model(*input)), dim = 1)#vector of probs
        #here we samples from the output vector(called after_tf)
        #we call the class Sampler_LeNeg because we developed a way to find back the positions
        #this can be taken out since it partially slow execution, but is good to keep for modifications
        obj = Sampler_LeNeg(self.clusters, after_tf[:, -1], self.num_samples )
        out_samples = obj(bool = True)
        self.clusters_id = obj.clusters_id
        self.zs = obj.zs
        return out_samples
        
    def predict(self,*input):
        #same as forward since loss is no problem in regards to softmax
        after_tf = F.softmax(self.model.generator(self.model(*input)), dim = 1)#vector of probs
        #sample again
        obj = Sampler_LeNeg(self.clusters, after_tf[:, -1], self.num_samples )
        out_samples = obj(bool = True)
        self.clusters_id = obj.clusters_id
        self.zs = obj.zs

        return out_samples, self.clusters_id


