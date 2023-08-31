import os
import requests
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# from google.colab import drive
# drive.mount('/content/drive')

from datetime import datetime

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input,self.weight.shape, self.weight, self.bias, 1e-5)

#trying to make it more modular so put attention as a separate class on it's own
class multiHeadAttn(nn.Module):
  def __init__(self, dFf, d_k, n_heads, vocabSize, blockSize, processor = 'cpu'):
    super().__init__()
    self.d_k = d_k
    self.n_heads = n_heads
    self.dModel = d_k  #512 in the paper
    self.dFf = dFf # 2048 in the paper
    self.processor = processor
    self.encodeQ = nn.Linear(d_k, d_k).to(device = processor) #to include in backpropogation graph you need to use "self."
    self.encodeK = nn.Linear(d_k, d_k).to(device = processor)
    self.encodeV = nn.Linear(d_k, d_k).to(device = processor)

  def forward(self, qkv):
    B, T, C = qkv.size()

    Q = self.encodeQ(qkv).to(device = self.processor)#same process for all 3 (see paper multi-head attn figure)
    K = self.encodeK(qkv).to(device = self.processor)
    V = self.encodeV(qkv).to(device = self.processor)

    Q = Q.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2) #B, heads, T, C
    K = K.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2) #B, heads, T, C
    V = V.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2) #B, heads, T, C

    #see (Q*K.T)/sqrt(d^k) in paper (this is self-attention where the embeddings are self-multiplied and trained separately
    # for example, 3 x 32 becomes 3 x 3 self score where one element will eventually figure out it's relation to members of neighbouring timesteps
    # maybe we can make a heat map for this and see how it progresses over time
    QkDotProduct = torch.matmul(Q,K.transpose(-2,-1))/torch.sqrt(torch.Tensor([self.d_k]).to(device = self.processor))# B, T, T
    #each row is 1 timestep more
    successiveTokensMask = torch.tril(torch.ones(T,T)).view(1,1,T,T).to(device = self.processor)
    # setting everything ahead of the mask to infinity to make sure labels aren't included in training data
    QkDotProduct = QkDotProduct.masked_fill(successiveTokensMask[:,:,:T,:T] == 0, float('-inf'))
    #gives scores per query with respect to the key it matches (softmax on final dim with shape = B, T, C)
    dotProdAttn = torch.softmax(QkDotProduct, dim=-1)
    #multiply attn weights with "V". The softmax after the mask forces the division of attention based on how many words there are in the sample
    headOutput = torch.matmul(dotProdAttn, V)
    headOutput = headOutput.transpose(1,2).contiguous().view(B,T,C)

    return headOutput

class DecoderBlock(nn.Module):
  def __init__(self, dFf, d_k, n_heads, vocabSize, blockSize, processor = 'cpu'):
    super().__init__()
    self.selfAttn = multiHeadAttn(dFf, d_k, n_heads, vocabSize, blockSize, processor)

    self.FFN = nn.Sequential(
                nn.Linear(d_k, dFf).to(device = processor),
                nn.ReLU().to(device = processor),
                nn.Linear(dFf, d_k).to(device = processor)
        ).to(device = processor)

    self.ln = LayerNorm(d_k, False).to(device = processor)


  def forward(self, qkv):
    x = self.selfAttn(qkv)
    x = x + self.ln(x)
    x = self.FFN(x)
    x = x + self.ln(x)
    return x

class theTransformer(nn.Module):
  def __init__(self, attnLayers, dFf, d_k, parallelHeads, vocabSize, blockSize, processor = 'cpu'):
    super().__init__()
    self.d_k = d_k
    self.embedTok = nn.Embedding(vocabSize, d_k).to(device = processor) #vocabSize, channelsWanted
    self.embedPos = nn.Embedding(blockSize, d_k).to(device = processor) #vocabSize, channelsWanted
    self.decoder = DecoderBlock(dFf, d_k, parallelHeads, vocabSize, blockSize, processor) #instantiate decoder block
    self.layers = nn.ModuleList([self.decoder for decoders in range(attnLayers)]).to(device = processor)
    self.LinearFinal = nn.Linear(self.d_k, vocabSize).to(device = processor)
    self.processor = processor

  def forward(self, batchData, target, pos, training = True, softmax = True):
    x = self.embedTok(batchData) + self.embedPos(pos)
    for l in self.layers: # running through all the decoder layers we wanted
      x = l(x)
    logits = self.LinearFinal(x) # get per token logits for later backprop
    B,T,C = logits.shape

    if softmax:
      logits = torch.softmax(logits, dim = 1) # get per token probabilities for later backprop

      #the cross_entropy function has dimension requirement which we solve by making pred dim = (B*T, C) and target dim = (B*T) (C is classes not channels, while target has ground truth so only 1 class)
      pred, target = logits.view(B*T, C), F.one_hot(target.long().view(B*T), num_classes = 65).type(torch.FloatTensor).to(device = self.processor)#if we want to use softmax (need to convert one-hot encoding to float so it can actually do cross entropy operations with pred which is float)
    else:
      pred, target = logits.view(B*T, C), target.long().view(B*T)

    if training:
      #remember, targets are offset by one token to the future, so this loss function pushes the model towards predicting the next token using the encoded information
      loss = F.cross_entropy(pred, target).to(device = self.processor)
    else:
      # logits = logits.view(B*T, C) #
      logits = logits[:, [-1],:]
      loss = None

    return loss, logits