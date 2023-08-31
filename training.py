import os
import requests
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from transformerMain import theTransformer
from datetime import datetime

wordToTensor = lambda sInput, tokenMapping: [tokenMapping[letter] for letter in sInput] # to get numerical tensor to feed into nn.Embedding function (each letter has it's index)
tensorToWord = lambda sIndexes, tokenMapping: [list(tokenMapping.keys())[i.item()] for i in sIndexes]

def saveModel(model, pathReq = None):

  fileName = 'model_' + str(datetime.now()).replace('-','').replace(':','').replace(' ','')[:14] + '.pth'

  if pathReq is None:
      filePath = os.path.join(os.getcwd(), fileName) 
  else:
      filePath =  os.path.join(pathReq, fileName) #/content/drive/MyDrive/myRepository/savedModels/
      
    

#   torch.save(model.state_dict(), filePath)
  torch.save(model, filePath) #dont need to know hyperparameters necessarily beforehand while running inference

  print('model saved in {}'.format(filePath))
  
def myDataLoader(dataSet, batchSize, blockSize):
  batch = []
  toPredict = []
  i = 0
  block = 0
  toShuffle = np.arange(0,batchSize)
  np.random.shuffle(toShuffle)

  for b in range(0, len(dataSet)//blockSize):
    batch.append(dataSet[block : (block + blockSize)])
    toPredict.append(dataSet[block + 1 : (block + 1 + blockSize)]) #adding next letter for the one to predict
    block += blockSize
    if (b+1) % batchSize == 0:
      # yield batch, toPredict
      yield np.array(batch)[toShuffle], np.array(toPredict)[toShuffle]
      batch = []
      toPredict = []
#====

def train(data, tokenMapping, lDff, lD_k, lNLayers, lParallelHeads, lVocabSize, lBatchSize, lBlockSize, lPos, lEpochs, softmax, lLearningRate, savePath, processor = 'cpu'):

  split = int(0.9*len(data)) #decided that we can se 90% on the dataset
  valData = data[split:]
  data = data[:split]

  numericRepresentation = wordToTensor(data,tokenMapping) #changing character to index-based integer

  finalTransformer = theTransformer(lNLayers, lDff, lD_k, lParallelHeads, lVocabSize, lBlockSize, processor)

  finalTransformer.to(device = processor)

  optimizer = torch.optim.Adam(finalTransformer.parameters(), lr=lLearningRate) #make optimizer for the decoder block

  i = 0
  measurementsPerEpoch = 20
  # resolutionFactor = (1/(measurementsPerEpoch))
  
  dataLength = len(data)
  
  for e in range(lEpochs):
    for batchData, forwardPrediction in myDataLoader(numericRepresentation, lBatchSize, lBlockSize):
      optimizer.zero_grad()
      loss, outputLogits = finalTransformer(torch.IntTensor(batchData).to(device = processor), torch.LongTensor(forwardPrediction).to(device = processor) , lPos.to(device = processor), softmax=softmax)
      loss.backward()
      optimizer.step()

      i += 1
      # print(int((len(data)/len(batchData)) / measurementsPerEpoch))
      if  i % int((len(data)/(lBatchSize*lBlockSize)) / measurementsPerEpoch) == 0:
        print('epoch:', e, 'training loss:', loss.item())

      if i  % int((dataLength/(lBatchSize*lBlockSize)) / (measurementsPerEpoch / 5)) == 0 :
        saveModel(finalTransformer, savePath)
        val(finalTransformer, valData, tokenMapping, lDff, lD_k, lNLayers, lParallelHeads, lVocabSize, lBatchSize, lBlockSize, lPos, softmax, lLearningRate, measurementsPerEpoch, processor)        
        i = 0
        finalTransformer.train()

def val(transformerEvalModel, dataVal, tokenMapping, lDff, lD_k, lNLayers, lParallelHeads, lVocabSize, lBatchSize, lBlockSize, lPos, softmax, lLearningRate, measurementsPerEpoch, processor = 'cpu'):

  numericRepresentationEval = wordToTensor(dataVal,tokenMapping) #changing character to index-based integer

  transformerEvalModel.to(device = processor)

  ii = 0

  transformerEvalModel.eval()
  
  print('validation set:')
  
  dataLength = len(dataVal)
  
  with torch.no_grad():
    for  batchData, forwardPrediction in myDataLoader(numericRepresentationEval, lBatchSize, lBlockSize):
      loss, outputLogits = transformerEvalModel(torch.IntTensor(batchData).to(device = processor), torch.LongTensor(forwardPrediction).to(device = processor) , lPos.to(device = processor), softmax=softmax)

      ii += 1

      if ii  % int((dataLength/(lBatchSize*lBlockSize)) / (measurementsPerEpoch / 5)) == 0:
        print('validation loss: ', loss)
        ii = 0
#====================================

def getRawData():
  data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
  data = requests.get(data_url).text

  print(f"length of raw dataset: {len(data):,}")

  # get all the unique characters that occur in this text
  possibleTokens = sorted(list(set(data)))

  return data, possibleTokens, len(possibleTokens)

import argparse
import sys
import os
# from google.colab import drive
# drive.mount('/content/drive')

def runProcess():
    
    data, gTokens, gVocabSize = getRawData()
    gBlockSize = 16
    gChannelsWanted = 64
    gD_k = gChannelsWanted
    gPos = torch.arange(0, gBlockSize, dtype=torch.long) #to make a kind of positional embedding
    vocab = gTokens # !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
    tokenMapping = {k:v for v, k in enumerate(vocab)} #make dict where value is index of token and key is token itself
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize", required = False, type=int, default = 8)
    parser.add_argument("--blockSize", required = False, type=int, default = gBlockSize)
    parser.add_argument("--learningRate", required = False, type=float, default = 1e-3)
    parser.add_argument("--epochs", required = False, type=int, default = 20)
    parser.add_argument("--parallelHeads", required = False, type=int, default = 8) #for batch processing
    parser.add_argument("--nLayers", required = False, type=int, default = 3) #how many blocks to use to make deeper network
    parser.add_argument("--channelsWanted", required = False, type=int, default = gChannelsWanted) # hopefully a good enough embdding to learn required features
    parser.add_argument("--dff", type=int, default = 128) #for intermediary feedforward process
    parser.add_argument("--softmax", required = False, default = False)
    parser.add_argument("--usedProcessor", required = False, default = 'cpu')
    parser.add_argument("--savePath", required = False, default = None)
    
    args = parser.parse_args()
    print('starting training with parameters: ', args )
    
    train(data, tokenMapping, args.dff, gD_k, args.nLayers, args.parallelHeads, gVocabSize, args.batchSize, args.blockSize, gPos, args.epochs,  args.softmax, args.learningRate, args.savePath, args.usedProcessor)
    
    

if __name__ == "__main__":

    if os.getcwd() + '/drive/My Drive/myRepository' not in sys.path:
        sys.path.append(os.getcwd() + '/My Drive/myRepository')
    
    runProcess()