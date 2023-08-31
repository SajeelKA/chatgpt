import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from transformerMain import theTransformer, DecoderBlock, multiHeadAttn, LayerNorm #while loading model, need these imported
import argparse
import sys
import os
import requests

wordToTensor = lambda sInput, tokenMapping: [tokenMapping[letter] for letter in sInput] # to get numerical tensor to feed into nn.Embedding function (each letter has it's index)
tensorToWord = lambda sIndexes, tokenMapping: [list(tokenMapping.keys())[i.item()] for i in sIndexes]

def getRawData():
  data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
  data = requests.get(data_url).text

  print(f"length of raw dataset: {len(data):,}")

  # get all the unique characters that occur in this text
  possibleTokens = sorted(list(set(data)))

  return data, possibleTokens, len(possibleTokens)

def outputWords(indices, tokenMapping):
  s=''
  for els in indices:
    for ss in tensorToWord(els, tokenMapping):
      s += ss
  return s
  
def loadModel(model, pathReq):
  model.load_state_dict(torch.load(pathReq))
  model.eval()
  return model

def generateScript(finalTransformer, blockSize, outputSizeWanted, numericRepresentation, processor, tokenMapping=None):
    
#   finalTransformer.to(device = processor)
  
  outString = ''
  i = 0
  inferenceData = numericRepresentation[:blockSize] #get next batch

  outString = outputWords(torch.Tensor(inferenceData).int().unsqueeze(0), tokenMapping)#just to have the initial output properly

  while i <= (outputSizeWanted - outputSizeWanted % blockSize): #subtract whatever is remaining in the block size if it's not exactly divisible by the size wanted
    gPos = torch.arange(0,blockSize, dtype=torch.long).to(device = processor) #adding pos embedding

    loss, outputLogits = finalTransformer(torch.tensor(inferenceData).unsqueeze(0).to(device = processor), torch.tensor(inferenceData).unsqueeze(0).to(device = processor), gPos, training=False, softmax=False)

    outputLogits = outputLogits[:, -1, :] #last token in the timestep

    i += 1

    probs = F.softmax(outputLogits, dim=-1) # (B, C)

    ind = torch.multinomial(probs.float(), num_samples=1).to(device = processor) # (B, 1)


    inferenceData = torch.cat((torch.tensor(inferenceData).to(device = processor), ind.long().squeeze(0)), dim = 0) #adding newest predictions into the inference data itself so it generates the string on its own


    outString += tensorToWord(ind, tokenMapping)[0]

    if len(inferenceData) >= blockSize:
      inferenceData = inferenceData[-blockSize:] #making sure next batch is less than blockSize

    if i % 100 == 0:
      print(outString)
      outString = ''


def runGenerationProcess():
    
    data, gTokens, gVocabSize = getRawData()
    vocab = gTokens # !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
    tokenMapping = {k:v for v, k in enumerate(vocab)} #make dict where value is index of token and key is token itself
    
    numericRepresentation = wordToTensor(data,tokenMapping) #changing character to index-based integer

    parser = argparse.ArgumentParser()
    parser.add_argument("--outputSize", required = False, type=int, default = 5000)
    parser.add_argument("--modelPath", required = True)
    parser.add_argument("--usedProcessor", required = False, default = 'cpu')
    
    args = parser.parse_args()
    
    path = args.modelPath

    finalTransformer = torch.load(path).to(device = args.usedProcessor)
    blockSize = finalTransformer.state_dict()['embedPos.weight'].shape[0]
    
    print('loading with parameters: ', args )
    
    generateScript(finalTransformer, blockSize, args.outputSize, numericRepresentation, args.usedProcessor, tokenMapping)
    
    


    
if __name__ == '__main__':
    runGenerationProcess()

    
