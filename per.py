#!/usr/bin/env python

import numpy as np

from collections import namedtuple, deque
import random
import typing

import io

def setSeed(seed):
  # For Python's random module
  random.seed(seed)
  # For NumPy's random module
  np.random.seed(seed)

def printToString(*args, **kwargs):
  output = io.StringIO()
  print(*args, file=output, **kwargs)
  contents = output.getvalue()
  output.close()
  return contents

class TypedTransition(typing.NamedTuple):
  state: int
  action: int
  nextState: int
  reward: float
  tdError: float

class PrioritizedExperienceReplay():
  '''This is the rank-based variant of Prioritized Experience Replay.'''

  def __init__(self, capacity: int, sampleSize: int, alpha: float=1.0):
    self.memory = deque([], maxlen=capacity)
    self.sampleSize = sampleSize
    self.alpha = alpha
    self.currentSize = 0
    self.exclusiveBucketEnds : list[int] = []

  def recomputeBounds(self):
    # Compute buckets
    self.exclusiveBucketEnds.clear()
    sum = 0.0
    for i in range(len(self.memory)):
      sum += (1.0/(i+1))**self.alpha
    cumulativeSum = 0.0
    currentBucket = 0
    for i in range(len(self.memory)):
      priority = (1.0/(i+1))**self.alpha
      probability = priority/sum
      cumulativeSum += probability
      if cumulativeSum >= (currentBucket+1) / self.sampleSize:
        self.exclusiveBucketEnds.append(i+1)
        currentBucket += 1
        if currentBucket == self.sampleSize-1:
          self.exclusiveBucketEnds.append(len(self.memory))
          break

  def push(self, transition: TypedTransition):
    """Save a transition"""
    self.memory.append(transition)
    if len(self.memory) >= self.sampleSize and len(self.memory) != self.currentSize:
      # List grew, recompute CDF bounds
      self.recomputeBounds()
      self.currentSize = len(self.memory)

  def sample(self) -> list[(int, TypedTransition)]:
    if len(self.memory) < self.sampleSize:
      raise ValueError(f'Trying to sample {self.sampleSize}, but only have {len(self.memory)} item(s)')
    
    errorsAndIndices = sorted([(transition.tdError, index) for index, transition in enumerate(self.memory)], reverse=True)
    
    lastEnd = 0
    result : list[(int, TypedTransition)] = []
    for end in self.exclusiveBucketEnds:
      indexInBucket = np.random.randint(lastEnd, end)
      _, index = errorsAndIndices[indexInBucket]
      result.append((index, self.memory[index]))
      lastEnd = end
    return result
  
  def updateErrors(self, index:int, transition:TypedTransition):
    self.memory[index] = transition

  def __len__(self):
    return len(self.memory)
  
  def __str__(self):
    return printToString('PrioritizedExperienceReplay:', *self.memory, sep='\n  ')

# Need to:
# Insert a new item with unknown priority
# Sample N items
# Update priorities of recently sampled N items
  
if __name__ == "__main__":
  kSeed = 0x533D
  setSeed(kSeed)
  kBufferSize = 32
  kSampleSize = 4
  kAlpha = 0.0

  replayMemory = PrioritizedExperienceReplay(kBufferSize, kSampleSize, kAlpha)

  # Push some random data
  for i in range(kBufferSize):
    replayMemory.push(TypedTransition(np.random.randint(50), np.random.randint(50), np.random.randint(50), np.random.rand(), np.random.rand()))
  
  for i in range(3):
    res = replayMemory.sample()
    print('Sample:',*res, sep='\n  ')
    print()
    for x in res:
      replayMemory.updateErrors(x[0], x[1]._replace(tdError = np.random.rand()))

  # alpha = 0.5
  # sum = 0.0
  # for i in range(kBufferSize):
  #   sum += (1.0/(i+1))**alpha
  # print(f'Sum: {sum}')
  # cumSum = 0.0
  # for i in range(kBufferSize):
  #   priority = (1.0/(i+1))**alpha
  #   prob = priority/sum
  #   cumSum += prob
  #   print(f'{i+1}: {priority}/{sum} = {prob}; {cumSum}')
