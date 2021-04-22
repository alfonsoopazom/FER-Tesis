#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2011-03-22

"""

#******************************************************************************
# Libraries Declaration
import numpy as N

#******************************************************************************
#******************************************************************************
class Room(object):
#------------------------------------------------------------------------------
  def __init__(self, size_x, size_y):
    self.y = size_y
    self.x = size_x
    self.R = N.zeros(size_x*size_y)
    self.agentPos = 0
    self.goalState = self.xy2idx(8,18)#self.x*self.y-1
    self.R[self.goalState] = 1.0
    print("Goal position: "), self.xy2idx(8,18)
    # set fear region
    self.R[self.xy2idx(6,6)] = -1.0
    #self.R[self.xy2idx(6,7)] = -1.0
    #self.R[self.xy2idx(5,6)] = -1.0
    #self.R[self.xy2idx(5,7)] = -1.0
  # end of the function

#------------------------------------------------------------------------------
  def idx2xy(self,idx):
    x = idx / self.y
    y = idx % self.y
    return x, y
  # end of the function

#------------------------------------------------------------------------------
  def xy2idx(self,x,y):
    return x*self.y + y
  # end of the function

#------------------------------------------------------------------------------
  def resetAgent(self, pos):
    self.agentPos = pos
  # end of the function

#------------------------------------------------------------------------------
  def getState(self):
    return self.agentPos
  # end of the function

#------------------------------------------------------------------------------
  def getReward(self):
    if self.agentPos == -1:
      return -1.0
    else:
      return self.R[self.agentPos]
  # end of the function

#------------------------------------------------------------------------------
  def getNumOfActions(self):
    return 4
  # end of the function

#------------------------------------------------------------------------------
  def getNumOfStates(self):
    return self.x*self.y
  # end of the function

#------------------------------------------------------------------------------
  def move(self,id):
    x_, y_ = self.idx2xy(self.agentPos)
    tmpX = x_
    tmpY = y_
    if id == 0: # move right
      tmpX += 1
    elif id == 1: # move left
      tmpX -= 1
    elif id == 2: # move up
      tmpY += 1
    elif id == 3: # move down
      tmpY -= 1
    else:
      print("ERROR: Unknown action")

    if self.validMove(tmpX, tmpY):
      self.agentPos = self.xy2idx(tmpX,tmpY)
    else:
      self.agentPos = -1
  # end of the function

#------------------------------------------------------------------------------
  def validMove(self,x,y):
    valid = True
    if x < 0 or x >= self.x:
      valid = False
    if y < 0 or y >= self.y:
      valid = False
    return valid
  # end of the function


#******************************************************************************
#******************************************************************************
class RL(object):
  def __init__(self, world):
    self.world = world
    self.numOfActions = self.world.getNumOfActions()
    self.numOfStates = self.world.getNumOfStates()
    self.Q = N.random.uniform(0.0,0.01,(self.numOfStates,self.numOfActions))
    self.mu = 0.7
    self.gamma = 0.4
    self.epsilon = 0.25
  # end of the function

#------------------------------------------------------------------------------
  def train(self, iter):
    for itr in range(iter):
      if itr < 5:
        print("iter: %d"%itr)
      if itr%100 == 0:
        print("iter: %d"%itr)
      state = N.random.randint(0,self.numOfStates)
      self.world.resetAgent(state)
      # choose action
      # epsilon-greedy action selection
      if (N.random.rand() <= self.epsilon):
        a = N.random.randint(self.numOfActions)
      else:
        a = N.argmax(self.Q[state,:])
      reward = 0
      expisode = True
      while expisode:
        # perform action
        self.world.move(a)
        # look for reward
        reward = self.world.getReward()
        state_new = self.world.getState()
        if state == -1:
          self.Q[state,a] = -0.1
          # agent left terrain
          break
        # new action
        if (N.random.rand() <= self.epsilon):
          a_new = N.random.randint(self.numOfActions)
        else:
          a_new = N.argmax(self.Q[state,:])
        # update Q-values
        self.Q[state,a] += self.mu*(reward +
                                    #self.gamma*N.max(self.Q[state_new])-
                                    self.gamma*self.Q[state_new,a_new]-
                                    self.Q[state,a])
        if reward == 1.0:
          expisode = False
        state = state_new
        a = a_new
      # end of while loop
    # end of big for loop
    print (self.Q)
    N.savetxt("QValues.csv", self.Q, delimiter=",")
  # end of the function


#******************************************************************************
#******************************************************************************
if __name__ == "__main__":
  room = Room(12,22)
  learner = RL(room)
  learner.train(1000)
# end of the function

#******************************************************************************
#******************************************************************************
