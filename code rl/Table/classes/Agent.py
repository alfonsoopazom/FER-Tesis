# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6 17:43:35 2015

@author: cruz
"""

import numpy as np
import Variables

class Agent(object):

    alpha = 0.3#0.1 #0.7
    gamma = 0.9 #0.4
    epsilon = 0.1 #0.25

    def __init__(self, scenario):
        self.scenario = scenario
        self.numberOfStates = self.scenario.getNumberOfStates()
        self.numberOfActions = self.scenario.getNumberOfActions()
        self.Q = np.random.uniform(0.0,0.01,(self.numberOfStates,self.numberOfActions))
        self.feedbackAmount = 0
    #end of __init__ method

    def selectAction(self, state):
        if (np.random.rand() <= self.epsilon):
            action = np.random.randint(self.numberOfActions)
        else:
            action = np.argmax(self.Q[state,:])
        #endIf
        return action
    #end of selectAction method
        
    def actionByFeedback(self, state, teacherAgent, feedbackProbability):
        if (np.random.rand() < feedbackProbability):
            #get advice
            action = np.argmax(teacherAgent.Q[state,:])
            self.feedbackAmount += 1
        else:
            action = self.selectAction(state)
        #endIf
        return action
    #end of actionByFeedback
    
    def train(self, episodes, teacherAgent=None, feedbackProbability=0):
        contCatastrophic = 0
        contFinalReached = 0
        steps = np.zeros(episodes)
        rewards = np.zeros(episodes)
        
        for i in range(episodes):
            contSteps = 0
            accReward = 0
            self.scenario.resetScenario()
            state = self.scenario.getState()
            action = self.actionByFeedback(state, teacherAgent, feedbackProbability)

            #expisode
            while True:
                #perform action
                self.scenario.executeAction(action)
                contSteps += 1

                #get reward
                reward = self.scenario.getReward()
                accReward += reward
                #catastrophic state

                stateNew = self.scenario.getState()

                if reward == Variables.punishment:
                    contCatastrophic += 1
                    self.Q[state,action] = -0.1
                    break

                actionNew = self.actionByFeedback(stateNew, teacherAgent, feedbackProbability)

                # updating Q-values
                self.Q[state, action] += self.alpha * (reward + self.gamma * 
                                         self.Q[stateNew,actionNew] - 
                                         self.Q[state,action])

                if reward == Variables.reward:
                    contFinalReached += 1
                    break


                state = stateNew
                action = actionNew
            #end of while
            steps[i] = contSteps
            rewards[i]=accReward
        #end of for
        return steps,rewards
    #end of train method

#end of class Agent
