import numpy as np
import random
from ChefsHatGym.Agents import IAgent
from ChefsHatGym.Rewards import RewardOnlyWinning

class AgentIntrospection(IAgent.IAgent):

    alpha = 0.3  # 0.1 #0.7
    gamma = 0.9  # 0.4
    epsilon = 0.5  # 0.25
    qValues = np.zeros([11, 200])

    def __init__(self, name="Introspection"):

        self.name = "Introspection"+name
        self.reward = RewardOnlyWinning.RewardOnlyWinning()

    def getAction(self, observations):
        possibleActions = observations[28:]

        itemindex = np.array(np.where(np.array(possibleActions) == 1))[0].tolist()

        random.shuffle(itemindex)
        aIndex = itemindex[0]
        a = np.zeros(200)
        a[aIndex] = 1

        return a

    def actionUpdate(self, observations, nextobs, action, reward, info):
        pass

    def observeOthers(self, envInfo):
        pass

    def matchUpdate(self, envInfo):
        pass

    def getReward(self, info, stateBefore, stateAfter):

        thisPlayer = info["thisPlayerPosition"]
        matchFinished = info["thisPlayerFinished"]

        return self.reward.getReward(thisPlayer, matchFinished)
