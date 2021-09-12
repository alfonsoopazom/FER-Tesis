import numpy
from ChefsHatGym.Agents import IAgent
import random
from ChefsHatGym.Rewards import RewardOnlyWinning

class AgentNaive_Random(IAgent.IAgent):

    def __init__(self, name):

        self.name = "RANDOM_" + name
        self.reward = RewardOnlyWinning.RewardOnlyWinning()
        self.rewardAcc = []
        self.sumRewardAcc = []
        self.sumRewards = 0

    def getAction(self,  observations):

        possibleActions = observations[28:]

        itemindex = numpy.array(numpy.where(numpy.array(possibleActions) == 1))[0].tolist()

        random.shuffle(itemindex)
        aIndex = itemindex[0]
        a = numpy.zeros(200)
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

        self.rewardAcc.append(self.reward.getReward(thisPlayer, matchFinished))
        self.sumRewards += self.reward.getReward(thisPlayer, matchFinished)
        self.sumRewardAcc.append(self.sumRewards)

        if thisPlayer != -1:
            self.saveValuesInATxt(str(thisPlayer), "dataRandom/randomPosition.txt")
            self.saveValuesInATxt(self.sumRewardAcc, "dataRandom/randomSumRewardsForSteps.txt")

        return self.reward.getReward(thisPlayer, matchFinished)

    def saveValuesInATxt(self, values, file_name):
        with open(file_name, 'a+') as val_file:
            val_file.write(','.join(map(str, values)) + "\n")

    def restartRewardsValues(self):
        self.rewardAcc = []
        self.sumRewardAcc = []