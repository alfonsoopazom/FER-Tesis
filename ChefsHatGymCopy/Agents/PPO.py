#Adapted from: https://github.com/LuEE-C/PPO-Keras/blob/master/Main.py
import random
import os
import sys
import urllib.request
import numpy
import copy

import numpy as np
import tensorflow.keras.backend as K

from ChefsHatGym.Agents import IAgent
from ChefsHatGym.Rewards import RewardOnlyWinning
from tensorflow import keras
from tensorflow.keras.layers import Dense, Multiply
from tensorflow.keras import layers, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model

def proximal_policy_optimization_loss():
    def loss(y_true, y_pred):
        LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
        ENTROPY_LOSS = 5e-3
        y_tru_valid = y_true[:, 0:200]
        old_prediction = y_true[:, 200:400]
        advantage = y_true[:, 400][0]

        prob = K.sum(y_tru_valid * y_pred, axis=-1)
        old_prob = K.sum(y_tru_valid * old_prediction, axis=-1)
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING,
                                                       max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(
                    prob * K.log(prob + 1e-10)))
    return loss

#Adapted from: https://github.com/germain-hug/Deep-RL-Keras

types=["Scratch", "vsRandom", "vsEveryone", "vsSelf"]

class PPO(IAgent.IAgent):

    epsilon = 0.9
    name = ""
    actor = None
    training = False


    loadFrom = {"vsRandom":["Trained/ppo_actor_vsRandom.hd5","Trained/ppo_critic_vsRandom.hd5"],
            "vsEveryone":["Trained/ppo_actor_vsEveryone.hd5","Trained/ppo_critic_vsEveryone.hd5"],
                "vsSelf":["Trained/ppo_actor_vsSelf.hd5","Trained/ppo_critic_vsSelf.hd5"]}

    downloadFrom = {"vsRandom":["https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/ppo_actor_vsRandom.hd5",
                                "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/ppo_critic_vsRandom.hd5"],
            "vsEveryone":["https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/ppo_actor_vsEveryone.hd5",
                          "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/ppo_critic_vsEveryone.hd5"],
                "vsSelf":["https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/ppo_actor_vsSelf.hd5",
                          "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/ppo_critic_vsSelf.hd5"],}


    def __init__(self,name, continueTraining=True, type="Scratch", initialEpsilon=epsilon, loadNetwork="", saveFolder="", verbose=True):
        self.training = continueTraining
        self.initialEpsilon = initialEpsilon
        self.name += name
        self.loadNetwork = loadNetwork
        self.saveModelIn = saveFolder
        self.verbose = verbose

        self.type = type
        self.reward = RewardOnlyWinning.RewardOnlyWinning()
        self.rewardAcc = []
        self.Qvalues = np.zeros([1, 200])
        self.lossNetwork = []
        self.epsilonArr = []
        self.probSucc = []
        self.promProb = []
        self.sumRewardAcc = []
        self.sumRewards = 0
        self.number = 0
        self.avgRewards = []

        self.startAgent()

        if not type == "Scratch":
            fileNameActor = os.path.abspath(sys.modules[PPO.__module__].__file__)[0:-6] + self.loadFrom[type][0]
            fileNameCritic = os.path.abspath(sys.modules[PPO.__module__].__file__)[0:-6] + self.loadFrom[type][1]
            if not os.path.exists(os.path.abspath(sys.modules[PPO.__module__].__file__)[0:-6] + "/Trained/"):
                os.mkdir(os.path.abspath(sys.modules[PPO.__module__].__file__)[0:-6] + "/Trained/")

            if not os.path.exists(fileNameCritic):
                urllib.request.urlretrieve(self.downloadFrom[type][0], fileNameActor)
                urllib.request.urlretrieve(self.downloadFrom[type][1], fileNameCritic)

            self.loadModel([fileNameActor,fileNameCritic])

        if not loadNetwork == "":
            self.loadModel(loadNetwork)

    def startAgent(self):

        self.hiddenLayers = 1
        self.hiddenUnits = 256
        self.gamma = 0.6  # discount rate

        #Game memory
        self.resetMemory()

        self.learning_rate = 1e-4

        if self.training:
            self.epsilon = self.initialEpsilon  # exploration rate while training
        else:
            self.epsilon = 0.0 #no exploration while testing

        self.epsilon_min = 0.1
        self.epsilon_decay = 0.01

        self.buildModel()

    def buildActorNetwork(self):

        inputSize = 28
        inp = Input((inputSize,), name="Actor_State")

        for i in range(self.hiddenLayers + 1):
            if i == 0:
                previous = inp
            else:
                previous = dense

            dense = Dense(self.hiddenUnits * (i + 1), name="Actor_Dense" + str(i), activation="relu")(previous)

        outputActor = Dense(200, activation='softmax', name="actor_output")(dense)

        actionsOutput = Input(shape=(200,), name="PossibleActions")

        outputPossibleActor = Multiply()([actionsOutput, outputActor])

        self.actor = Model([inp,actionsOutput], outputPossibleActor)

        self.actor.compile(optimizer=Adam(lr=self.learning_rate), loss=[proximal_policy_optimization_loss()])

    def getReward(self, info, stateBefore, stateAfter):

        thisPlayer = info["thisPlayerPosition"]
        matchFinished = info["thisPlayerFinished"]

        self.rewardAcc.append(self.reward.getReward(thisPlayer, matchFinished))

        self.sumRewards += self.reward.getReward(thisPlayer, matchFinished)
        self.sumRewardAcc.append(self.sumRewards)

        if thisPlayer != -1:
            self.saveValuesInATxt(str(thisPlayer), "dataPPO/ppoPosition.txt")
            self.saveValuesInATxt(self.sumRewardAcc, "dataPPO/ppoSumRewardsForSteps.txt")

        return self.reward.getReward(thisPlayer, matchFinished)

    def buildCriticNetwork(self):
        # Critic model
        inputSize = 28

        inp = Input((inputSize,), name="Critic_State")

        for i in range(self.hiddenLayers + 1):
            if i == 0:
                previous = inp
            else:
                previous = dense

            dense = Dense(self.hiddenUnits * (i + 1), name="Critic_Dense" + str(i), activation="relu")(previous)

        outputCritic = Dense(1, activation='linear', name="critic_output")(dense)

        self.critic = Model([inp], outputCritic)

        self.critic.compile(Adam(self.learning_rate), 'mse')

    def buildModel(self):

       self.buildCriticNetwork()
       self.buildActorNetwork()

    def getAction(self, observations):

        stateVector = numpy.concatenate((observations[0:11], observations[11:28]))
        possibleActions = observations[28:]

        stateVector = numpy.expand_dims(numpy.array(stateVector), 0)
        possibleActions2 = copy.copy(possibleActions)

        if numpy.random.rand() <= self.epsilon:
            itemindex = numpy.array(numpy.where(numpy.array(possibleActions2) == 1))[0].tolist()
            random.shuffle(itemindex)
            aIndex = itemindex[0]
            a = numpy.zeros(200)
            a[aIndex] = 1
        else:
            possibleActionsVector = numpy.expand_dims(numpy.array(possibleActions2), 0)
            a = self.actor.predict([stateVector, possibleActionsVector])[0]

        if np.max(a) != 1:
            if self.Qvalues[0, (np.argmax(a))] < np.max(a):
                self.Qvalues[0, (np.argmax(a))] = np.max(a)

        self.saveValuesInATxt(self.Qvalues[0], "dataPPO/QValuesForRound.txt")

        return a

    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode"""
        discounted_r, cumul_r = numpy.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r


    def loadModel(self, model):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
            ENTROPY_LOSS = 5e-3
            y_tru_valid = y_true[:, 0:200]
            old_prediction = y_true[:, 200:400]
            advantage = y_true[:, 400][0]

            prob = K.sum(y_tru_valid * y_pred, axis=-1)
            old_prob = K.sum(y_tru_valid * old_prediction, axis=-1)
            r = prob / (old_prob + 1e-10)

            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING,
                                                           max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(
                    prob * K.log(prob + 1e-10)))

        actorModel, criticModel = model
        self.actor  = load_model(actorModel, custom_objects={'loss':loss})
        self.critic = load_model(criticModel, custom_objects={'loss':loss})


    def updateModel(self, game, thisPlayer):

        state =  numpy.array(self.states)

        action = self.actions
        reward = numpy.array(self.rewards)
        possibleActions = numpy.array(self.possibleActions)
        realEncoding = numpy.array(self.realEncoding)

        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(reward)
        state_values = self.critic.predict(numpy.array(state))
        advantages = discounted_rewards - numpy.reshape(state_values, len(state_values))

        criticLoss = self.critic.train_on_batch([state], [reward])

        actions = []
        for i in range(len(action)):
            advantage = numpy.zeros(numpy.array(action[i]).shape)
            advantage[0] = advantages[i]
            # print ("advantages:" + str(numpy.array(advantage).shape))
            # print ("actions:" + str(numpy.array(action[i]).shape))
            # print("realEncoding:" + str(numpy.array(realEncoding[i]).shape))
            concatenated = numpy.concatenate((action[i], realEncoding[i], advantage))
            actions.append(concatenated)
        actions = numpy.array(actions)

        actorLoss = self.actor.train_on_batch([state, possibleActions], [actions])

        #Update the decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        #save model
        if (game + 1) % 100 == 0 and not self.saveModelIn=="":
            self.actor.save(self.saveModelIn + "/actor_iteration_" + str(game) + "_Player_"+str(thisPlayer)+".hd5")
            self.critic.save(self.saveModelIn + "/critic_iteration_"  + str(game) + "_Player_"+str(thisPlayer)+".hd5")

        if self.verbose:
            print ("-- "+self.name + ": Epsilon:" + str(self.epsilon) + " - ALoss:" + str(actorLoss) + " - " + "CLoss: " + str(criticLoss))

    def resetMemory (self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.possibleActions = []
        self.realEncoding = []

    def matchUpdate(self, info):

        if self.training:
            rounds = info["rounds"]
            thisPlayer = info["thisPlayer"]
            self.updateModel(rounds, thisPlayer)
            self.resetMemory()

        self.promRewards(self.sumRewardAcc)
        self.probability()
        self.saveValuesInATxt(self.rewardAcc, "dataPPO/rewardsForGame.txt")
        self.saveValuesInATxt(self.Qvalues[0], "dataPPO/QValuesForGame.txt")

    def actionUpdate(self, observation, nextObservation, action, reward, info):

        if self.training:
            state = numpy.concatenate((observation[0:11], observation[11:28]))
            possibleActions = observation[28:]

            realEncoding = action
            action = numpy.zeros(action.shape)
            action[numpy.argmax(realEncoding)] = 1

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.possibleActions.append(possibleActions)
            self.realEncoding.append(realEncoding)

    def probability(self):

        rewCalculate = 1 + (int(sum(self.rewardAcc)) * (-0.001))
        print(self.Qvalues[0])
        probability = self.pSuccess(self.Qvalues[0], rewCalculate, 0.65)

        self.saveValuesInATxt(probability, "dataPPO/probabilityForSteps.txt")
        self.promProbability(probability)

    def saveValuesInATxt(self, values, file_name):
        with open(file_name, 'a+') as val_file:
            val_file.write(','.join(map(str, values)) + "\n")

    def pSuccess(self, Q, reward, gamma):
        n = np.log(Q / reward) / np.log(gamma)
        log10baseGamma = np.log(10) / np.log(gamma)
        probOfSuccess = (n / (2 * log10baseGamma)) + 1
        probOfSuccessLimit = np.minimum(1, np.maximum(0, probOfSuccess))
        # probOfSuccessLimit = probOfSuccessLimit * (1 - stochasticity) #Usar solo si usamos transiciones estocasticas o el parametro sigma
        return probOfSuccessLimit

    def promProbability(self, probFinal):
        self.promProb.append(numpy.average(probFinal))
        self.saveValuesInATxt(self.promProb, "dataPPO/promediosProbability.txt")

    def promRewards(self, rewards):
        self.avgRewards.append(numpy.average(rewards))

    def restartRewardsValues(self):
        self.rewardAcc = []
        self.sumRewardAcc = []