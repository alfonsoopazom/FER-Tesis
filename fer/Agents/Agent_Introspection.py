import copy
import os
import random
import sys
import urllib.request

import numpy
import numpy as np
from ChefsHatGym.Agents import IAgent
from ChefsHatGym.Rewards import RewardOnlyWinning
from ChefsHatGym.Util import MemoryBuffer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


class DQL(IAgent.IAgent):
    epsilon = 0.6
    name = ""
    actor = None

    loadFrom = {"vsRandom": "Trained/dql_vsRandom.hd5",
                "vsEveryone": "Trained/dql_vsEveryone.hd5",
                "vsSelf": "Trained/dql_vsSelf.hd5", }
    downloadFrom = {
        "vsRandom": "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/dql_vsRandom.hd5",
        "vsEveryone": "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/dql_vsEveryone.hd5",
        "vsSelf": "https://github.com/pablovin/ChefsHatPlayersClub/raw/main/playersClub/src/ChefsHatPlayersClub/Agents/Classic/Trained/dql_vsSelf.hd5", }

    def __init__(self, name, continueTraining=True, type="Scratch", initialEpsilon=epsilon, loadNetwork="",
                 saveFolder="", verbose=True):

        self.reward = RewardOnlyWinning.RewardOnlyWinning()
        self.numberActions = 200
        self.numberStates = 28
        self.Qvalues = np.zeros([1, 200])

        self.training = continueTraining
        self.initialEpsilon = initialEpsilon
        self.name += name
        self.loadNetwork = loadNetwork
        self.saveModelIn = saveFolder
        self.verbose = verbose
        self.type = type
        self.rewardAcc = []
        self.sumReward = 0
        self.mse = []
        self.lossNetwork = []
        self.probSucc = []
        self.promProb = []
        self.sumRewardAcc = []
        self.sumRewards = 0
        self.avgRewards = []
        self.epsilonArr = []
        self.number = 0

        self.startAgent()

        if not type == "Scratch":
            fileName = os.path.abspath(sys.modules[DQL.__module__].__file__)[0:-6] + self.loadFrom[type]
            if not os.path.exists(os.path.abspath(sys.modules[DQL.__module__].__file__)[0:-6] + "/Trained/"):
                os.mkdir(os.path.abspath(sys.modules[DQL.__module__].__file__)[0:-6] + "/Trained/")

            if not os.path.exists(fileName):
                urllib.request.urlretrieve(self.downloadFrom[type], fileName)

            self.loadModel(fileName)

        if not loadNetwork == "":
            self.loadModel(loadNetwork)

    def getAction(self, observations):

        stateVector = np.concatenate((observations[0:11], observations[11:28]))
        possibleActions = observations[28:]

        stateVector = np.expand_dims(stateVector, 0)
        possibleActions = np.array(possibleActions)
        possibleActions2 = copy.copy(possibleActions)

        # ramdom actions - e-greedy
        if np.random.rand() <= self.epsilon:
            itemindex = np.array(np.where(np.array(possibleActions2) == 1))[0].tolist()
            random.shuffle(itemindex)
            aIndex = itemindex[0]
            a = np.zeros(200)
            a[aIndex] = 1

        else:
            # Q-values
            possibleActionsVector = np.expand_dims(np.array(possibleActions2), 0)
            a = self.actor.predict([stateVector, possibleActionsVector])[0]

        if np.max(a) != 1:
            if self.Qvalues[0, (np.argmax(a))] < np.max(a):
                self.Qvalues[0, (np.argmax(a))] = np.max(a)

        self.saveValuesInATxt(self.Qvalues[0], "dataIntrospection/QValuesForRound.txt")

        return a

    def observeOthers(self, envInfo):
        pass

    def getReward(self, info, stateBefore, stateAfter):

        thisPlayer = info["thisPlayerPosition"]
        matchFinished = info["thisPlayerFinished"]

        self.rewardAcc.append(self.reward.getReward(thisPlayer, matchFinished))

        self.sumRewards += self.reward.getReward(thisPlayer, matchFinished)
        self.sumRewardAcc.append(self.sumRewards)

        if thisPlayer != -1:
            self.saveValuesInATxt(str(thisPlayer), "dataIntrospection/introspectionPosition.txt")
            self.saveValuesInATxt(self.sumRewardAcc, "dataIntrospection/introspectionSumRewardsForSteps.txt")

        return self.reward.getReward(thisPlayer, matchFinished)

    def selectAction(self, observations):
        # exploracion
        if np.random.rand() <= self.epsilon:
            # Numero de acciones posibles
            return np.random.randint(self.numberActions)
        # explotacion
        else:
            # Posicion del valor maximo del arreglo
            positionMaxValue = observations[28:]
            return np.argmax(positionMaxValue)

    def startAgent(self):

        self.hiddenLayers = 1
        self.hiddenUnits = 256
        self.batchSize = 10
        self.tau = 0.52  # target network update rate

        self.gamma = 0.6  # discount rate
        self.loss = "mse"

        self.epsilon_min = 0.1
        self.epsilon_decay = 0.990

        # self.tau = 0.1 #target network update rate
        if self.training:
            self.epsilon = self.initialEpsilon  # exploration rate while training
        else:
            self.epsilon = 0.0  # no exploration while testing

        self.prioritized_experience_replay = False
        self.dueling = False

        QSize = 20000
        self.memory = MemoryBuffer.MemoryBuffer(QSize, self.prioritized_experience_replay)

        self.learning_rate = 0.5
        self.buildModel()

    def buildModel(self):

        self.buildSimpleModel()
        self.actor.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate), metrics=["mse"])
        self.targetNetwork.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate), metrics=["mse"])

    def buildSimpleModel(self):
        """Build the DQN"""

        def model():
            inputSize = 28
            # Se crea la entrada a la red. La capa de entrada es de tamaño 28.
            inputLayer = layers.Input(shape=(inputSize,), name="States")

            # dense = Dense(self.hiddenLayers, activation="relu", name="dense_0")(inputLayer)
            for i in range(self.hiddenLayers + 1):

                if i == 0:
                    previous = inputLayer
                else:
                    previous = dense

                dense = layers.Dense(self.hiddenUnits * (i + 1), name="Dense" + str(i), activation="relu")(previous)

            if (self.dueling):
                # Have the network estimate the Advantage function as an intermediate layer
                dense = layers.Dense(self.outputSize + 1, activation='linear', name="duelingNetwork")(dense)
                dense = layers.Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                                      output_shape=(200,))(dense)

            possibleActions = layers.Input(shape=(200,), name="PossibleAction")

            dense = layers.Dense(200, activation='softmax')(dense)
            output = layers.Multiply()([possibleActions, dense])
            # probOutput =  Dense(self.outputSize, activation='softmax')(dense)

            return keras.Model([inputLayer, possibleActions], output)

        self.actor = model()
        self.targetNetwork = model()

    def loadModel(self, model):

        self.actor = load_model(model)
        self.targetNetwork = load_model(model)

    def updateTargetNetwork(self):

        W = self.actor.get_weights()
        tgt_W = self.targetNetwork.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.targetNetwork.set_weights(tgt_W)
        return tgt_W

    def updateModel(self, game, thisPlayer):
        """ Train Q-network on batch sampled from the buffer"""
        # Sample experience from memory buffer (optionally with PER)
        s, a, r, d, new_s, possibleActions, newPossibleActions, idx = self.memory.sample_batch(self.batchSize)

        # Apply Bellman Equation on batch samples to train our DQN
        q = self.actor.predict([s, possibleActions])
        next_q = self.actor.predict([new_s, newPossibleActions])
        q_targ = self.targetNetwork.predict([new_s, newPossibleActions])

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = np.argmax(next_q[i, :])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]

        # Train on batch
        history = self.actor.fit([s, possibleActions], q, verbose=False)

        if (game + 1) % 5 == 0 and not self.saveModelIn == "":
            self.actor.save(self.saveModelIn + "/actor_iteration_" + str(game) + "_Player_" + str(thisPlayer) + ".hd5")
            print(self.saveModelIn)

        if self.verbose:
            print("-- " + self.name + ": Epsilon:" + str(self.epsilon) + " - Loss:" + str(history.history['loss']))

            self.mse.append(history.history["mse"])
            self.lossNetwork.append(history.history['loss'])
            self.epsilonArr.append(self.epsilon)

            self.saveValuesInATxt(self.epsilonArr, "dataIntrospection/epsilonValues.txt")

    def memorize(self, state, action, reward, next_state, done, possibleActions, newPossibleActions):

        if (self.prioritized_experience_replay):
            state = np.expand_dims(np.array(state), 0)
            next_state = np.expand_dims(np.array(next_state), 0)
            q_val = self.actor.predict(state)
            q_val_t = self.targetNetwork.predict(next_state)
            next_best_action = np.argmax(q_val)
            new_val = reward + self.gamma * q_val_t[0, next_best_action]
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0

        self.memory.memorize(state, action, reward, done, next_state, possibleActions, newPossibleActions, td_error)

    def actionUpdate(self, observation, nextObservation, action, reward, info):

        if self.training:
            done = info["thisPlayerFinished"]

            state = np.concatenate((observation[0:11], observation[11:28]))
            possibleActions = observation[28:]

            next_state = np.concatenate((nextObservation[0:11], nextObservation[11:28]))
            newPossibleActions = nextObservation[28:]

            action = np.argmax(action)

            self.memorize(state, action, reward, next_state, done, possibleActions, newPossibleActions)

    def matchUpdate(self, info):

        if self.training:
            rounds = info["rounds"]
            thisPlayer = info["thisPlayer"]
            if self.memory.size() > self.batchSize:
                self.updateModel(rounds, thisPlayer)
                self.updateTargetNetwork()
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

        self.promRewards(self.sumRewardAcc)
        self.probability()
        self.saveValuesInATxt(self.rewardAcc, "dataIntrospection/rewardsForGame.txt")
        self.saveValuesInATxt(self.Qvalues[0], "dataIntrospection/QValuesForGame.txt")

    def probability(self):

        rewCalculate = 1 + (int(sum(self.sumRewardAcc)) * (-0.01))
        probability = self.pSuccess(self.Qvalues[0], rewCalculate, 0.9)
        print(probability)

        self.probSucc.append(probability)
        self.promProbability(probability)

        self.saveValuesInATxt(probability, "dataIntrospection/probabilityForGame.txt")

    def saveValuesInATxt(self, values, file_name):
        with open(file_name, 'a+') as val_file:
            val_file.write(','.join(map(str, values)) + "\n")

    def pSuccess(self, Q, reward, gamma):
        n = np.log(Q / reward) / np.log(gamma)
        log10baseGamma = np.log(10) / np.log(gamma)
        probOfSuccess = (n / (2 * log10baseGamma)) + 1
        probOfSuccessLimit = np.minimum(1, np.maximum(0, probOfSuccess))

        return probOfSuccessLimit

    def promProbability(self, probFinal):
        self.promProb.append(numpy.average(probFinal))
        self.saveValuesInATxt(self.promProb, "dataIntrospection/promediosProbability.txt")

    def promRewards(self, rewards):
        self.avgRewards.append(numpy.average(rewards))

    def restartRewardsValues(self):
        self.rewardAcc = []
        self.sumRewardAcc = []
