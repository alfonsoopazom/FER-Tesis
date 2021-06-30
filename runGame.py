import gym
import matplotlib.pyplot as plt
import numpy as np
from ChefsHatGym.Agents import Agent_Introspection
from ChefsHatGym.Agents import Agent_Naive_Random
from ChefsHatGym.Agents import PPO
from ChefsHatGym.Rewards import RewardOnlyWinning
from ChefsHatGym.env import ChefsHatEnv

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

Qvalues = np.zeros([1, 200])
rew = []
epsilon = []
gamma = 0.9

q_values_file = open("qValues.txt", "w")

"""Game parameters"""
gameType = ChefsHatEnv.GAMETYPE["MATCHES"]

gameStopCriteria = 10
rewardFunction = RewardOnlyWinning.RewardOnlyWinning()

"""Player Parameters"""
agent1 = Agent_Introspection.DQL(name="Introspection")
agent2 = PPO.PPO(name="PPO")
agent3 = Agent_Naive_Random.AgentNaive_Random("Random3")
agent4 = Agent_Naive_Random.AgentNaive_Random("Random4")
agentNames = [agent1.name, agent2.name, agent3.name, agent4.name]
playersAgents = [agent1, agent2, agent3, agent4]

rewards = []

for r in playersAgents:
    rewards.append(r.getReward)

"""Experiment parameters"""
saveDirectory = "examples/"
verbose = False
saveLog = False
saveDataset = False
episodes = 50

"""Setup environment"""
env = gym.make('chefshat-v0')  # starting the game Environment
env.startExperiment(rewardFunctions=rewards, gameType=gameType, stopCriteria=gameStopCriteria, playerNames=agentNames,
                    logDirectory=saveDirectory, verbose=verbose, saveDataset=True, saveLog=True)

"""Start Environment"""
for a in range(episodes):

    observations = env.reset()

    while not env.gameFinished:

        currentPlayer = playersAgents[env.currentPlayer]
        observations = env.getObservation()
        action = currentPlayer.getAction(observations)

        info = {"validAction": False}

        while not info["validAction"]:
            nextobs, reward, isMatchOver, info = env.step(action)

            agent1.actionUpdate(observations, nextobs, action, reward, info)
            epsilon.append(agent1.epsilon)
            rew.append(reward)
            Qvalues = agent1.matchUpdate(info)

            #print(len(Qvalues))
            # print(epsilon)
            # agente = agent1.buildSimpleModel().summary()

        if isMatchOver:
            '''for p in playersAgents:
                p.matchUpdate(info)'''

            print("-------------")
            print("Match:" + str(info["matches"]))
            print("Score:" + str(info["score"]))
            print("Performance:" + str(info["performanceScore"]))
            print("-------------")

#agent1.pSuccess(Qvalues, rew)

for row in Qvalues:
    np.savetxt(q_values_file, row)
q_values_file.close()

plt.plot(Qvalues)
plt.title("Q-Values de Introspection")
plt.xlabel('Episodios')
plt.ylabel('Q-Values')
plt.xlim([0, 10])
plt.legend()
plt.savefig("q_values.png")
plt.close()

plt.plot(epsilon, label="Valores de Epsilon")
plt.title("Epsilon de Introspection")
plt.xlabel('Episodios')
plt.ylabel('Epsilon')
plt.legend()
plt.savefig("valores_epsilon.png")
plt.close()

plt.plot(rew, label="Recompensas ")
plt.title("Recompensas de Introspection")
plt.xlabel('Episodios')
plt.ylabel('Rewards')
plt.legend()
plt.ylim([-0.5, 2])
plt.savefig("valores_recompensas.png")
plt.close()

"""Evaluate Agent"""
env.startExperiment(rewardFunctions=rewards, gameType=gameType, stopCriteria=gameStopCriteria, playerNames=agentNames,
                    logDirectory=saveDirectory, verbose=verbose, saveDataset=True, saveLog=True)

"""Start Environment"""
for a in range(episodes):

    observations = env.reset()

    while not env.gameFinished:
        currentPlayer = playersAgents[env.currentPlayer]

        observations = env.getObservation()
        action = currentPlayer.getAction(observations)

        info = {"validAction": False}

        while not info["validAction"]:
            nextobs, reward, isMatchOver, info = env.step(action)

        if isMatchOver:
            print("-------------")
            print("Match:" + str(info["matches"]))
            print("Score:" + str(info["score"]))
            print("Performance:" + str(info["performanceScore"]))
            print("-------------")
