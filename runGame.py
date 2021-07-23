import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from ChefsHatGym.Agents import Agent_Introspection
from ChefsHatGym.Agents import Agent_Naive_Random
from ChefsHatGym.Agents import PPO
from ChefsHatGym.Rewards import RewardOnlyWinning
from ChefsHatGym.env import ChefsHatEnv

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

"""General parameters"""

rew = []
qvaluesAux = []
height = 200
promQ = np.zeros([1, 200])
index = 0

"""Game parameters"""
gameType = ChefsHatEnv.GAMETYPE["MATCHES"]

gameStopCriteria = 300

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
episodes = 1

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

        while not info["validAction"]:  # aca estan los steps del juego (descartar cartas o pasar)

            nextobs, reward, isMatchOver, info = env.step(action)

        currentPlayer.actionUpdate(observations, nextobs, action, reward, info)

        if isMatchOver:  # aca es cuando un jugador se descarta la mano

            for p in playersAgents:
                p.matchUpdate(info)

            print("-------------")
            print("Match:" + str(info["matches"]))
            print("Score:" + str(info["score"]))
            print("Performance:" + str(info["performanceScore"]))
            print("-------------")


df = pd.read_csv('dataIntrospection/probabilityForGame.txt', header=None, delimiter="\t")
df['sum'] = df.sum(axis=1)

# print(df.loc[[626]])
# print(df['sum'])
# print(df.shape[0]) # imprime el numero de filas
# print(df.shape[1]) # imprime el numero de columnas

# Plotlty Express
# fig = px.line(Qvalues[0], title='Maximos Q-Values')
# fig.show()

# fig2 = px.line(prom, title='Promedios Q-values')
# fig2.show()

'''Ploteo del agente introspection'''
plt.plot(agent1.promProb, Label='Probabilidades de exito')
plt.title("Promedios probabilidades de exito")
plt.xlabel('games')
plt.ylabel('probabilidades')
plt.legend()
plt.savefig("imagesIntrospection/prom_probabilidades.png")
plt.close()

plt.plot(list(range(height)), agent1.Qvalues[0], Label='Q-values final State')
plt.title("Q-Values Introspection entrenamiento")
plt.ylabel('Q-Values')
plt.legend()
plt.savefig("imagesIntrospection/valores_q.png")
plt.close()

plt.plot(agent1.epsilonArr, label="Valores de Epsilon")
plt.title("Epsilon de Introspection entrenamiento")
plt.xlabel('Rondas')
plt.ylabel('Epsilon')
plt.legend()
plt.savefig("imagesIntrospection/valores_epsilon.png")
plt.close()

plt.plot(agent1.rewardAcc, label="Rewards")
plt.title("Recompensas del Introspection")
plt.xlabel('steps')
plt.ylabel('rewards')
plt.legend()
plt.savefig("imagesIntrospection/valores_recompensas.png")
plt.close()

plt.plot(agent1.mse, label="Error cuadratico medio")
plt.title("Error cuadratico medio Introspection")
plt.xlabel('Games')
plt.ylabel('ECM')
plt.legend()
plt.savefig("imagesIntrospection/valores_ECM.png")
plt.close()

'''Ploteo del agente PPO'''
'''plt.plot(prom, Label='Promedios Q-values')
plt.title("Q-Values Introspection entrenamiento")
plt.xlabel('Steps')
plt.ylabel('Q-Values')
plt.legend()
plt.savefig("prom_q_values.png")
plt.close()'''

plt.plot(list(range(height)), agent2.Qvalues[0], Label='Q-values final State')
plt.title("Q-Values Introspection entrenamiento")
plt.ylabel('Q-Values')
plt.legend()
plt.savefig("imagesPPO/valores_q.png")
plt.close()

plt.plot(agent2.epsilonArr, label="Valores de Epsilon")
plt.title("Epsilon de Introspection entrenamiento")
plt.xlabel('Rondas')
plt.ylabel('Epsilon')
plt.legend()
plt.savefig("imagesPPO/valores_epsilon.png")
plt.close()

plt.plot(agent2.rewardAcc, label="Rewards")
plt.title("Recompensas del Introspection")
plt.xlabel('steps')
plt.ylabel('rewards')
plt.legend()
plt.savefig("imagesPPO/valores_recompensas.png")
plt.close()

plt.plot(agent2.lossNetwork, label="Error cuadratico medio")
plt.title("Error cuadratico medio Introspection")
plt.xlabel('Games')
plt.ylabel('ECM')
plt.legend()
plt.savefig("imagesPPO/valores_ECM.png")
plt.close()


"""Evaluate Agent"""
# env.startExperiment(rewardFunctions=rewards, gameType=gameType, stopCriteria=gameStopCriteria, playerNames=agentNames,
#                   logDirectory=saveDirectory, verbose=verbose, saveDataset=True, saveLog=True)

"""Start Environment"""
'''for a in range(episodes):

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
            print("-------------")'''
