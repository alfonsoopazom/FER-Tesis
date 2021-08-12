import gym
import matplotlib.pyplot as plt
import numpy as np

from ChefsHatGym.Agents import Agent_Introspection
from ChefsHatGym.Agents import Agent_Naive_Random
from ChefsHatGym.Agents import Agent_Naive_Random1
from ChefsHatGym.Agents import PPO
from ChefsHatGym.Rewards import RewardOnlyWinning
from ChefsHatGym.env import ChefsHatEnv

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

"""General parameters"""

height = 200

"""Game parameters"""
gameType = ChefsHatEnv.GAMETYPE["MATCHES"]

gameStopCriteria = 50

rewardFunction = RewardOnlyWinning.RewardOnlyWinning()

"""Player Parameters"""
agent1 = Agent_Introspection.DQL(name="Introspection")
agent2 = PPO.PPO(name="PPO")
agent3 = Agent_Naive_Random.AgentNaive_Random(name="Random3")
agent4 = Agent_Naive_Random1.AgentNaive_Random(name="Random4")
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

# df = pd.read_csv('dataIntrospection/probabilityForGame.txt', header=None, delimiter="\t")
# df['sum'] = df.sum(axis=1)

# print(df.loc[[626]])
# print(df['sum'])
# print(df.shape[0]) # imprime el numero de filas
# print(df.shape[1]) # imprime el numero de columnas

'''Ploteo del agente introspection'''
plt.plot(agent1.promProb, 'r', Label='Probabilidades de exito DQN')
#plt.plot(agent2.promProb, 'g', Label='Probabilidades de exito PPO')
plt.title("Promedios probabilidades de éxito - entrenamiento")
plt.xlabel('juegos')
plt.ylabel('probabilidades')
plt.legend()
plt.savefig("imagesTraining/prom_probabilidades.png")
plt.close()

plt.plot(list(range(height)), agent1.Qvalues[0], 'r', Label='Q-values DQN')
plt.plot(list(range(height)), agent2.Qvalues[0], 'g', Label='Q-values PPO')
plt.title("Q-Values finales - entrenamiento")
plt.ylabel('Q-Values')
plt.xlabel('Acciones')
plt.legend()
plt.savefig("imagesTraining/valores_q.png")
plt.close()

plt.plot(agent1.epsilonArr, label="Valores de Epsilon")
plt.title("Epsilon de DQN - entrenamiento")
plt.xlabel('juegos')
plt.ylabel('valores epsilon')
plt.legend()
plt.savefig("imagesTraining/valores_epsilon.png")
plt.close()

plt.plot(agent1.rewardAcc, 'r-', label="Rewards DQN")
plt.plot(agent2.rewardAcc, 'g', label="Rewards PPO")
plt.plot(agent3.rewardAcc, 'b', label="Rewards Random 1")
plt.plot(agent4.rewardAcc, 'k', label="Rewards Random 2")
plt.title("Recompensas por steps - entrenamiento")
plt.xlabel('steps')
plt.ylabel('rewards')
plt.legend()
plt.savefig("imagesTraining/valores_recompensas.png")
plt.close()

plt.plot(agent1.sumRewardAcc, 'r-', label="Rewards DQN")
plt.plot(agent2.sumRewardAcc, 'g-', label="Rewards PPO")
plt.plot(agent3.sumRewardAcc, 'b', label="Rewards Random 1")
plt.plot(agent4.sumRewardAcc, 'k', label="Rewards Random 2")
plt.title("Recompensas acumuladas por steps - entrenamiento")
plt.xlabel('steps')
plt.ylabel('rewards')
plt.legend()
plt.savefig("imagesTraining/valores_recompensas_acumuladas.png")
plt.close()

"""Reseteo de los valores"""
for p in playersAgents:
    p.restartRewardsValues()

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

plt.plot(agent1.rewardAcc, 'r-', label="Rewards DQN")
plt.plot(agent2.rewardAcc, 'g', label="Rewards PPO")
plt.plot(agent3.rewardAcc, 'b', label="Rewards Random 1")
plt.plot(agent4.rewardAcc, 'k', label="Rewards Random 2")
plt.title("Recompensas por steps - validación")
plt.xlabel('steps')
plt.ylabel('rewards')
plt.legend()
plt.savefig("imagesValidation/valores_recompensas.png")
plt.close()

plt.plot(agent1.sumRewardAcc, 'r-', label="Rewards DQN")
plt.plot(agent2.sumRewardAcc, 'g-', label="Rewards PPO")
plt.plot(agent3.sumRewardAcc, 'b', label="Rewards Random 1")
plt.plot(agent4.sumRewardAcc, 'k', label="Rewards Random 2")
plt.title("Recompensas acumuladas por steps - validacion")
plt.xlabel('steps')
plt.ylabel('rewards')
plt.legend()
plt.savefig("imagesValidation/valores_recompensas_acumuladas.png")
plt.close()

plt.plot(agent1.promProb, 'r', Label='Probabilidades de exito DQN')
plt.plot(agent2.promProb, 'g', Label='Probabilidades de exito PPO')
plt.title("Promedios probabilidades de éxito - validacion")
plt.xlabel('juegos')
plt.ylabel('probabilidades')
plt.legend()
plt.savefig("imagesValidation/prom_probabilidades_validacion.png")
plt.close()

plt.plot(list(range(height)), agent1.Qvalues[0], 'r', Label='Q-values DQN')
plt.plot(list(range(height)), agent2.Qvalues[0], 'g', Label='Q-values PPO')
plt.title("Q-Values finales - validacion")
plt.ylabel('Q-Values')
plt.xlabel('Acciones')
plt.legend()
plt.savefig("imagesValidation/valores_q.png")
plt.close()
