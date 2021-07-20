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

def pSuccess(Q, reward, gamma):
    n = np.log(Q / reward) / np.log(gamma)
    log10baseGamma = np.log(10) / np.log(gamma)
    probOfSuccess = (n / (2 * log10baseGamma)) + 1
    probOfSuccessLimit = np.minimum(1, np.maximum(0, probOfSuccess))
    # probOfSuccessLimit = probOfSuccessLimit * (1 - stochasticity) #Usar solo si usamos transiciones estocasticas o el parametro sigma
    return probOfSuccessLimit


"""General parameters"""
Qvalues = np.zeros([1, 200])
rew = []
epsilon = []
qvaluesAux = []
gamma = 0.9
height = 200
promQ = np.zeros([1, 200])
index = 0

"""Game parameters"""
gameType = ChefsHatEnv.GAMETYPE["MATCHES"]

gameStopCriteria = 5
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
episodes = 2

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
        action = agent1.getAction(observations)

        info = {"validAction": False}

        while not info["validAction"]:

            nextobs, reward, isMatchOver, info = env.step(action)
            #index += 1
            #print(index, reward)

        agent1.actionUpdate(observations, nextobs, action, reward, info)
        rew.append(reward)
        Qvalues[0,(np.argmax(action))] = np.max(action)
        #print(Qvalues[0])

        with open("qValues.txt", 'a+') as val_file:
            val_file.write(','.join(map(str, action)) + "\n")

        if isMatchOver: # aca es cuando un jugador se descarta la mano

            q = agent1.matchUpdate(info)
            epsilon.append(agent1.epsilon)

            '''for p in playersAgents:
                q = agent1.matchUpdate(info)
                epsilon.append(agent1.epsilon)'''

            print("-------------")
            print("Match:" + str(info["matches"]))
            print("Score:" + str(info["score"]))
            print("Performance:" + str(info["performanceScore"]))
            print("-------------")


df = pd.read_csv('qValues.txt', header=None)
df['sum'] = df.sum(axis=1)
prom = df['sum'] / 200

sumRew = sum(rew)
print(Qvalues)

prob = pSuccess(Qvalues, sumRew, gamma)
np.sort(prob)
print("probabilidad", prob)

#promQ = prom
#print(Qvalues[0])

#print(df.loc[[626]])
#print(df['sum'])
#print(df.shape[0]) # imprime el numero de filas
#print(df.shape[1]) # imprime el numero de columnas

#Plotlty Express
#fig = px.line(Qvalues[0], title='Maximos Q-Values')
#fig.show()

#fig2 = px.line(prom, title='Promedios Q-values')
#fig2.show()


plt.plot(prom, Label='Promedios Q-values')
plt.title("Q-Values Introspection entrenamiento")
plt.xlabel('Steps')
plt.ylabel('Q-Values')
plt.legend()
plt.savefig("prom_q_values.png")
plt.close()

plt.plot(list(range(height)), prob[0], '-', Label='Q-values')
plt.title("Q-Values Introspection entrenamiento")
plt.xlabel('Steps')
plt.ylabel('Q-Values')
plt.legend()
plt.savefig("q_values.png")
plt.close()

plt.plot(epsilon, label="Valores de Epsilon")
plt.title("Epsilon de Introspection entrenamiento")
plt.xlabel('Steps')
plt.ylabel('Epsilon')
plt.legend()
plt.savefig("valores_epsilon.png")
plt.close()

plt.plot(rew, label="Recompensas")
plt.title("Recompensas de Introspection entrenamiento")
plt.xlabel('Steps')
plt.ylabel('Rewards')
plt.legend()
plt.ylim([-0.5, 1.5])
plt.savefig("valores_recompensas.png")
plt.close()

"""Evaluate Agent"""
#env.startExperiment(rewardFunctions=rewards, gameType=gameType, stopCriteria=gameStopCriteria, playerNames=agentNames,
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
