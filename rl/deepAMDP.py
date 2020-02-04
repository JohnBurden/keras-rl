import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from rl.memory import SequentialMemory
from keras.utils import normalize
from collections import deque
import random
from keras.optimizers import Adam


class deepAMDP():

    def __init__(self, inputDim = 16, alpha=1e-4, gamma=0.99, epsilon =0.1, numberOfActions=0, tau=1e-1):

        self.predictionModel = Sequential()
        self.predictionModel.add(Dense(16, input_dim=inputDim, activation='relu'))
        self.predictionModel.add(Dense(16, activation='relu'))
        self.predictionModel.add(Dense(numberOfActions, activation='linear'))
        self.predictionModel.compile(loss="mse", optimizer=Adam(lr=alpha))

        self.targetModel = Sequential()
        self.targetModel.add(Dense(16, input_dim=inputDim))
        self.targetModel.add(Dense(16, activation='relu'))
        self.targetModel.add(Dense(numberOfActions, activation="linear"))
        self.targetModel.compile(loss="mse", optimizer=Adam(lr=alpha))

        self.memory = SequentialMemory(limit=100000, window_length=1)


        self.otherMemory = deque(maxlen=2000)

        self.numberOfActions = numberOfActions
        self.alpha=alpha
        self.gamma=gamma
        self.tau=tau
        self.epsilon=epsilon

        self.initialEpsilon = 0.1
        self.finalEpsilon = 0.01
        self.currentEpsilon = self.initialEpsilon
        self.episodesToDecay = 500

    def addExperience(self, latentState, action, reward, done):
        self.memory.append(latentState, action, reward, done)
        #print(latentState, action, reward)


    def memorize(self, state, action, reward, next_state, done):
        self.otherMemory.append((state, action, reward, next_state, done))

    def action(self, state):
        if np.random.random() < self.currentEpsilon:
            return np.random.randint(self.numberOfActions)
        state = state.reshape(1,-1)
        qValues = self.predictionModel.predict(state)
        return np.argmax(self.predictionModel.predict(state)[0])


    def replay(self, batchSize=8):
        #print("replay")
        #if len(self.memory) < batchSize: 
        #    return

        experiences = self.memory.sample(batchSize)
       

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0Batch = []
        rewardBatch = []
        actionBatch = []
        terminal1Batch = []
        state1Batch = []
        for e in experiences:
           # print(e.state0, e.state1, e.reward, e.action)
            state0Batch.append(e.state0[0])
            state1Batch.append(e.state1[0])
            rewardBatch.append(e.reward)
            actionBatch.append(e.action)
            terminal1Batch.append(0. if e.terminal1 else 1.)

        state0Batch = np.array(state0Batch)
        rewardBatch = np.array(rewardBatch)
        actionBatch = np.array(actionBatch)
        terminal1Batch = np.array(terminal1Batch)
        state1Batch = np.array(state1Batch)

        #state0Batch = normalize(state0Batch, axis=-1)
        #state1Batch = normalize(state1Batch, axis=-1)

        targetQValues = self.targetModel.predict_on_batch(state1Batch)

        #print("Target Q Values")
        #print(targetQValues)
        #print("Target Q 1")
        #print(state1Batch[0])
        qBatch = np.max(targetQValues, axis=1).flatten()
        #targets = np.zeros((batchSize, self.numberOfActions))
        #targets = np.random.rand(batchSize, self.numberOfActions)
        targets = targetQValues

        discountedRewardBatch = self.gamma * qBatch
        discountedRewardBatch *= terminal1Batch
        Rs = rewardBatch + discountedRewardBatch

        #print("Our Targets")
        #print(targets)
        for (target, r, action) in zip(targets, Rs, actionBatch):
            target[action] = r
        #print("Updated Targets")
        #print(targets)


        self.predictionModel.fit(state0Batch, targets, verbose=0)
        #self.updateTargetModel()



    def otherReplay(self, batch_size=8):
        minibatch = random.sample(self.otherMemory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            state = state.reshape(1,-1)
            next_state = next_state.reshape(1,-1)
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.predictionModel.predict(next_state)[0]))
            target_f = self.predictionModel.predict(state)
            target_f[0][action] = target
            self.predictionModel.fit(state, target_f, epochs=1, verbose=0)
        #self.updateTargetModel()

    def updateTargetModel(self):
        predictionWeights = self.predictionModel.get_weights()
        targetWeights = self.targetModel.get_weights()
        
        for i in range(0, len(targetWeights)):
            targetWeights[i] = predictionWeights[i]*self.tau + targetWeights[i]*(1-self.tau)
        self.targetModel.set_weights(targetWeights)
