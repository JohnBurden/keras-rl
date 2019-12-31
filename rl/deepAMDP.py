import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from rl.memory import SequentialMemory

from keras.optimizers import Adam


class deepAMDP():

    def __init__(self, alpha=1e-5, gamma=0.99, numberOfActions=0, tau=0.125):

        self.predictionModel = Sequential()
        self.predictionModel.add(Dense(64, input_dim=16))
        self.predictionModel.add(Dense(numberOfActions, activation='softmax'))
        self.predictionModel.compile(loss="mse", optimizer=Adam(lr=alpha))

        self.targetModel = Sequential()
        self.targetModel.add(Dense(64, input_dim=16))
        self.targetModel.add(Dense(numberOfActions, activation="softmax"))
        self.targetModel.compile(loss="mse", optimizer=Adam(lr=alpha))

        self.memory = SequentialMemory(limit=100000, window_length=1)

        self.numberOfActions = numberOfActions

        self.alpha=alpha
        self.gamma=gamma
        self.tau=tau

    def addExperience(self, latentState, action, reward, done):
        self.memory.append(latentState, action, reward, done)

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

        

        targetQValues = self.targetModel.predict_on_batch(state1Batch)

        qBatch = np.max(targetQValues, axis=1).flatten()
        targets = np.zeros((batchSize, self.numberOfActions))

        discountedRewardBatch = self.gamma * qBatch
        discountedRewardBatch *= terminal1Batch
        Rs = rewardBatch + discountedRewardBatch

        for (target, r, action) in zip(targets, Rs, actionBatch):
            target[action] = r
        self.predictionModel.fit(state0Batch, targets, verbose=0)
        self.updateTargetModel()

    def updateTargetModel(self):
        predictionWeights = self.predictionModel.get_weights()
        targetWeights = self.targetModel.get_weights()
        for i in range(0, len(targetWeights)):
            targetWeights[i] = predictionWeights[i]*self.tau + targetWeights[i]*(1-self.tau)
        self.targetModel.set_weights(targetWeights)