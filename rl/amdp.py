
import numpy as np
import math



class AMDP():

    def __init__(self, alpha=0.1, gamma=0.99):
        self.states = []
        self.transitions = np.zeros(1, dtype=object); self.transitions[0]=[]
        self.v = np.random.random(1)
        self.alpha=alpha
        self.gamma=gamma


    def addState(self, newState):
        if newState not in self.states:
            self.states.append(newState)
            if len(self.states) == self.transitions.shape[0]:
                self.updateTransitionArray((2*len(self.states)))


    def addInitialState(self, state):
        if state not in self.states:
            self.states.append(state) 
            if len(self.states) == self.transitions.shape[0]:
                self.updateTransitionArray((2*len(self.states)))


    def addTransition(self, prevState, action, newState):
        prevStateIndex = self.states.index(prevState)
        newStateIndex = self.states.index(newState)
        actionIndex = self.actions.index(action)
        if not newStateIndex in self.transitions[prevStateIndex]:
            self.transitions[prevStateIndex].append(newStateIndex)


    def updateTransitionArray(self, newSize):
        try:
            transitionCopy = np.copy(self.transitions)
            vCopy = np.copy(self.v)
            self.transitions = np.zeros(newSize, dtype=object)
            for i in range(0, len(self.transitions)):
                self.transitions[i] = []
            self.v = np.random.random(newSize)
            ## Copy Old Values into new array
            for idx, vs in enumerate(transitionCopy):
                        self.transitions[idx] = vs
            for idx, vs in enumerate(vCopy):
                self.v[idx] = vs
        except:
            print("EXCEPTION AT SIZE:" + str(newSize))
        

    def valueUpdate(self, prevState, reward, newState, steps):
        prevStateIndex = self.states.index(prevState)
        if newState == -1: ###### Terminal / Unseen Update
            delta = reward - self.v[prevStateIndex]
            self.v[prevStateIndex] = self.v[prevStateIndex] + self.alpha*delta
        else: ######  Seen Update
            newStateIndex = self.states.index(newState)
            delta = reward + self.gamma*self.v[newStateIndex] - self.v[prevStateIndex]
            self.v[prevStateIndex] = self.v[prevStateIndex] + self.alpha*(reward + math.pow(self.gamma, steps)*self.v[newStateIndex] - self.v[prevStateIndex])


    def value(self, state):
        stateIndex = self.states.index(state)
        return self.v[stateIndex]