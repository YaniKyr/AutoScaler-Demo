#TODO : Find the query bug
#TODO : Theoretical implementation of the DQN


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense  #type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from collections import deque
import random
from functions import Prometheufunctions 
import json

# Define the DQN agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(), loss=MeanSquaredError(),metrics=['accuracy','f1_score','mean_squared_error','categorical_crossentropy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def reward(self,data):
        RTT = int(round(float(data.getRTT())))
        if RTT < 250:
            return    1/(1+(RTT/250))
        else:
            return -1
            

    def replay(self, batch_size):
        #import ipdb; ipdb.set_trace()
        #minibatch = np.array(random.sample(self.memory, batch_size))
        print("Having Replay")
        for state, action, reward, next_state, done in self.memory:
            target = reward
            state,next_state = np.array(state),np.array(next_state)
            if not done:
                try :
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                except ValueError as e:
                    print(e)
           
            target_f = self.model.predict(state)
            
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=10,verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
# Create the environment

def Post(action):
    file = '/tmp/shared_file.json'
    data = {'action': action}
    with open(file, 'w') as file:
        json.dump(data, file)
    


def main():
    data = Prometheufunctions()
    action = [-2,-1,0,1,2]  # Actions to take
    state = data.fetchState()  # Number of features
    state_size  =state[['value_cpu','value_user','num_pods']].shape[1]
    action_size = len(action)  # Number of actions
    # Initialize the DQN agent
    agent = DQNAgent(state_size, action_size)

    # Training loop
    batch_size = 32
    num_episodes = 1000
    for episode in range(num_episodes):
        
        for t in range(500):
            # Choose an action
            action = agent.act(state[['value_cpu','value_user','num_pods']])
            
            # Perform the action
            #next_state, reward, done, _ = Post(action)
            Post(action)
            next_state = data.fetchState()
            reward = agent.reward(data)
            done = False

            
            # Remember the experience
            agent.remember(state[['value_cpu','value_user','num_pods']].values.tolist(), action, reward, next_state[['value_cpu','value_user','num_pods']].values.tolist(), done)

            # Update the state
            state = next_state

            # Check if episode is finished
            print("Episode: {}/{}, step: {}, action: {}, reward: {}, done: {}".format(episode, num_episodes, t, action, reward, done))
            if done:
                break

            # Train the agent
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)


if __name__ == '__main__':
    main()