import numpy as np
import json
from functions import Prometheufunctions 
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense  #type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from collections import deque
import os
import tensorflow as tf
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


# Define the DQN agent class
class DQNAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.action = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(len(self.action), activation='linear'))
        model.compile(optimizer=Adam(), loss=MeanSquaredError())
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action)
        state = np.array([state])  # Add batch dimension
        q_values = self.model.predict(state, verbose=0)
        return self.action[np.argmax(q_values[0])]

    def reward(self, data):
        RTT = int(round(float(data.getRTT())))
        return 1 / (1 + RTT / 250) if RTT < 250 else -0.5

    def replay(self, batch_size):
        minibatch = np.random.choice(self.memory, batch_size, replace=False)
        for state, action, reward, next_state in minibatch:
            state, next_state = np.array([state]), np.array([next_state])
            target_q = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target = self.model.predict(state, verbose=0)
            target[0][self.action.index(action)] = target_q
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
# Create the environment

def Post(action):
    cpods = Prometheufunctions().fetchState()[2]
    target_pods = max(1, 2)

    print(f"Applying action: {action}, Target pods: {target_pods}")
    file = '/tmp/shared_file.json'

    # Write scaling action
    with open(file, 'w') as file:
        json.dump({'action': int(target_pods)}, file)

    # Wait for Kubernetes to reach the target
    start_time = time.time()
    while Prometheufunctions().fetchState()[2] != target_pods:
        if time.time() - start_time > 60:  # Timeout after 60 seconds
            print("Timeout waiting for pods to scale.")
            break
        time.sleep(1)



def main():
    data = Prometheufunctions()
      # Actions to take
    state = data.fetchState()  # Number of features
    state_size  = 3
  
    # Initialize the DQN agent
    agent = DQNAgent(state_size)

    # Training loop
    batch_size = 32
    num_episodes = 100
    for episode in range(num_episodes):
        
        for t in range(500):
            # Choose an action
            action = agent.act(state)
            
            # Perform the action
            #next_state, reward, done, _ = Post(action)
            Post(action)
            next_state = data.fetchState()
            reward = agent.reward(data)
            
            # Remember the experience
            agent.remember(state, action, reward, next_state)

            # Update the state
            state = next_state
         

            # Train the agent
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)


if __name__ == '__main__':
    main()