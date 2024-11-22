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
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.action = [-4,-3,-2,-1,0,1,2, 3, 4]
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(len(self.action), activation='linear'))
        model.compile(optimizer=Adam(), loss=MeanSquaredError(),metrics=['accuracy','f1_score','mean_squared_error','categorical_crossentropy'])
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def reward(self,data):
        RTT = int(round(float(data.getRTT())))
        if RTT < 250:
            return    1/(1+(RTT/250))
        else:
            return -1
            

    def replay(self):

        print("Having Replay")
        for state, action, reward, next_state in self.memory:
            state,next_state = np.array(state),np.array(next_state)
            y = reward + self.gamma * np.amax(self.model.predict(next_state)[0] )

            y_ = self.model.predict(state)[0]
            print("Y: ",y)
            print("Y_: ",y_)
            target = (y - y_)**2
            
            self.model.fit(state, target, epochs=10,verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
# Create the environment

def Post(action):
    cpods = Prometheufunctions().fetchState()[2]
    
    file = '/tmp/shared_file.json'
    print("Posting action: ", action, "Current pods", cpods)
    if action + cpods <1 :
        data = {'action': 1}
    else:
        data = {'action': int(action+ cpods)}

    with open(file, 'w') as file:
        json.dump(data, file)
    while Prometheufunctions().fetchState()[2] != data['action']:
        print("Waiting for action to take effect")
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
                agent.replay()


if __name__ == '__main__':
    main()