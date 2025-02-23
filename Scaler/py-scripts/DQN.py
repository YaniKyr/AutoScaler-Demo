import numpy as np
import json
from functions import Prometheufunctions 
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense  #type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import MeanSquaredError  # type: ignore
from tensorflow.keras.losses import Huber  # type: ignore
from collections import deque
import os
import tensorflow as tf
import time
losses = []
rewards = []

class DQNAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.memory = deque(maxlen=5000)
        self.rewards = []
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
        self.action = [-2, -1, 0, 1, 2]
        self.model = self._build_model()
        self.target_model = self._build_model() 
        self.update_target_model()  

    def _build_model(self):
        #print(f'At build model State size: {self.state_size}')
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(len(self.action), activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())
        return model

    def update_target_model(self):        
        self.target_model.set_weights(self.model.get_weights())
        self.model.save_weights('scaler.weights.h5')

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            cpu_z = [ (state[0] * state[2])/max(state[2]+i,1)  for i in self.action]
            maxc = 0
            midx = 0
            for idx, cpu in enumerate(cpu_z):
                if maxc <  cpu and cpu <=40:
                    maxc = cpu
                    midx = idx
            return self.action[midx]
        try:
            state = np.array([state])  # Add batch dimension
            q_values = self.model.predict(state, verbose=1)
            return self.action[np.argmax(q_values[0])]
        except Exception as e:
            print(f"\u26A0 Error fetching state:{e}")
            return np.random.choice(self.action,p=[0.35,0.4,0.25])

    def reward(self, data):
        try:
            RTT = int(round(float(data.getRTT())))
            return data.fetchState()[0] + (1 / (1 + RTT / 250) if RTT < 250 else -200)
        except Exception as e:
            print(f'\u26A0 Error {e}, Prometheus Error, during data retrieval')
            return 0

    def replay(self, batch_size):

        if len(self.memory) < batch_size:
            return

        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = np.array([self.memory[i][0] for i in minibatch])
        next_states = np.array([self.memory[i][3] for i in minibatch])
        try:        
            q_values = self.model.predict(states, verbose=0)
            q_values_next = self.target_model.predict(next_states, verbose=0)
        except Exception as e:
            print(f'\u26A0 Exception {e}, error during the target model prediction')

        for idx, i in enumerate(minibatch):
            _,action, reward, _ = self.memory[i]
            target_q = reward + self.gamma * np.amax(q_values_next[idx])
            q_values[idx][action] = target_q
        
        # Train the model
        history = self.model.fit(states, q_values, epochs=10,verbose = 0)
        loss = history.history['loss'][-1]
        losses.append(loss)
        batch_rewards = [self.memory[i][2] for i in minibatch]
        avg_reward = np.mean(batch_rewards)
        rewards.append(avg_reward)
        
        print(f"Replay - Avg Loss: {loss:.4f}, Avg Reward: {avg_reward:.4f}")
        print(f"========================After Training=============================")
        print(f"Values before decay:")
        print(f"Epsilon: {self.epsilon}, Epsilon Decay: {self.epsilon_decay}, Epsilon Min: {self.epsilon_min}")
        print(f"===================================================================")
        save_rewards_to_file(self.rewards)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



def save_rewards_to_file(rewards, filename='rewards.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(rewards, f)
        print(f"\u2705 Rewards successfully saved to {filename}")
    except Exception as e:
        print(f"\u26A0 Error saving rewards: {e}")


def Post(agent,state,step_count):
    action = agent.act(state)
    target_pods = max(1, min(Prometheufunctions().fetchState()[2] + action, 9))

    print(f'\u27A1 Step of Randomnes {step_count}, with Action={action} and State: {state}, Going to scale to: {target_pods}')
    file = '/tmp/shared_file.json'

    # Write scaling action
    with open(file, 'w') as file:
        json.dump({'action': int(target_pods)}, file)

    # Wait for Kubernetes to reach the target
    start_time = time.time()
    #keda might have a bug. When reaching max Replicas e.g. 10 and trying to scale down to 9, it fails
    #to perform the operation. In the other hand all the other scaling actions work properly
    while True:
        try:
            curr_state = Prometheufunctions().fetchState()[2] 
        except Exception as e:
            print(f'\u26A0 Error while fetching pods, encountered {e}')
            time.sleep(1)
            continue

        if curr_state == target_pods:
            return action, True


        elapsed_time = time.time() - start_time  # Calculate the elapsed time
        #Grace Period
        if elapsed_time > 45:
            print("\u26A0 Error: Timeout exceeded while waiting for pods to scale! Restarting...")
            return 0,False
            #print("Timeout waiting for pods to scale.")
        
        time.sleep(5)

def main():
    data = Prometheufunctions()
    
    try:
        state = data.fetchState()
    except Exception as e:
        print(f'\u26A0 Error {e}, Prometheus Error, during data retrieval')
        return [0,0,0]

    state_size = 3
    agent = DQNAgent(state_size)

    batch_size = 160
    replay_frequency = 160
    target_update_frequency = 100
    step_count = 0


    while 1:
        step_count += 1
        if step_count==1 and os.path.exists('Scaler.weights.h5'):
            agent.model.load_weights('Scaler.weights.h5')
        # Perform the action
        action = 0
        flag =False
        while not flag:
            action,flag = Post(agent, state, step_count)
        if action < 0:
            print("\u2705 Scaled down Saccessfuly")
        elif action > 0: 
            print("\u2705 Scaled up Saccessfuly")
        else:
            print("\u2705 Remaining in the same replica count")
        print('\U0001F504 Stabilizing for 60 secs...')
        time.sleep(60)
        try:
            print("\U0001F504 Fetching Data for the next state...")
            next_state = data.fetchState()
        except Exception as e:
            print(f'\u26A0 Error {e}, Prometheus Error, during data retrieval')
            next_state=[0,0,0]
        reward = agent.reward(data)
        print(f'\u2705 Calculated the Reward: {reward}')
        # Remember the experience
        agent.remember(state, int(action), reward, next_state)
        state = next_state
        
        # Train the agent (experience replay) 
        if len(agent.memory) >= batch_size and step_count % replay_frequency == 0:
            print("\U0001F504 Training...")
            agent.replay(batch_size)

        if step_count % target_update_frequency == 0:
            
            print("\U0001F504 Updating Values of Target...")
            agent.update_target_model()
            

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    main()
