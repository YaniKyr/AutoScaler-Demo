import numpy as np
import json
from functions import Prometheufunctions 
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense  #type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import MeanSquaredError  # type: ignore
from collections import deque
import os
import tensorflow as tf
import time


class DQNAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.memory = deque(maxlen=5000)
        self.rewards = []
        self.gamma = 0.9
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.75
        self.action = [-2, -1, 0, 1, 2]
        self.model = self._read_model_()
        self.target_model = self._build_model() 
        self.update_target_model()  
        self.cpu_scaler_weight = 0.9      # Start fully imitating CPU scaler
        self.cpu_scaler_decay = 0.75      # Decay factor per training cycle
        self.cpu_scaler_min = 0.01         # Minimum weight 
        self.losses = []
        self.rewards = []
    def _build_model(self):
        #print(f'At build model State size: {self.state_size}')
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(len(self.action), activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
        return model
    
    def _read_model_(self):
        try:
            model = load_model('scaler.model.h5')
            model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
            model.build()
            model.summary()
            print("\u2705 Model successfully loaded")
            return model
        except Exception as e:
            print(f"\u26A0 Error loading model: {e}")
            return self._build_model()

    def update_target_model(self):        
        self.target_model.set_weights(self.model.get_weights())
        self.model.save_weights('scaler.weights.h5')

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state, step):
        cpu_z = [(state[0] * state[2]) / max(state[2] + i, 1) for i in self.action]
        valid_indices = [idx for idx, cpu in enumerate(cpu_z) if cpu < 35]
        if valid_indices:
            cpu_scaler_action = self.action[max(valid_indices, key=lambda idx: cpu_z[idx])]
        else:
            cpu_scaler_action = 0

        # --- DQN Predicted Action ---
        try:
            state_input = np.array([state])  # Add batch dimension
            q_values = self.model.predict(state_input, verbose=0)
            dqn_action = self.action[np.argmax(q_values[0])]
        except Exception as e:
            print(f"\u26A0 Error fetching state: {e}")
            dqn_action = np.random.choice(self.action)

        # --- Blended Action Selection ---
        # With probability (cpu_scaler_weight) choose the heuristic; otherwise, the DQN decision.
        prob = np.random.rand()
        if  prob< self.epsilon and step % 2==0:
            print("🔴Randomness every 2 steps")
            chosen_action = np.random.choice(self.action)
        elif prob < self.cpu_scaler_weight:
            print("🟡Cpu Scaler In Action")
            chosen_action = cpu_scaler_action
        else:
            print("🟢 Prediction")
            chosen_action = dqn_action

        # Enforce bounds (e.g., ensure the new replica count stays between 1 and 9)
        if chosen_action + state[2] > 9 or chosen_action + state[2] < 1:
            return 0
        return chosen_action

    def reward(self, data):
        try:
            RTT = int(round(float(data.getRTT())))
            return data.fetchState()[0] + (1 / (1 + RTT / 250) if RTT < 250 else -20)
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
            q_values[idx][action] = (target_q - q_values[idx][action])**2
            norm = np.linalg.norm(q_values[idx])
            if norm == 0:
                norm = 1.0  # Avoid division by zero
            q_values[idx] = q_values[idx] / norm 
        
        # Train the model
        history = self.model.fit(states, q_values, epochs=10,verbose = 0)
        loss = history.history['loss'][-1]
        self.losses.append(loss)
        batch_rewards = [self.memory[i][2] for i in minibatch]
        avg_reward = np.mean(batch_rewards)
        self.rewards.append(avg_reward)
        
        print(f"Replay - Avg Loss: {loss:.4f}, Avg Reward: {avg_reward:.4f}")
        print(f"========================After Training=============================")
        print(f"Values before decay:")
        print(f"Epsilon: {self.epsilon}, Epsilon Decay: {self.epsilon_decay}, Epsilon Min: {self.epsilon_min}")
        print(f"===================================================================")
        save_rewards_to_file(self.rewards)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.cpu_scaler_weight = max(self.cpu_scaler_min, self.cpu_scaler_weight * self.cpu_scaler_decay)


def save_rewards_to_file(rewards, filename='rewards.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(rewards, f)
        print(f"\u2705 Rewards successfully saved to {filename}")
    except Exception as e:
        print(f"\u26A0 Error saving rewards: {e}")


def Post(agent,state,step_count):
    action = agent.act(state, step_count)
    target_pods = state[2] + action

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

    batch_size = 80
    replay_frequency = 80
    target_update_frequency = 50
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