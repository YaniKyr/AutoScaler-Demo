#TODO : Fetch vectors of data, not just scalars. To fill the Replay buffer faster with more data
#TODO: Fill Replay Buffer and reduce Train period
import numpy as np
import json
from functions import Prometheufunctions 
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense, Dropout  #type: ignore
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
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.action = [-2, -1, 0, 1, 2]
        self.model = self._read_model_()
        self.target_model = self._build_model() 
        self.update_target_model()  
        self.losses = []
        self.rewards = []

    def _build_model(self):
        #print(f'At build model State size: {self.state_size}')
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(len(self.action), activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError())
        return model
    
    def _read_model_(self):

        return self._build_model()

    def update_target_model(self):        
        self.target_model.set_weights(self.model.get_weights())
        self.model.save_weights('scaler.weights.h5')

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state, step):

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
        
        if  prob< self.epsilon :
            print("🔴Randomness In Action")
            chosen_action = np.random.choice(self.action)
        else:
            print("🟢 Prediction")
            chosen_action = dqn_action

        # Enforce bounds (e.g., ensure the new replica count stays between 1 and 9)
        if chosen_action + state[2] > 9 or chosen_action + state[2] < 1:
            return 0
        return chosen_action

    def reward(self,data, RTT=0, flooded = False):
        try:
            if not flooded:
                RTT = int(round(float(Prometheufunctions().getRTT())))

            return data[0] + (1 / (1 + RTT / 250) if RTT < 250 else -2)
        except Exception as e:
            print(f'\u26A0 Error {e}, Prometheus Error, during data retrieval')
            return 0

    def replay(self, batch_size):

        if len(self.memory) < batch_size:
            return

        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        sdata = np.array([self.memory[i][0] for i in minibatch])
        min_vals = np.min(sdata, axis=0)
        max_vals = np.max(sdata, axis=0)
        states = (sdata - min_vals) / (max_vals - min_vals)
        
        nst = np.array([self.memory[i][3] for i in minibatch])
        min_vals = np.min(nst, axis=0)
        max_vals = np.max(nst, axis=0)
        next_states = (nst - min_vals) / (max_vals - min_vals)
        
        try:        
            q_values = self.model.predict(states, verbose=0)
            q_values_next = self.target_model.predict(next_states, verbose=0)
        except Exception as e:
            print(f'\u26A0 Exception {e}, error during the target model prediction')

        for idx, i in enumerate(minibatch):
            _,action, reward, _ = self.memory[i]
            q_values[idx][action] = reward + self.gamma * np.amax(q_values_next[idx])
            

        
        # Train the model
        history = self.model.fit(states, q_values, epochs=1,verbose = 0)
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

    def floodData(self,Prometheus):
        cpu, reqs, pods, response_t = Prometheus.floodReplayBuffer(20)
        cpu = np.nan_to_num(cpu)
        reqs = np.nan_to_num(reqs)
        pods = np.nan_to_num(pods)
        response_t = np.nan_to_num(response_t)

        for idx in range(len(cpu) - 1):
            if cpu[idx] == 0 and reqs[idx] == 0 and pods[idx] == 0:
                continue
            state = [cpu[idx], reqs[idx], pods[idx]]
            
            next_state = [cpu[idx + 1], reqs[idx + 1], pods[idx + 1]]
            reward = self.reward(next_state,response_t[idx+1],True)
            action_step = pods[idx+1] - pods[idx]
            
            self.remember(state, action_step, reward, next_state)
            
            #print(f'\u2705 Flooded data for step {idx+1} with action {action_step} and reward {reward}')



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

def futureCheck(): 
    #Counter for the number of maximum pods --Done
    #System Stability -- Can't define
    #Maybe an SLA higher than 2000 or persisting in high levels  --Done
    #set a time threshold per episode -- Done
    if Prometheufunctions().getSlaVioRange():
        return Prometheufunctions().fetchState(), True
    elif Prometheufunctions().getMaxPodsRange():
        return Prometheufunctions().fetchState(), True
    return Prometheufunctions().fetchState(), False
    

def main():
    data = Prometheufunctions()
    
   

    state_size = 3
    agent = DQNAgent(state_size)
    episodes = 1000
    batch_size = 64
    replay_frequency = 2
    target_update_frequency = 100
    step_count = 0
    
    for i in range(episodes):

        done = False
        print('\n\n')
        print(f'\u27A1 Episode {i+1}/{episodes}')
        
        while not done:
            try:
                agent.floodData(data)
                print("\U0001F504 Flooding Data...")
            except Exception as e:
                print(f'\u26A0 Error {e}, Prometheus Error, during data retrieval')
                return [0,0,0]
            
            step_count += 1
            #if step_count==1 and os.path.exists('Scaler.weights.h5'):
            #    agent.model.load_weights('Scaler.weights.h5')
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
            print('\U0001F504 Stabilizing for 40 secs...')
            time.sleep(40)
            try:
                print("\U0001F504 Fetching Data for the next state...")
                next_state, _ = futureCheck()
            except Exception as e:
                print(f'\u26A0 Error {e}, Prometheus Error, during data retrieval')
                next_state=[0,0,0]
            reward = agent.reward(next_state,0 ,flooded=False)
            print(f'\u2705 Calculated the Reward: {reward}')
            # Remember the experience
            agent.remember(state, int(action), reward, next_state)
            state = next_state
            
            # Train the agent (experience replay) 
            if len(agent.memory) >= batch_size and step_count % replay_frequency == 0:
                print("\U0001F504 Training...")
                agent.replay(batch_size)
                done = True

            if step_count % target_update_frequency == 0:
                
                print("\U0001F504 Updating Values of Target...")
                agent.update_target_model()
            

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    main()
