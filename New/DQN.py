import numpy as np
import padas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from prometheus_api_client import PrometheusConnect
from collections import deque
import requests

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
        model.compile(optimizer=Adam(), loss=MeanSquaredError())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def state():
        numpods = 'sum(kube_pod_info)'
        userRequests = 'sum(kube_pod_container_resource_requests_cpu_cores)'
        cpuUtil = 'sum(rate(container_cpu_usage_seconds_total{pod=~"php.*"}[1m])*100) by (pod)[10m:]'
        prom = PrometheusConnect(url="http://10.152.183.236:9090", disable_ssl=True)
        pods = prom.custom_query(query=numpods)
        requests = prom.custom_query(query=userRequests)
        cpu = prom.custom_query(query=cpuUtil)


        #TODO - Create deparsh 
        df = pd.DataFrame.from_dict(data[0]['values'])
        df = df.rename(columns={0: 'timestamp', 1: 'value'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['value'] = df['value'].astype(float)

        response_data = {"timestamp":"0/0/0","value":df['value'].iloc[-1]}
        print(response_data)
        return (pods, requests, cpu)


    def liveness():
        try:
            prom = PrometheusConnect(url="http://10.152.183.236:9090", disable_ssl=True)
            liveness_query = 'up{job="prometheus"}'
            liveness_data = prom.custom_query(query=liveness_query)
            print("Prometheus is live")
        except requests.exceptions.ConnectionError as e:
        
            print(e, "Prometheus is not live")
            return
    

    def replay(self, batch_size):
        minibatch = np.array(random.sample(self.memory, batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
# Create the environment

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize the DQN agent
agent = DQNAgent(state_size, action_size)

# Training loop
batch_size = 32
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for t in range(500):
        # Render the environment (optional)
        env.render()

        # Choose an action
        action = agent.act(state)

        # Perform the action
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # Remember the experience
        agent.remember(state, action, reward, next_state, done)

        # Update the state
        state = next_state

        # Check if episode is finished
        if done:
            break

        # Train the agent
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)