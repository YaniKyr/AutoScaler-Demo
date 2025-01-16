import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from locust import HttpUser, task, between, events
import gevent

# Parameters
n_users = 100  # Max users
n_timesteps = 2500 # Number of time steps in the dataset
timestep_min = 1  # Minimum timestep in seconds
timestep_max = 10  # Maximum timestep in seconds
start_end_period_seconds = 20 * 60  # 10 minutes in seconds
start_end_user_range = (20, 30)  # User count during start/end period

# Generate time series data with a dome-shaped pattern and bursts
timesteps = [random.randint(timestep_min, timestep_max) for _ in range(n_timesteps)]
timestamps = np.cumsum(timesteps)

# Dome-shaped activity
activity = np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, n_timesteps))
activity = (activity - activity.min()) / (activity.max() - activity.min())
activity = (activity * (n_users-1 )) + 10
noise = np.random.normal(0, 0.5, n_timesteps)
activity = activity + noise
activity = np.clip(activity, 1, n_users).round().astype(int)

# DataFrame for time series
data = pd.DataFrame({
    'Timestamp': timestamps,
    'Activity': activity
})

# Helper to simulate bursts at start and end
def generate_start_end_data(start_time, duration_seconds, user_range):
    additional_timesteps = []
    while sum(additional_timesteps) < duration_seconds:
        additional_timesteps.append(random.randint(timestep_min, timestep_max))
    additional_timestamps = np.cumsum(additional_timesteps) + start_time
    additional_activity = [random.randint(*user_range) for _ in range(len(additional_timestamps))]
    return pd.DataFrame({'Timestamp': additional_timestamps, 'Activity': additional_activity})

# Generate data for start and end bursts
start_data = generate_start_end_data(0, start_end_period_seconds, start_end_user_range)
data['Timestamp'] += start_data['Timestamp'].iloc[-1]
end_data = generate_start_end_data(data['Timestamp'].iloc[-1], start_end_period_seconds, start_end_user_range)
data = pd.concat([start_data, data, end_data], ignore_index=True)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['Timestamp'], data['Activity'], label="User Activity", color='blue')
plt.title('User Activity Over Time with Dome-shaped Pattern')
plt.xlabel('Timestamp (seconds)')
plt.ylabel('Number of Active Users')
plt.grid(True)
plt.savefig('../../Data/Input_data.png')
data['wait_time'] = data['Timestamp'].diff().fillna(0)
data.to_csv('../../Data/op_data.csv',index=False)

# Locust setup
class UserBehavior(HttpUser):
    wait_time = between(0.5,1)  # Time between requests
    host = "http://localhost:8080"  # Set the base host

    @task
    def my_task(self):
        self.client.get("/productpage") 

def adjust_user_count(environment):
    for _, row in data.iterrows():
        current_user_count = row['Activity']
        # Adjust the number of users
        environment.runner.start(current_user_count, spawn_rate=10)
        gevent.sleep(row['wait_time'])

# Start Locust with periodic user count adjustment
@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    gevent.spawn(adjust_user_count, environment)
