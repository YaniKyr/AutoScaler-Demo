import numpy as np
from functions import Prometheufunctions 
import tensorflow as tf



class A2CAgent:
    def __init__(self, state_size):
        

        self.rewards = []
        self.gamma = 0.9
        self.action = [-2, -1, 0, 1, 2]
        self.losses = []
        self.rewards = []
        self.state_size = state_size
        self.action_size = len(self.action)
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(self.action), activation='softmax')])

        self.critic = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)])
        
        self.actor_optimizer =tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def build_linear_actor_critic(self):
        inputs = tf.keras.Input(shape=(self.state_size,))
        actor = tf.keras.layers.Dense(self.action, activation="softmax")(inputs)
        critic = tf.keras.layers.Dense(1)(inputs)
        return tf.keras.Model(inputs, [actor, critic])

    def build_shallow_mlp_actor_critic(self):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(16, activation="relu")(inputs)
        actor = tf.keras.layers.Dense(self.action_dim, activation="softmax")(x)
        critic = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs, [actor, critic])

    def build_deep_mlp_actor_critic(self):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(32, activation="relu")(inputs)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        actor = tf.keras.layers.Dense(self.action_dim, activation="softmax")(x)
        critic = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs, [actor, critic])
    
    def select_best_actor(actors, state):
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        advantages = []

        for actor in actors:
            _, value = actor(state)
            advantages.append(value.numpy().squeeze())  # Use predicted value for selection

        best_actor_idx = np.argmax(advantages)
        return actors[best_actor_idx]

    def reward(self,data):
        try:
            RTT = int(round(float(Prometheufunctions().getRTT())))
            return data[0] + (1 / (1 + RTT / 250) if RTT < 250 else -2)
        except Exception as e:
            print(f'\u26A0 Error {e}, Prometheus Error, during data retrieval')
            return 0


