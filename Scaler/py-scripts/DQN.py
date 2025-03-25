import numpy as np
import json
from functions import Prometheufunctions 
import os
import tensorflow as tf
import time
from collections import deque
import math


class A2CAgent:
    def __init__(self, state_size):
        self.state_size = state_size
        
        self.rewards = []
        self.gamma = 0.9
        self.action = [-2, -1, 0, 1, 2]
        self.losses = []
        self.rewards = []
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(self.action), activation='softmax')])

        self.critic = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)])
        
        self.actor_optimizer =tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    def reward(self,data):
        try:
            RTT = int(round(float(Prometheufunctions().getRTT())))
            return data[0] + (1 / (1 + RTT / 250) if RTT < 250 else -2)
        except Exception as e:
            print(f'\u26A0 Error {e}, Prometheus Error, during data retrieval')
            return 0


def save_rewards_to_file(rewards, filename='rewards.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(rewards, f)
        print(f"\u2705 Rewards successfully saved to {filename}")
    except Exception as e:
        print(f"\u26A0 Error saving rewards: {e}")

def save_losses_to_file(losses, filename='losses.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(losses, f)
        print(f"\u2705 Losses successfully saved to {filename}")
    except Exception as e:
        print(f"\u26A0 Error saving losses: {e}")


def Post(agent,state,step_count):
    #keda might have a bug. When reaching max Replicas e.g. 10 and trying to scale down to 9, it fails
    #to perform the operation. In the other hand all the other scaling actions work properly

    action_probs = agent.actor(np.array([state]))
    action = np.random.choice(agent.action, p=action_probs.numpy()[0])
    print(f'\n ->This Message is Only for control: action_probs: {action_probs.numpy()}\n')

    if action + state[2] > 9 or action + state[2] < 1:
        action =0
    target_pods = state[2] + action
    
    print(f'\u27A1 Step of Randomnes {step_count}, with Action={action} and State: {state}, Going to scale to: {target_pods}')
    file = '/tmp/shared_file.json'

    # Write scaling action
    with open(file, 'w') as file:
        json.dump({'action': int(target_pods)}, file)


    start_time = time.time()

    ####################################
    ########## Wait for Scaling ########
    ####################################
    while True:
        try:
            curr_state = Prometheufunctions().fetchState()[2] 
        except Exception as e:
            print(f'\u26A0 Error while fetching pods, encountered {e}')
            time.sleep(1)
            continue

        if curr_state == target_pods:
            return action, action_probs, True

        elapsed_time = time.time() - start_time  # Calculate the elapsed time
        #Grace Period
        if elapsed_time > 45:
            print("\u26A0 Error: Timeout exceeded while waiting for pods to scale! Restarting...")
            return 0,0,False
            #print("Timeout waiting for pods to scale.")
        
        time.sleep(5)
    ####################################
    ####################################
    ####################################
    
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
    n = 3 #keep n very small
    min_loss = math.inf

    maxlen = n
    states = deque(maxlen=maxlen)
    actions = deque(maxlen=maxlen)
    rewards = deque(maxlen=maxlen)
    next_states = deque(maxlen=maxlen)
    dones = deque(maxlen=maxlen)
    try:
        state = data.fetchState()
    except Exception as e:
        print(f'\u26A0 Error {e}, Prometheus Error, during data retrieval')
        return [0,0,0]

    state_size = 3
    a2c = A2CAgent(state_size)
    episodes = 1000
    step_count = 0
    episode_reward =0
    for i in range(episodes):
        done = False
        print('\n\n')
        print(f'\u27A1 Episode {i+1}/{episodes}')
        
        for _ in range(3):
                
            step_count += 1
        
            action = 0
            flag =False

            while not flag:
                action, action_probs ,flag = Post(a2c, state, step_count)


            ############################################################
            ################ Stabilization And Prints ##################
            ############################################################

            if action < 0:
                print("\u2705 Scaled down Saccessfuly")
            elif action > 0: 
                print("\u2705 Scaled up Saccessfuly")
            else:
                print("\u2705 Remaining in the same replica count")

            print('\U0001F504 Stabilizing for 45 secs...')
            time.sleep(45)

            try:
                print("\U0001F504 Fetching Data for the next state...")
                next_state, _ = futureCheck()
            except Exception as e:
                print(f'\u26A0 Error {e}, Prometheus Error, during data retrieval')
                next_state=[0,0,0]

            ############################################################
            ############################################################
            ############################################################
            reward = a2c.reward(next_state)
            print(f'\u2705 Calculated the Reward: {reward}')

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

                ######### A2C #########
                
            state = next_state
                
                
            
        try:
            with tf.GradientTape(persistent = True) as tape:
                try:
                    state_values = a2c.critic(np.array(states))
                    next_state_values = a2c.critic(np.array(next_states))
                    advantages = np.array(rewards) + a2c.gamma * np.array(next_state_values) - np.array(state_values)
                except Exception as e:
                    print("\u26A0 Error during critic calculation: ", e)
                actor_losses = []
                critic_losses = []
                try:
                    for idx, (advantage, action_prob, action) in enumerate(zip(advantages, action_probs, actions)):

                        actor_loss = tf.math.log(action_prob[0, action]) * advantage
                        critic_loss = tf.square(advantage)
                        actor_losses.append(actor_loss)
                        critic_losses.append(critic_loss)
                        episode_reward += rewards[idx]
                except Exception as e:
                    print("\u26A0 Error during loss calculation: ", e)

                total_actor_loss = tf.reduce_mean(actor_losses)
                total_critic_loss = tf.reduce_mean(critic_losses)
                try:
                    # Update actor and critic
                    actor_gradients = tape.gradient(total_actor_loss, a2c.actor.trainable_variables)
                    critic_gradients = tape.gradient(total_critic_loss, a2c.critic.trainable_variables)
                    a2c.actor_optimizer.apply_gradients(zip(actor_gradients, a2c.actor.trainable_variables))
                    a2c.critic_optimizer.apply_gradients(zip(critic_gradients, a2c.critic.trainable_variables))
                except Exception as e:
                    print("\u26A0 Error during gradient application: ", e)
        except Exception as e:
                print(f'\u26A0 Error during training step: {e}')
        print(f'\u2705 Actor Loss: {actor_loss.numpy()}, Critic Loss: {critic_loss.numpy()}')

        # Save rewards after each episode
        a2c.rewards.append(episode_reward)
        save_rewards_to_file(a2c.rewards)
        a2c.losses.append((actor_loss.numpy(), critic_loss.numpy()))
        save_losses_to_file(a2c.losses)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    main()
