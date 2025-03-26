import numpy as np
import json
from functions import Prometheufunctions 
import os
import tensorflow as tf
import time
from collections import deque
import Model as md

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


def Post(a2c, models,states,step_count):
    #keda might have a bug. When reaching max Replicas e.g. 10 and trying to scale down to 9, it fails
    #to perform the operation. In the other hand all the other scaling actions work properly

    best_actor = a2c.select_best_actor(models, states)
    state_tensor = np.expand_dims(states, axis=0)
    action_probs, value = best_actor(state_tensor)

    action = np.random.choice(a2c.action_size, p=action_probs.numpy().squeeze())

    print(f'\n ->This Message is Only for control: action_probs: {action_probs.numpy()}\n')

    if action + states[-1][2] > 9 or action + states[-1][2] < 1:
        action =0
    target_pods = states[-1][2] + action
    
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
            return action, action_probs, value, True

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
    
def futureCheck(single=False): 
    #Counter for the number of maximum pods --Done
    #System Stability -- Can't define
    #Maybe an SLA higher than 2000 or persisting in high levels  --Done
    #set a time threshold per episode -- Done
    if single:
        return Prometheufunctions().fetchState()
    return Prometheufunctions().floodReplayBuffer(30)
    

def main():
    episodes = 1000
    action =0
    trial_period = 10

    try:
        state = Prometheufunctions().fetchState()
    except Exception as e:
        print(f'\u26A0 Error {e}, Prometheus Error, during data retrieval')
        return [0,0,0]
    
    
    model = md.A2CAgent(len(state))
    n = 3 #keep n very small
    actors = [
        model.build_linear_actor_critic(model.state_size, model.action_size),
        model.build_shallow_mlp_actor_critic(model.state_size, model.action_size),
        model.build_deep_mlp_actor_critic(model.state_size, model.action_size)
    ]
    
    optimizers = [tf.keras.optimizers.Adam(0.001) for _ in actors]

    for i in range(episodes):
        print('\n\n')
        print(f'\u27A1 Episode {i+1}/{episodes}')
        step_count = 0
        logs = []
        values = []
        rewards = []
        for _ in range(trial_period):
            
            step_count += 1

            flag =False
            while not flag:
                action, action_probs, value, flag = Post(model, state, step_count)


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
                next_state = futureCheck(True)
            except Exception as e:
                print(f'\u26A0 Error {e}, Prometheus Error, during data retrieval')
                next_state=[0,0,0]

            ############################################################
            ############################################################
            ############################################################
            reward = model.reward(next_state)
            print(f'\u2705 Calculated the Reward: {reward}')
            log_prob = tf.math.log(action_probs[0, action])

            ######### A2C #########
            logs.append(log_prob)
            values.append(value)
            rewards.append(reward)

                
                
            
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + model.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns, dtype=np.float32)
        values = tf.concat(values, axis=0)
        log_probs = tf.stack(log_probs)

        # Compute advantage
        advantages = returns - values.numpy().squeeze()
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        # Compute loss
        actor_loss = -tf.reduce_mean(log_probs * advantages)
        critic_loss = tf.reduce_mean(tf.square(advantages))
        loss = actor_loss + critic_loss

        # Update all models
        for i, optimizer in enumerate(optimizers):
            with tf.GradientTape() as tape:
                actor_probs, value_preds = actors[i](np.expand_dims(state, axis=0))
                value_preds = tf.squeeze(value_preds)
                policy_loss = -tf.reduce_mean(tf.math.log(actor_probs[0, action]) * advantages)
                value_loss = tf.reduce_mean(tf.square(advantages))
                total_loss = policy_loss + value_loss
            
            grads = tape.gradient(total_loss, actors[i].trainable_variables)
            optimizer.apply_gradients(zip(grads, actors[i].trainable_variables))
        print(f'\u2705 Actor Loss: {actor_loss.numpy()}, Critic Loss: {critic_loss.numpy()}')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    main()
