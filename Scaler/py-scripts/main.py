from stable_baselines3 import DQN, PPO, A2C
from env import KubernetesEnv
import numpy as np
import os
import pandas as pd
import time


def load_Dataset():
    if os.path.exists("dataset.csv"):
        print("✅ Dataset Loaded\n")
        return pd.read_csv("dataset.csv"), True
    else:
        print("⚠ Dataset not found. Please create a dataset first.")
        return None, False

def save_replay_buffer_to_csv(model, filename="replay_buffer.csv"):
    try:
        if hasattr(model, "replay_buffer") and model.replay_buffer is not None:
            data = []
            for i in range(len(model.replay_buffer)):
                obs, action, reward, next_obs, done = model.replay_buffer.sample(1)
                data.append({
                    "obs": obs.flatten().tolist(),
                    "action": action.flatten().tolist(),
                    "reward": reward.flatten().tolist(),
                    "next_obs": next_obs.flatten().tolist(),
                    "done": done.flatten().tolist()
                })
            df = pd.DataFrame(data) 
            df.to_csv(filename, index=False)
            print(f"✅ Replay buffer saved to {filename}\n")
        else:
            print("⚠ Model does not have a replay buffer to save.\n")
    except Exception as e:
        print(f"⚠ Error {e}, during replay buffer saving\n")

def offline_training(env, model_type='DQN', total_timesteps=200, episode=50):
    dataset , _= load_Dataset()
    for _ in range(episode):
        
        model = load_model(env, model_type=model_type, verbose=2 )
        print("✅ Model Loaded\n")
        time.sleep(10)

        
        for _, row in dataset.iterrows():
            obs = row['obs']
            action = row['action']
            reward = row['reward']
            next_obs = row['next_obs']
            try:
                model.replay_buffer.add(obs, next_obs, action, reward, done=False)
            except Exception as e:
                print(f'⚠ Error {e}, during replay buffer addition')
                break
        
        try:
            model.learn(total_timesteps=total_timesteps)
        except Exception as e:
            print(f'⚠ Error {e}, during model training')
            break
        
        try:
            model.save(f"{model_type}_model") 
            print("✅ Model Saved\n")
        except Exception as e:
            print(f'⚠ Error {e}, during model saving')
            break


def online_training(env, model_type='DQN', total_timesteps=1000, episode=10):
    for _ in range(episode):
        try:
            model = load_model(env, model_type=model_type)
            print("✅ Model Loaded\n")
            time.sleep(10)

            model.learn(total_timesteps=total_timesteps)
            
        except Exception as e:
            print(f'⚠ Error {e}, during training')
            break

        
        try:
            model.save(f"{model_type}_model") 
            print("✅ Model Saved\n")
        except Exception as e:
            print(f'⚠ Error {e}, during model saving')
            break

    save_replay_buffer_to_csv(model, filename="replay_buffer.csv")

def load_model(env, model_type='DQN', verbose=2):
    
    match model_type:
        case 'DQN':
            if os.path.exists(f"{model_type}_model.zip"):
                model = DQN.load(f"{model_type}_model.zip")
            else:
                print(f"⚠ Model file for {model_type} not found. Please train the model first.")
                model = DQN("MlpPolicy", env, verbose=verbose ) 
        case 'PPO':
            if os.path.exists(f"{model_type}_model.zip"):
                model = PPO.load(f"{model_type}_model.zip")
            else:
                print(f"⚠ Model file for {model_type} not found. Please train the model first.")
                model = PPO("MlpPolicy", env, verbose=verbose)
        case 'A2C':
            if os.path.exists(f"{model_type}_model.zip"):
                model = A2C.load(f"{model_type}_model.zip")
            else:
                print(f"⚠ Model file for {model_type} not found. Please train the model first.")
                model = A2C("MlpPolicy", env, verbose=verbose )
        case _:
            print(f"⚠ Unsupported model type: {model_type}")

    print(f"✅ Model Loaded: {model_type}\n")
    model.set_env(env)
    return model

def main():
    env = KubernetesEnv()
    print("✅ Environment Created\n")
    env.reset()
    print("✅ Environment Reset\n")

    if load_Dataset()[1]:
        print("✅ Starting Offline Training...\n")
        for ele in ['DQN', 'PPO', 'A2C']:
            offline_training(env, model_type=ele, total_timesteps=1000, episode=10)
    else:
        for ele in ['DQN']:
            online_training(env, model_type=ele, total_timesteps=200, episode=50)

if __name__ == '__main__':
    main()
