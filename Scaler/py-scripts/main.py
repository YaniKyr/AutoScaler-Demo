from stable_baselines3 import DQN, PPO, A2C
from env import KubernetesEnv
import numpy as np
import os


def train_model(env, model_type='DQN', total_timesteps=1000, episode=10):
    for _ in range(episode):
        try:
            model = load_model(env, model_type=model_type)
            
            model.learn(total_timesteps=total_timesteps)
            print("✅ Training Completed\n")
        except Exception as e:
            print(f'⚠ Error {e}, during training')
            break

        
        try:
            model.save(f"{model_type}_model") 
            print("✅ Model Saved\n")
        except Exception as e:
            print(f'⚠ Error {e}, during model saving')
            break

def load_model(env, model_type='DQN'):
    
    if os.path.exists(f"{model_type}_model.zip"):
        match model_type:
            case 'DQN':
                model = DQN.load(f"{model_type}_model.zip")
            case 'PPO':
                model = PPO.load(f"{model_type}_model.zip")
            case 'A2C':
                model = A2C.load(f"{model_type}_model.zip")
            case _:
                print(f"⚠ Unsupported model type: {model_type}")
        print(f"✅ Model Loaded: {model_type}\n")
    else:
        match model_type:
            case 'DQN':
                print(f"⚠ Model file for {model_type} not found. Please train the model first.")
                model = DQN("MlpPolicy", env, verbose=2 )
            case 'PPO':
                print(f"⚠ Model file for {model_type} not found. Please train the model first.")
                model = PPO("MlpPolicy", env, verbose=2 )
            case 'A2C':
                print(f"⚠ Model file for {model_type} not found. Please train the model first.")
                model = A2C("MlpPolicy", env, verbose=2 )
            
    return model

def main():
    env = KubernetesEnv()
    print("✅ Environment Created\n")
    
    for ele in ['DQN']:
        train_model(env, model_type=ele, total_timesteps=1, episode=1)
    
 
    episode = 60
    print("✅ Environment Reset\n")
    print("✅ Starting Prediction...\n")

    # Buffer to store prediction results
    prediction_buffer = []

    while True:
        state, _ = env.reset()
        model = load_model(env, model_type='DQN')
        
        for i in range(episode):
            try:
                action, _ = model.predict(state)
            except Exception as e:
                print(f'⚠ Error {e}, during prediction')
                break
            print(action)
            action = int(np.argmax(action[0]))
            state, Reward, _, _, meanReward = env.step(action)

            # Save results to buffer
            prediction_buffer.append({
                "Episode": i + 1,
                "Model": ele,
                "Action": action,
                "State": state,
                "Reward": Reward,
                "Mean Reward": meanReward
            })

            print(f"Episode: {i+1}, Action: {action}, State: {state}, Reward: {Reward}, Mean Reward: {meanReward}")
    
    # Optionally, save the buffer to a file
    with open("prediction_results.json", "w") as f:
        import json
        json.dump(prediction_buffer, f, indent=4)
    print("✅ Prediction results saved to prediction_results.json")

if __name__ == '__main__':
    main()
