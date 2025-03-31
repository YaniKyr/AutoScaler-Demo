from stable_baselines3 import DQN, PPO, A2C
from env import KubernetesEnv
import numpy as np

def train_model(env, model_type='DQN', total_timesteps=1000):
    for _ in range(10):
        if model_type == 'DQN':
            try:
                model = DQN("MlpPolicy", env, verbose=2)
                model.learn(total_timesteps=1000)
            except Exception as e:
                print(f'⚠ Error {e}, during training')
                break
            print("✅ Training Completed\n")

            model.save(f"{model_type}_model") 
            print("✅ Model Saved\n")

            

        elif model_type == 'PPO':
            try:
                model = PPO("MlpPolicy", env, verbose=2 )
                model.learn(total_timesteps=total_timesteps)
            except Exception as e:
                print(f'⚠ Error {e}, during training')
                break
     
            print("✅ Training Completed\n")
            model.save(f"{model_type}_model") 
            print("✅ Model Saved\n")
            

        elif model_type == 'A2C':
            try:
                model = A2C("MlpPolicy", env, verbose=2 )
                model.learn(total_timesteps=total_timesteps)
            except Exception as e:
                print(f'⚠ Error {e}, during training')
                break
           
            print("✅ Training Completed\n")
            model.save(f"{model_type}_model")
            print("✅ Model Saved\n")
            
        

def load_model(env, model_type='DQN', episodes=10):
    if model_type == 'DQN':
        model = DQN.load(f"{model_type}_model")
        print("✅ Model Loaded\n")
    elif model_type == 'PPO':
        model = PPO.load(f"{model_type}_model")
        print("✅ Model Loaded\n")
    elif model_type == 'A2C':
        model = A2C.load(f"{model_type}_model")
        print("✅ Model Loaded\n")
    return model

def main():
    env = KubernetesEnv()
    print("✅ Environment Created\n")
    
    for ele in ['DQN']:
        train_model(env, model_type=ele, total_timesteps=1000)
    
 
    episode = 60
    print("✅ Environment Reset\n")
    print("✅ Starting Prediction...\n")

    # Buffer to store prediction results
    prediction_buffer = []

    while True:
        state, _ = env.reset()
        model = load_model(env, model_type='DQN', episodes=10)
        print(f"✅ Model Loaded: {ele}\n")
        for i in range(episode):
            try:
                action, _ = model.predict(state)
            except Exception as e:
                print(f'⚠ Error {e}, during prediction')
                break
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
