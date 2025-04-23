from stable_baselines3 import DQN, PPO, A2C
from env import KubernetesEnv
import numpy as np
import os

def load_model(env, model_type='DQN', verbose=2):
    
    
    try:
        model = DQN.load(f"dqn_model.zip")
        return model
    except Exception as e:
        print(f'⚠ Error {e}, during model loading')
        return None
    


def main():
    print("⚙️  Init Env...\n")
    env = KubernetesEnv()

    episode = 60
    print("✅ Environment Reset\n")
    print("⚙️  Starting Prediction...\n")
    # Buffer to store prediction results
    prediction_buffer = []
    state, _ = env.reset()
    model = load_model(env, model_type='DQN')
    
    
    for i in range(episode):
        try:
            action, _ = model.predict(np.array(state))
        except Exception as e:
            print(f'⚠ Error {e}, during prediction')
            break
        state, Reward, _, _, meanReward = env.step(action)

        # Save results to buffer
        prediction_buffer.append({
            "Episode": i + 1,
            "Model": "DQN",
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

if __name__ == "__main__":
    main()