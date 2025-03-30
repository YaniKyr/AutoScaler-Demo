from stable_baselines3 import DQN
from env import KubernetesEnv
import numpy as np

def main():
    env = KubernetesEnv()
    
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500)

    model.save("dqn_model")

    print("✅ Training Completed")

    episode = 40
    state = env.reset()
    print()
    for i in range(episode):
        action, _ = model.predict(state)
        action = int(np.argmax(action[0]))
        state, Reward, _, _, meanReward   = env.step(action)

        print(f"Episode: {i+1}, Action: {action}, State: {state}, Reward: {Reward}, Mean Reward: {meanReward}")
        
    

if __name__ == '__main__':

    main()
