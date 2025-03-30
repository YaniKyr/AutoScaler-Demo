# TODO: Offline Training
# TODO: Online Training
# TODO: Separate the training and prediction scripts 
# TODO:
from stable_baselines3 import DQN
from env import KubernetesEnv
import numpy as np

def main():
    env = KubernetesEnv()
    
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500)

    model.save("dqn_model") 

    print("✅ Training Completed")

    
    

if __name__ == '__main__':

    main()
