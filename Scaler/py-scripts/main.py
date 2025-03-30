from stable_baselines3 import DQN
from env import KubernetesEnv


    

def main():
    env = KubernetesEnv()
    
    model = DQN("MlpPolicy", env, verbose=1).learn(1000)
    episode = 40
    state = env.reset()
    for i in range(episode):
        action, _ = model.predict(state)
        data, _,   = env.step(action)
        state = data['nextstate']
        
        print(f"Episode: {i+1}, Action: {action}, State: {state}, Reward: {data['reward']}")
        
    

if __name__ == '__main__':

    main()
