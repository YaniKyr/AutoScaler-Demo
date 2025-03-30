import gymnasium
import json
import time
from functions import Prometheufunctions
import numpy as np
from tabulate import tabulate

class KubernetesEnv(gymnasium.Env):
    def __init__(self):
        super(KubernetesEnv, self).__init__()
        
        self.action = [-2, -1, 0, 1, 2]
        self.action_space = gymnasium.spaces.Discrete(len(self.action))
        self.data = {
            'state': None,
            'action': None,
            'reward': None,
            'sla': None,
            'nextstate': None
        }
        self.count = 0 
        self.rewards = []
        self.state_size = 3
        self.observation_space = gymnasium.spaces.Box(
            low=np.array([0, 0, 1]),  # Minimum values for CPU, requests, and pods
            high=np.array([1, 1000, 10]),  # Maximum values for CPU, requests, and pods
            dtype=np.float32
            )

    def reset(self, seed=None, option=None):
        super().reset(seed=seed)
        print("\u2705 Resetting Environment...\n")
        self.scaleAction(Prometheufunctions().fetchState(),0, True) 
        time.sleep(10)

        return Prometheufunctions().fetchState(), {}
        
        
    def step(self,  action_idx):
        action = self.action[action_idx]
        state = Prometheufunctions().fetchState()
        done = False

        self.scaleAction(state, action)       

        if action < 0:
            print("\u2705 Scaled down Saccessfuly")
        elif action > 0: 
            print("\u2705 Scaled up Saccessfuly")
        else:
            print("\u2705 Remaining in the same replica count")
        print('\U0001F504 Stabilizing for 30 secs...')
        time.sleep(30)

        next_state = Prometheufunctions().fetchState()
        
        reward, RTT  = self.getReward(next_state,0, flooded=False)
        self.rewards.append(reward)
        print(f"\u2705 Reward: {reward}, RTT: {RTT}ms")
        

        info = {
            "AvgRewards": np.mean(self.rewards) ,
        }

        return next_state, reward, False, False, info
        
    
    def getReward(self,data, RTT=0, flooded = False):
        p=0.9
        try:
            if not flooded:
                RTT = int(round(float(Prometheufunctions().getRTT())))

            if data[0] == 1:
                print("⚠ Warning: data[0] is 1, avoiding division by zero")
                return 0
            return ((1 - np.exp(-p * (1 - (RTT / 250)))) if RTT < 250 else (1 - np.exp(-p))) / (1 - data[0]), RTT
        except Exception as e:
            print(f'⚠ Error {e}, Prometheus Error, during data retrieval')
            return 0

    def scaleAction(self,state, action, _ResetAction = False):
        self.count += 1
        target_pods = 1
        

        if action + state[2] > 9 or action + state[2] < 1:
            action = 0

        if not _ResetAction:
            target_pods = state[2] + action

        file = '/tmp/shared_file.json'
        table_data = [
            ["Iter","_ResetAction","Action", "State", "Scaling to"],
            [self.count, _ResetAction, action, state, target_pods]
        ]
        print(tabulate(table_data, tablefmt="grid"))
        # Write scaling action
        with open(file, 'w') as file:
            json.dump({'action': int(target_pods)}, file)

        # Wait for Kubernetes to reach the target
        
        #keda might have a bug. When reaching max Replicas e.g. 10 and trying to scale down to 9, it fails
        #to perform the operation. In the other hand all the other scaling actions work properly
        start_time = time.time()
        while True:
            
            try:
                curr_state = Prometheufunctions().fetchState()[2] 
            except Exception as e:
                print(f'⚠ Error while fetching pods, encountered {e}')
                time.sleep(1)
                continue

            if curr_state == target_pods:
                break


            elapsed_time = time.time() - start_time  # Calculate the elapsed time
            #Grace Period
            if elapsed_time > 45:
                print("⚠ Error: Timeout exceeded while waiting for pods to scale! Restarting...")
                start_time = time.time()
                continue
                #print("Timeout waiting for pods to scale.")
            
            time.sleep(5)
        

