episode = 40
state = env.reset()
print()
for i in range(episode):
    action, _ = model.predict(state)
    action = int(np.argmax(action[0]))
    state, Reward, _, _, meanReward   = env.step(action)

    print(f"Episode: {i+1}, Action: {action}, State: {state}, Reward: {Reward}, Mean Reward: {meanReward}")
    