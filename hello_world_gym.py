import gym
env = gym.make('CartPole-v0') # create environment. Can try out MountainCar-v0, MsPacman-v0, Hopper-v1
env.reset()
for _ in range(100000):
    env.render() # display
    env.step(env.action_space.sample()) # take a random action
env.close()