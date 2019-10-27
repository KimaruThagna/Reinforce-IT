import gym
from Agent import DQNAgent
import numpy as np
episodes = 5000
env = gym.make('CartPole-v0')
agent = DQNAgent(4,2)

for e in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, 4])
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            env.render()
            # Decide action
            action = agent.act(state)
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action) # advance based on action
            next_state = np.reshape(next_state, [1, 4])
            agent.remember(state, action, reward, next_state, done)# Remember the previous state, action, reward, and done

            state = next_state # make step

            if done: # true if game ends
                # print the score and break out of the loop
                print(f'episode: {e}/{episodes}, score: {time_t}')
                break
        # train the agent with the experience of the episode
        if e >40: # allow memory to build so as to access a batch size of atleast 32 items
            agent.replay(32)

