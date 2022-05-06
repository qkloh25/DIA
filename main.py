from pettingzoo.mpe import simple_adversary_v2
import numpy as np
import time

env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=True)
env.reset()
for i in range(10):
    for agent in env.agent_iter():
        print(agent)
        action = 1
        observation, reward, done, info = env.last()
        print(observation)
        print(reward)
        print(done)
        print(info)
        print(env.action_space(agent))
        print("\n")
        if not done:
            env.step(np.array([0,0,0,0.1,0]))
        else:
            env.step(None)
        env.render()
        time.sleep(0.1)


# env.reset()
# for agent in env.agent_iter():
#     print(agent)
#     if agent == "speaker_0":
#         action = np.random.randint(0,3)
#     else:
#         action = np.random.randint(0,5)
#     observation, reward, done, info = env.last()
#     print(observation)
#     if not done:
#         env.step(1)
#     else:
#         env.step(None)
#     env.render()
#     time.sleep(0.1)
