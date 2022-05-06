import numpy as np
import time
from pettingzoo.mpe import simple_adversary_v2
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, observation[obs]])
    return state

if __name__ == '__main__':
    scenario = 'simple_adversary'

    PRINT_INTERVAL = 500
    N_GAMES = 5_000_000
    MAX_STEPS = 50
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0
    scenario = 'simple_adversary_v2'
    env = simple_adversary_v2.parallel_env(N=2, max_cycles=MAX_STEPS, continuous_actions=True)
    env.reset()
    n_agents = 3
    actor_dims = []
    score_history = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space(env.agents[i]).shape[0])
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = []
    for i in range(n_agents):
        n_actions.append(env.action_space(env.agents[i]).shape[0])

    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)
    
    

    if evaluate:
        maddpg_agents.load_checkpoint()
    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not all(done):
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)
            done = np.fromiter(done.values(), dtype=bool)              
            reward = np.fromiter(reward.values(), dtype=float)              
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)
            if i % (PRINT_INTERVAL*10) == 0:
                env.render()
                # time.sleep(0.1) # to slow down the action for the video
            if episode_step >= MAX_STEPS:
                done = [True]*n_agents

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_
            score += sum(reward)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))