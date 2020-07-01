import datetime
import gym
import numpy as np
import tensorflow as tf
from dqn_agent import DQN
from ddqn_agent import DDQN
from player_bot import PLAYER
from dqn_fixed import DQN_F
import os
import glob
import sys
import matplotlib.pyplot as plt

# Run 'tensorboard --logdir (logs|play_logs)' to see detailed learning statistics

env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

def train_dqn(episode, net):

    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    loss = []
    if net == "ddqn":
        agent = DDQN(env.action_space.n, env.observation_space.shape[0])
    elif net == "dqn_fixed":
        agent = DQN_F(env.action_space.n, env.observation_space.shape[0])
    else:
        agent = DQN(env.action_space.n, env.observation_space.shape[0])

    current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
    log_dir = 'logs/'+net+'/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, agent.state_space))
        score = 0
        done = False
        i = 0
        while not done:
            action = agent.act_greedy_policy(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, (1, agent.state_space))
            score += reward
            agent.remember(state, action, reward, next_state, done)
            #env.render()
            state = next_state
            agent.experience_replay()
            i += 1
            if i % 100 == 0 and not isinstance(agent, DQN):
                agent.copy_weights()
                # if you want to try a soft_update with the DDQN, substitute with agent.soft_update_weights()
                #agent.soft_update_weights()
            if done:
                print("Episode: {}/{}, score: {}, eps: {}".format(e+1, episode, np.round(score, decimals=2), np.round(agent.epsilon, decimals=2)))
                break
        loss.append(score)

        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        count_is_solved = 0
        with summary_writer.as_default():
            tf.summary.scalar('Episode reward', score, step=e)
            tf.summary.scalar('Avg reward (last 100):', is_solved, step=e)

        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
        if is_solved >= 200:
            print('\n Task Completed! \n')
            if count_is_solved == 0:
                count_is_solved += 1
                agent.dqn_network.model.save_weights(
                    './checkpoints/' + net + '_' + current_time + '.h5')
                if isinstance(agent, DDQN) or isinstance(agent, DQN_F):
                    agent.target_network.model.save_weights(
                        './checkpoints/' + net + 'target' + '_' + current_time + '.h5')

    return loss

def play_game():

    # Load an already trained model
    player = PLAYER(env.observation_space.shape[0], env.action_space.n)

    try:
        print("\nInsert the number of the file (.h5) to load:\n")
        files = [os.path.basename(x) for x in glob.glob('./checkpoints/*.h5')]
        for n_file in range(len(files)):
            print('%d: %s' % (n_file + 1, files[n_file]))
        print()
        file_to_load = int(input())
        player.model.load_weights('./checkpoints/' + files[file_to_load - 1])
    except:
        print("Something's wrong, check if the file exists or if you are in the root of the checkpoints folder")
        sys.exit(1)

    # Variables for the tensorboard statistics
    log_dir = 'play_logs/' + files[file_to_load - 1] + '-'
    summary_writer = tf.summary.create_file_writer(log_dir)

    n_goal = 0
    for episode in range(100):  # play for 100 episodes
        state = env.reset()
        score = 0
        for step in range(3000):  # max steps
            q_values = player.model.predict(state[np.newaxis])
            action = np.argmax(q_values[0])
            state, reward, done, _ = env.step(action)
            score += reward
            env.render()
            if done:
                print("Episode: {}/100, score: {}".format(episode, np.round(score, decimals=2)))
                if score > 200:
                    n_goal += 1
                    print('Task completed!')
                else:
                    print('Goal not achieved')

                with summary_writer.as_default():
                    tf.summary.scalar('Episode reward', score, step=episode)
                    tf.summary.scalar('Goals achieved', n_goal, step=episode)

                break
    print('\nTotal Goals achieved: ', n_goal)
if __name__ == '__main__':
    print("""\
    
    
        |                              |                     |          
     |     |   | __ \   _` |  __|   |      _` | __ \   _` |  _ \  __|
     |     |   | |   | (   | |      |     (   | |   | (   |  __/ |   
    _____|\__,_|_|  _|\__,_|_|     _____|\__,_|_|  _|\__,_|\___|_| 
                          
                AUTONOMOUS AND ADAPTIVE SYSTEM PROJECT  
    """)
    mode = input('\nDo you want to play a trained agent or to train? ["play", "train"]: ')
    train_episodes = 600

    if mode == "train":
        net = input('What network do you want to use? ["dqn", "dqn_fixed", "ddqn"]: ')
        net = net.lower()
        if net != "dqn" and net != "ddqn" and net != "dqn_fixed":
            print("Wrong argument\n")
            exit(1)

        print('State space: ' + str(env.observation_space))
        print('Action space: ' + str(env.action_space))
        loss = train_dqn(train_episodes, net)

        # Uncomment this if you want to plot the current statistics without using tensorboard
        '''plt.xlabel("Episodes")
        plt.ylabel("Scores")
        plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
        plt.show()'''

    elif mode == "play":
        play_game()
    else:
        print("Wrong argument\n")
        exit(1)
    env.close()
