import gym
import time
import numpy as np
import random
import math

# for DQN
# enforce to use CPU only
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, num_observations, num_actions):
        self.num_observations= num_observations
        self.num_actions = num_actions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.exploration_rate = 1.0
        self.exploration_rate_min = 0.01
        self.exploration_rate_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.num_observations, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.num_actions, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, observation_curret, action, reward, observation_next, done):
        self.memory.append((observation_curret, action, reward, observation_next, done))

    def select_action(self, observation):
        if random.random() <= self.exploration_rate:
            action = random.randrange(self.num_actions)

        action_values = self.model.predict(observation)
        return np.argmax(action_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for observation_curret, action, reward, observation_next, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(observation_next)[0]))
            target_f = self.model.predict(observation_curret)
            target_f[0][action] = target
            self.model.fit(observation_curret, target_f, epochs=1, verbose=0)

        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate *= self.exploration_rate_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def train_DQNAgent(env):
    # task parameters
    max_episodes = 1000  # maximum number of episodes to try
    max_time_steps = 500  # maximum number of time steps to try for each episode
    success_time_threshold = 199  # a successful episode needs 199 successful time steps
    num_success_episodes_so_far = 0
    num_success_episodes_to_declare_victory = 5  # solved if we have 100 consecutive successful episodes (Gym)
    render_flag = False  # flag to render the environment

    num_observations = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent = DQNAgent(num_observations, num_actions)
    # agent.load("./save/cartpole-dqn.h5")
    batch_size = 32

    for e in range(max_episodes):
        print("Episode {} starting...".format(e))

        # get the initial observation of the environment
        observation_current = env.reset()

        # reshape the observation for later training
        observation_current = np.reshape(observation_current, [1, num_observations])

        # start to learn
        for t in range(max_time_steps):
            # render the environment
            if render_flag:
                print("time step: {}".format(t))
                env.render()
                time.sleep(0.05)

            # select an action according epsilon-greedy algorithm
            action = agent.select_action(observation_current)

            # execute one step with the selected action, collect info
            observation_next, reward, done, _ = env.step(action)

            # reshape the observation
            observation_next = np.reshape(observation_next, [1, num_observations])

            # assign the reward for later learning
            reward = reward if not done else -10

            # store this experience in memory
            agent.memorize(observation_current, action, reward, observation_next, done)

            # update the observation
            observation_current = observation_next

            # if this episode is finished
            if done:
                print("Episode {} finished after {} successful time steps".format(e, t + 1))
                if t >= success_time_threshold:  # successful
                    num_success_episodes_so_far += 1
                else:  # failed
                    num_success_episodes_so_far = 0
                break

            # if memory is big enough, replay.
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        print("Number of consecutive successful episodes: {}\n".format(num_success_episodes_so_far))

        # render the environment for the last successful episode
        if num_success_episodes_so_far > num_success_episodes_to_declare_victory - 1:
            render_flag = True

        # if we have solved the CartPole problem, stop.
        if num_success_episodes_so_far > num_success_episodes_to_declare_victory:
            break

        # store the learned model
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

class QAgent:
    def __init__(self, env):
        self.env = env
        self.num_actions = self.env.action_space.n
        # hyper-parameters
        self.gamma = 0.99  # discount rate
        self.exploration_rate_min = 0.01
        self.learning_rate_min = 0.1
        # other empirical settings
        self.observation_buckets = (1, 1, 6, 3)
        self.Q_table = np.zeros(self.observation_buckets + (self.num_actions,))
        self.observation_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        self.observation_bounds[1] = [-0.5, 0.5]  # changed from inf for velocity
        self.observation_bounds[3] = [-math.radians(50), math.radians(50)]  # changed from inf for angular velocity

    def select_action(self, observation, exploration_rate):
        if random.random() < exploration_rate:
            action = random.randrange(self.num_actions)  # smaller than the exploration rate
        else:
            action = np.argmax(self.Q_table[observation])
        return action

    def select_exploration_rate(self, x):
        # decrease the exploration rate as time advances
        #return math.pow(0.995, x)
        return max(self.exploration_rate_min, min(1, 1.0 - math.log10((x + 1) / 25)))

    def select_learning_rate(self, x):
        # decrease the learning rate as time advances
        #return math.pow(0.995, x)
        return max(self.learning_rate_min, min(0.5, 1.0 - math.log10((x + 1) / 25)))

    def discretize_observation(self, observation):
        bucket_indices = []
        # process each observation entry to its discrete index
        for i in range(len(observation)):
            if observation[i] <= self.observation_bounds[i][0]:
                # smaller than the lower bound, assign the bucket index 0
                bucket_index = 0
            elif observation[i] >= self.observation_bounds[i][1]:
                # greater than the upper bound, assign the bucket index (max)
                bucket_index = self.observation_buckets[i] - 1
            else:
                bound_width = self.observation_bounds[i][1] - self.observation_bounds[i][0]
                offset = (self.observation_buckets[i] - 1) * self.observation_bounds[i][0] / bound_width
                scaling = (self.observation_buckets[i] - 1) / bound_width
                bucket_index = int(round(scaling * observation[i] - offset))
            bucket_indices.append(bucket_index)
        return tuple(bucket_indices)

def train_QAgent(env):
    # task parameters
    max_episodes = 1000  # maximum number of episodes to try
    max_time_steps = 250  # maximum number of time steps to try for each episode
    success_time_threshold = 199  # a successful episode needs 199 successful time steps
    num_success_episodes_so_far = 0
    num_success_episodes_to_declare_victory = 100  # solved if we have 100 consecutive successful episodes (Gym)
    render_flag = False  # flag to render the environment

    # get the Q learning agent
    agent = QAgent(env)

    for e in range(max_episodes):
        # get exploration and learning rates for each episode
        exploration_rate = agent.select_exploration_rate(e)
        learning_rate = agent.select_learning_rate(e)
        print("Episode {} starting...".format(e))
        print("Exploration rate: {}".format(exploration_rate))
        print("Learning rate: {}".format(learning_rate))

        # get the initial observation of the environment
        observation_current = env.reset()

        # discretize observations into their indices
        observation_current = agent.discretize_observation(observation_current)

        # start to learn
        for t in range(max_time_steps):
            # render the environment
            if render_flag:
                print("time step: {}".format(t))
                env.render()
                time.sleep(0.05)

            # select an action according epsilon-greedy algorithm
            action = agent.select_action(observation_current, exploration_rate)

            # execute one step with the selected action, collect info
            observation_next, reward, done, _ = env.step(action)

            # discretize the continuous observation from the environment
            observation_next = agent.discretize_observation(observation_next)

            # retrieve the best Q value
            best_Q = np.amax(agent.Q_table[observation_next])

            # update the Q value table
            agent.Q_table[observation_current + (action,)] += \
                learning_rate * (reward + agent.gamma * best_Q - agent.Q_table[observation_current + (action,)])

            # the episode is finished, it can be either successful or failed
            if done:
                if t >= success_time_threshold:  # successful
                    print("Episode {} finished after {} successful time steps".format(e, t + 1))
                    num_success_episodes_so_far += 1
                else:  # failed
                    num_success_episodes_so_far = 0
                break

            # update the observation so that we can use it to further update the Q value table
            observation_current = observation_next

        print("Number of consecutive successful episodes: {}\n".format(num_success_episodes_so_far))

        # render the environment for the last successful episode
        if num_success_episodes_so_far > num_success_episodes_to_declare_victory - 1:
            render_flag = True

        # if we have solved the CartPole problem, stop.
        if num_success_episodes_so_far > num_success_episodes_to_declare_victory:
            break


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    train_DQNAgent(env)
    #train_QAgent(env)
    env.close()


