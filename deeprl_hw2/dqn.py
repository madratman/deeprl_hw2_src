"""Main DQN agent."""
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Reshape, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D
from objectives import mean_huber_loss
import gym
import numpy as np
from policy import *
from preprocessors import *
from core import *
import matplotlib.pyplot as plt
import cPickle as pkl

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 env,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size):

        self.env_string = env
        self.env = gym.make(env)
        self.num_of_actions = self.env.action_space.n
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.iter_ctr = 0

        self.qavg_list = np.array([0])
        self.reward_list = []
        self.eval_episode_ctr = 0
        self.atari_preprocessor = AtariPreprocessor()

    def create_model(self, num_actions):  # noqa: D103
        """Create the Q-network model.

        Use Keras to construct a keras.models.Model instance (you can also
        use the SequentialModel class).

        We highly recommend that you use tf.name_scope as discussed in
        class when creating the model and the layers. This will make it
        far easier to understnad your network architecture if you are
        logging with tensorboard.

        Parameters
        ----------
        window: int
          Each input to the network is a sequence of frames. This value
          defines how many frames are in the sequence.
        input_shape: tuple(int, int)
          The expected input image size.
        num_actions: int
          Number of possible actions. Defined by the gym environment.
        model_name: str
          Useful when debugging. Makes the model show up nicer in tensorboard.

        Returns
        -------
        keras.models.Model
          The Q-model.
        """
        model = Sequential()
        model.add(Convolution2D(filters=16, kernel_size=(8,8), strides=(4,4), input_shape=(84,84,4), activation='relu', name='conv_1'))
        model.add(Convolution2D(filters=32, kernel_size=(4,4), strides=(2,2), activation='relu', name='conv_2'))
        model.add(Convolution2D(filters=32, kernel_size=(4,4), strides=(2,2), activation='relu', name='fc_1'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', name='fc_2'))
        model.add(Dense(num_actions, activation='relu', name='final'))
        return model

    def compile(self, num_actions, optimizer='Adam'):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        self.q_network = self.create_model(num_actions)
        self.target_q_network = self.create_model(num_actions)
        self.q_network.compile(loss=mean_huber_loss, optimizer='adam') 
        self.target_q_network.compile(optimizer='adam', loss=mean_huber_loss) #todo metrics 

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.
        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """ 
        # state = self.atari_preprocessor.process_state_for_network(state)
        q_vals = self.q_network.predict(np.swapaxes(state,0,3))
        return q_vals

    # we're not using this at all 
    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        q_values = self.calc_q_values(self.preprocessor.process_state_for_network(state))
        
        if kwargs['stage'] == "burning_in":
            self.policy = UniformRandomPolicy()
            return self.policy.select_action()

        if kwargs['stage'] == "training":
            self.policy = LinearDecayGreedyEpsilonPolicy()
            return self.policy.select_action(q_values)

        if kwargs['stage'] == "testing":
            self.policy = GreedyPolicy()
            return self.policy.select_action()

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        # this is update_policy 
        # sample batch of 32 from the memory
        batch_of_samples = self.replay_memory.sample(batch_size=32)
        current_state_samples = batch_of_samples['current_state_samples']
        next_state_samples = batch_of_samples['next_state_samples']
        #print type(current_state_samples[0])

        # fetch stuff we need from samples 32*84*84*4
        current_state_images = np.zeros([32, 84, 84, 4])
        for (idx, each_list_of_samples) in enumerate(current_state_samples):
            current_state_images[idx, ...] = np.dstack([sample.state for sample in each_list_of_samples])

        next_state_images = np.zeros([32, 84, 84, 4])
        for (idx, each_list_of_samples) in enumerate(next_state_samples):
            next_state_images[idx, ...] = np.dstack([sample.state for sample in each_list_of_samples])

        # preprocess
        current_state_images = self.atari_preprocessor.process_batch(current_state_images)
        next_state_images = self.atari_preprocessor.process_batch(next_state_images)

        q_current = self.q_network.predict(current_state_images) # 32*num_actions
        q_next = self.target_q_network.predict(next_state_images)

        # targets
        y_targets_all = q_current #32*num_actions

        for (idx, each_list_of_samples) in enumerate(current_state_samples):
            last_sample = each_list_of_samples[-1]
            if last_sample.is_terminal:
                y_targets_all[idx, last_sample.action] = last_sample.reward
            else:
                y_targets_all[idx, last_sample.action] = last_sample.reward + self.gamma*np.max(q_next[idx])
                # bla = self.target_q_network.predict(next_state_images[idx])
                # y_target_curr = reward_proc + self.gamma*bla[np.argmax(self.q_network(next_state_images[idx]))]

        loss = self.q_network.train_on_batch(current_state_images, np.float32(y_targets_all))

        if (self.iter_ctr > (self.num_burn_in+1)) and not(self.iter_ctr%self.target_update_freq):
            # copy weights
            print "Iter {} Updating target Q network".format(self.iter_ctr)
            [self.target_q_network.trainable_weights[i].assign(self.q_network.trainable_weights[i]) \
                for i in range(len(self.target_q_network.trainable_weights))]

    def fit(self, num_iterations, max_episode_length=250, eval_every_nth=1000, save_model_every_nth=1000):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        self.compile(self.num_of_actions)

        random_policy = UniformRandomPolicy(num_actions=self.num_of_actions) # for burn in 
        self.policy = LinearDecayGreedyEpsilonPolicy(start_value=1., end_value=0.1, num_steps=1e6) # for training
        history_memory = HistoryPreprocessor(history_length=4)
        self.replay_memory = ReplayMemory(max_size=1000000)

        num_episodes = 0
        while self.iter_ctr < num_iterations:
            state = self.env.reset()
            num_timesteps_in_curr_episode = 0

            while num_timesteps_in_curr_episode < max_episode_length:
                self.iter_ctr+=1 # number of steps overall
                num_timesteps_in_curr_episode += 1 # number of steps in the current episode

                # logging
                # if not self.iter_ctr % 1000:
                    # print "iter_ctr {}, num_episodes : {} num_timesteps_in_curr_episode {}".format(self.iter_ctr, num_episodes, num_timesteps_in_curr_episode)

                # append to history
                history_memory.process_state_for_network(self.atari_preprocessor.process_state_for_memory(state))

                # burning in 
                if self.iter_ctr < self.num_burn_in:
                    action = random_policy.select_action() # goes from 0 to n-1
                    next_state, reward, is_terminal, _ = self.env.step(action)
                    self.replay_memory.append(self.atari_preprocessor.process_state_for_memory(state), action, \
                                                self.atari_preprocessor.process_reward(reward), is_terminal)

                    if is_terminal or (num_timesteps_in_curr_episode > max_episode_length-1):
                        state = self.env.reset()
                        print "iter_ctr {}, num_episodes : {} num_timesteps_in_curr_episode {}".format(self.iter_ctr, num_episodes, num_timesteps_in_curr_episode)
                        num_episodes += 1
                        num_timesteps_in_curr_episode = 0
                        # this should be called when num_timesteps_in_curr_episode > max_episode_length, but we can call it in is_terminal as well. 
                        # it won't change anything as it just sets the last entry's is_terminal to True
                        self.replay_memory.end_episode() 
                        break

                # training
                else:
                    # print "iter_ctr {}, num_episodes : {} num_timesteps_in_curr_episode {}".format(self.iter_ctr, num_episodes, num_timesteps_in_curr_episode)
                    history = history_memory.get_history()
                    q_values = self.calc_q_values(history)
                    #print "q_values.shape ", q_values.shape
                    action = self.policy.select_action(q_values=q_values, is_training=True)
                    next_state, reward, is_terminal, _ = self.env.step(action)
                    self.replay_memory.append(self.atari_preprocessor.process_state_for_memory(state), action, \
                                                self.atari_preprocessor.process_reward(reward), is_terminal)

                    # validation. keep this clause before the breaks!
                    if not(self.iter_ctr%eval_every_nth):
                        print "\n\nEvaluating at iter {}".format(self.iter_ctr)
                        self.evaluate(num_episodes=20, max_episode_length=max_episode_length)
                        print "Done Evaluating\n\n"

                    # save model
                    if not(self.iter_ctr%save_model_every_nth):
                        self.q_network.save('/data/datasets/ratneshm/deeprl_hw2/q_network_{}.h5'.format(str(self.iter_ctr).zfill(7)))

                    if num_timesteps_in_curr_episode > max_episode_length-1:
                        state = self.env.reset() # being safe
                        print "iter_ctr {}, num_episodes : {} num_timesteps_in_curr_episode {}".format(self.iter_ctr, num_episodes, num_timesteps_in_curr_episode)
                        self.replay_memory.end_episode()
                        num_episodes += 1
                        num_timesteps_in_curr_episode = 0
                        break

                    if is_terminal:
                        state = self.env.reset()
                        print "iter_ctr {}, num_episodes : {} num_timesteps_in_curr_episode {}".format(self.iter_ctr, num_episodes, num_timesteps_in_curr_episode)
                        num_episodes += 1
                        num_timesteps_in_curr_episode = 0
                        break

                    if not(self.iter_ctr % self.train_freq):
                        self.update_policy()

                state = next_state

    def evaluate(self, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        evaluation_policy = GreedyPolicy()

        iter_ctr_valid = 0
        Q_sum = 0
        history_memory = HistoryPreprocessor(history_length=4)
        eval_episode_ctr_valid = 0
        env_valid = gym.make(self.env_string)

        while eval_episode_ctr_valid < num_episodes:
            state = env_valid.reset()
            num_timesteps_in_curr_episode = 0
            total_reward_all_episodes = []
            total_reward_curr_episode = 0

            while num_timesteps_in_curr_episode < max_episode_length:
                num_timesteps_in_curr_episode += 1
                iter_ctr_valid += 1

                history_memory.process_state_for_network(self.atari_preprocessor.process_state_for_memory(state))
                history = history_memory.get_history()
                q_values = self.calc_q_values(history)
                Q_sum += np.max(q_values) # todo fix this

                action = evaluation_policy.select_action(q_values)
                next_state, reward, is_terminal, _ = env_valid.step(action)
                reward = self.atari_preprocessor.process_reward(reward)
                total_reward_curr_episode += reward

                if is_terminal or (num_timesteps_in_curr_episode > max_episode_length-1):
                    state = env_valid.reset() 
                    # print "Evaluate() : iter_ctr_valid {}, eval_episode_ctr_valid : {} num_timesteps_in_curr_episode {}".format(iter_ctr_valid, eval_episode_ctr_valid, num_timesteps_in_curr_episode)
                    total_reward_all_episodes.append(total_reward_curr_episode)
                    print "total_reward_curr_episode ", total_reward_curr_episode
                    eval_episode_ctr_valid += 1
                    num_timesteps_in_curr_episode = 0
                    break

                state = next_state

        Q_avg = Q_sum/float(iter_ctr_valid)
        all_episode_avg_reward = sum(total_reward_all_episodes)/float(len(total_reward_all_episodes))
        self.qavg_list = np.append(self.qavg_list, Q_avg)
        self.reward_list.append(all_episode_avg_reward)

        pkl.dump(self.reward_list, open("/data/datasets/ratneshm/deeprl_hw2/eval_rewards.pkl", "wb"))
        
        print "all_episode_avg_reward ", all_episode_avg_reward
        print "\n\n\n self.reward_list \n\n\n", self.reward_list


        # plt.plot(np.asarray(range(eval_episode_ctr_valid)), self.reward_list)
        # plt.xlabel('Epochs')
        # plt.ylabel('Avg reward per episode')
        # plt.title('Avg reward per episode during training')
        # plt.grid(True)
        # plt.savefig("rewardPlot.png")

        # plt.clear()
        # plt.plot(np.asarray(range(eval_episode_ctr_valid)), self.qavg_list)
        # plt.xlabel('Epochs')
        # plt.ylabel('Avg Q per step')
        # plt.title('Avg Q per step during training')
        # plt.grid(True)
        # plt.savefig("qPlot.png")


