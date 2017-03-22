"""Main DQN agent."""
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Reshape, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D
from objectives import mean_huber_loss

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
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 num_of_actions,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size):

        self.q_network = q_network #todo
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy #todo don't need this
        self.num_of_actions = num_of_actions
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size

    def create_model(window, input_shape, num_actions,
                     model_name='q_network'):  # noqa: D103
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
        model.add(Convolution2D(16,8,8, strides=(4,4), input_shape=(4,84,84), activation='relu', name='conv_1'))
        model.add(Convolution2D(32,4,4, strides=(2,2), activation='relu', name='conv_2'))
        model.add(Dense(32,4,4, strides=(2,2), activation='relu', name='fc_1'))
        model.add(Flatten())
        model.add(Dense, 256, activation='relu', name='fc_2')
        model.add(Dense, num_of_actions, activation='relu', name='final')
        return model

    def compile(self, optimizer='Adam', loss_func):
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
        self.q_network = self.create_model()
        self.target_q_network = self.create_model()
        self.q_network.compile(optimizer=loss_func, loss=mean_huber_loss) 
        self.target_q_network.compile(optimizer=loss_func, loss=mean_huber_loss) #todo metrics 

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.
        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """ 
        state = self.preprocessor.process_state_for_network(state)
        q_vals = self.q_network.predict(state)
        return q_vals

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
        
        if kwargs['stage'] == "burning_in"
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


    def fit(self, env, num_iterations, max_episode_length=None):
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

        iter_ctr = 0
        state = env.reset()

        random_policy = UniformRandomPolicy() # for burn in 
        self.policy = LinearDecayGreedyEpsilonPolicy() # for training
        history_memory = HistoryPreprocessor()

        while 1:
            iter_ctr+=1
            if iter_ctr < self.num_burn_in:
                action = random_policy.select_action()
                next_state, reward, is_terminal, _ = env.step(action)
                state_dump = atari_processor.process_state_for_memory(state)

                self.memory.append((state_dump, action, reward, is_terminal))
                state = next_state

            else:
                # note that history_memory is just saving the last 4 states. It ain't doing any image manipulation. 
                history_memory.process_state_for_network(atari_processor.process_state_for_memory(state))
                q_values = self.calc_q_values(history_memory)
                action = self.policy.select_action()
                next_state, reward, is_terminal, _ = env.step(action)

                self.memory.append((atari_processor.process_state_for_memory(state), action, \
                                    atari_processor.process_reward(reward), is_terminal))
                
                if not(is_terminal):
                    if ctr_episode_timestep > max_episode_length-2:
                        self.memory.end_episode()
                        break
                    else:
                        state = next_state
                else:
                    break

                if not (iter_ctr%train_freq):
                    # this is update_policy 
                    # sample batch of 32 from the memory
                    sample_batch = self.memory.sample(self.batch_size)
                    sample_states = [sample(0) for sample in sample_batch]
                    # make numpy array 
                    sample_batch = self.processor.process_batch(sample_states)

                    states = 

                    if is_terminal:
                        y_target = reward
                    else:
                        y_target = reward + self.gamma*np.max(self.q_network.predict(stack_of_4_states))

                    x_batch = [sample_batch[] for each_sample in sample_batch]
                    y_batch = 

                    if iter_ctr % target_update_freq:
                        # copy weights

                    # train on batch of 32 
                    # self.q_network.train_on_batch()

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        initial_state = env.reset()
        self.policy = GreedyPolicy()

