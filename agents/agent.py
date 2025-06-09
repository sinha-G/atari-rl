import abc

class Agent(abc.ABC):
    """
    Abstract base class for a reinforcement learning agent.
    """
    def __init__(self, state_space_shape, action_space_size):
        """
        Initializes the agent.

        Args:
            state_space_shape (tuple): The shape of the state space.
            action_space_size (int): The number of possible actions.
        """
        self.state_space_shape = state_space_shape
        self.action_space_size = action_space_size

    @abc.abstractmethod
    def act(self, state):
        """
        Selects an action for the given state.

        Args:
            state: The current state.

        Returns:
            int: The action to take.
        """
        pass

    @abc.abstractmethod
    def learn(self, state, action, reward, next_state, done):
        """
        Updates the agent's knowledge based on an experience.

        Args:
            state: The state before the action.
            action: The action taken.
            reward: The reward received.
            next_state: The state after the action.
            done (bool): Whether the episode has ended.
        """
        pass

    @abc.abstractmethod
    def save(self, path):
        """
        Saves the agent's model/parameters.

        Args:
            path (str): The path to save the model.
        """
        pass

    @abc.abstractmethod
    def load(self, path):
        """
        Loads the agent's model/parameters.

        Args:
            path (str): The path to load the model from.
        """
        pass
    