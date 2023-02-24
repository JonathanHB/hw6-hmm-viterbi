import numpy as np
class HiddenMarkovModel:
    """
    a hidden markov model object
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_probabilities: np.ndarray, transition_probabilities: np.ndarray, emission_probabilities: np.ndarray):
        """
        initialize a hidden markov model with the parameters described below

        Args:
            observation_states (np.ndarray): a list of the possible observations resulting from measuring the system
            hidden_states (np.ndarray): a list of the possible hidden system states, which are not directly observable
            prior_probabilities (np.ndarray): the initial probabilities of the hidden states given no other information
            transition_probabilities (np.ndarray): the probability of the system transitioning from each hidden state to
                each other hidden state (over the same time step separating consecutive observations in this implementation)
            emission_probabilities (np.ndarray): the probability of measuring each observation state given that the
                system occupies each hidden state
        """             
        self.observation_states = observation_states
        self.observation_states_dict = {observation_state: observation_state_index \
                                   for observation_state_index, observation_state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {hidden_state_index: hidden_state \
                                   for hidden_state_index, hidden_state in enumerate(list(self.hidden_states))}
        

        self.prior_probabilities = prior_probabilities
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities