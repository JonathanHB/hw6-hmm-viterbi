import copy
import numpy as np
from src.models.hmm import HiddenMarkovModel # for testing

class ViterbiAlgorithm:
    """
        an object that stores hidden markov model parameters and can decode the most likely path through the hmm hidden
        states given a series of observation states
    """    

    def __init__(self, hmm_object):
        """
        create an instance of ViterbiAlgorithm using the parameters of a hidden markov model (see below)

        Args:
            hmm_object (_type_): the hidden markov model parameters:
                - lists of observed and hidden states and dictionaries mapping those states to indices in the transition
                    and emission matrices
                - prior probabilities of each hidden state
                - transition matrix describing hidden state transition probabilities
                - emission matrix describing the probability of observing each observable state given that the system
                    is in any given hidden state
        """              
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:

        """
        Calculate the most likely path through the hmm hidden states given a series of observation states using the
        Viterbi algorithm. For each observation in a series thereof, the Viterbi algorithm calculates the relative
        probability that system was in each hidden state using the hidden state probabilities at the previous timestep
        and the transition and emission probabilities. It also calculates the previous state from which the system would
        most likely have reached each current state. The most likely path is reconstructed by following the series of
        transitions backwards from the highest-probability final state.

        Args:
            decode_observation_states (np.ndarray): a list of the observations of the system in the order that they were
                observed

        Returns:
            np.ndarray: the most likely series of hidden states that the system traversed to produce the observed
                observation states. There is a 1:1 correspondence between the states in the input observation state
                trajectory and the output hidden state trajectory.
        """
        #---------------------------------------------
        #initialize variables

        #define number of observations t and number of hidden states k
        t = len(decode_observation_states) #indexed i
        m = len(self.hmm_object.hidden_states) #indexed hsi_...

        #define matrices

        #the relative probability of the most likely path leading to each hmm state at each step [given obs.]
        path_prob = np.zeros((t, m))

        #the most recent step along the most likely path to hmm each state at each step [given obs.]
        path = np.zeros((t, m))

        #set initial relative hmm state probabilities using hmm state prior probabilities and the initial observation
        path_prob[0,:] = [self.hmm_object.prior_probabilities[hidden_state_index]*self.hmm_object.emission_probabilities[hidden_state_index, self.hmm_object.observation_states_dict[decode_observation_states[0]]]
                          for hidden_state_index in range(m)]

        #---------------------------------------------
        #fill in subsequent rows of the path and path_prob matrices to describe different paths through hmm state space
        # and their relative probabilities

        #for each frame of the trajectory
        for ii, obs in enumerate(decode_observation_states[1:]):
            # account for the first row of each matrix, which was set outside of the loop
            i = ii + 1
            #get the emission probability matrix index associated the observation from the current frame
            obs_ind = self.hmm_object.observation_states_dict[obs]

            #compute the relative probability of each possible hmm state given the probabilities
            # estimated at the previous frame of the trajectory and the observation from the current trajectory frame

            #for each possible hmm state at this frame
            for hsi_curr in range(m):
                #relative probability of reaching the current state by passing through each state at the previous timestep
                relative_last_state_probabilities = [path_prob[i - 1, hsi_prev] * self.hmm_object.transition_probabilities[hsi_prev, hsi_curr] for hsi_prev in range(m)]

                #relative probability of being in state hsi_curr given the probability of the most likely path there
                # and the probability of getting the observed emission from a system in state hsi_curr
                path_prob[i, hsi_curr] = max(relative_last_state_probabilities) * self.hmm_object.emission_probabilities[hsi_curr, obs_ind]

                #the state from which state hsi_curr was most likely reached
                #note that this variable is not actually used to compute path_prob or later rows of itself
                # it is used later to determine the most likely path
                path[i, hsi_curr] = np.argmax(relative_last_state_probabilities)

        #---------------------------------------------
        #find the most likely hidden state path by following the path matrix backwards from the most likely final state

        best_hidden_state_path_inds = np.zeros(t)
        best_hidden_state_path_inds[t-1] = np.argmax(path_prob[t-1])

        for ii in range(2, t+1):
            #run index backwards wrt observation trajectory
            i = t-ii
            #follow the pointers in the path matrix backwards in time
            best_hidden_state_path_inds[i] = path[i+1, int(best_hidden_state_path_inds[i+1])]

        #convert from state indices to state names
        best_hidden_state_path = [self.hmm_object.hidden_states[int(i)] for i in best_hidden_state_path_inds]

        #debugging output
        print("------------------------------------")
        print("observations:")
        print([self.hmm_object.observation_states_dict[obs] for obs in decode_observation_states])
        print("emission probs:")
        print(self.hmm_object.emission_probabilities)
        print("transition probs:")
        print(self.hmm_object.transition_probabilities)
        print("prior probs:")
        print(self.hmm_object.prior_probabilities)
        print("------------------------------------")
        print(best_hidden_state_path)
        print(path_prob)
        print(path)

        return best_hidden_state_path


#copied from unit tests; now unused
def use_case_lecture():
    """_summary_
    """
    # index annotation observation_states=[i,j]
    observation_states = ['committed', 'ambivalent']  # A graduate student's dedication to their rotation lab

    # index annotation hidden_states=[i,j]
    hidden_states = ['R01', 'R21']  # The NIH funding source of the graduate student's rotation project

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('../../data/UserCase-Lecture.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                                         use_case_one_data['prior_probabilities'],
                                         # prior probabilities of hidden states in the order specified in the hidden_states list
                                         use_case_one_data['transition_probabilities'],
                                         # transition_probabilities[:,hidden_states[i]]
                                         use_case_one_data[
                                             'emission_probabilities'])  # emission_probabilities[hidden_states[i],:][:,observation_states[j]]

    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

    # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states
    #
    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities,
                       use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)
    #
    # #Check HMM dimensions and ViterbiAlgorithm
    #
    # # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(
        use_case_one_data['observation_states'])

    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])

#use_case_lecture()

# trimmings

# print(self.hmm_object.observation_states_dict)
# print(self.hmm_object.prior_probabilities)
# print(self.hmm_object.emission_probabilities)

# print(path_prob)

# path[0,:] = [hidden_state_index for hidden_state_index in range(len(self.hmm_object.hidden_states))]


# return
#
# # original contents
#
#
# # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
# path = np.zeros((len(decode_observation_states),
#                  len(self.hmm_object.hidden_states)))
# path[0, :] = [hidden_state_index for hidden_state_index in range(len(self.hmm_object.hidden_states))]
#
# best_path = np.zeros((len(decode_observation_states),
#                       len(self.hmm_object.hidden_states)))
#
# # Compute initial delta:
# # 1. Calculate the product of the prior and emission probabilities. This will be used to decode the first observation state.
# # 2. Scale
# delta = np.multiply(self.hmm_object.emission_probabilities, self.hmm_object.prior_probabilities)
#
# # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
# for trellis_node in range(1, len(decode_observation_states)):
#
#     # TODO: comment the initialization, recursion, and termination steps
#
#     product_of_delta_and_transition_emission = np.multiply()
#
#     # Update delta and scale
#
#     # Select the hidden state sequence with the maximum probability
#
#     # Update best path
#     for hidden_state in range(len(self.hmm_object.hidden_states)):
#         pass
#         # Set best hidden state sequence in the best_path np.ndarray THEN copy the best_path to path
#
#     path = best_path.copy()
#
# # Select the last hidden state, given the best path (i.e., maximum probability)
#
# best_hidden_state_path = np.array([])
#
# return best_hidden_state_path