import copy
import numpy as np
class ViterbiAlgorithm:
    """_summary_
    """    

    def __init__(self, hmm_object):
        """_summary_

        Args:
            hmm_object (_type_): _description_
        """              
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            decode_observation_states (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """        
        
        # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
        path = np.zeros((len(decode_observation_states), 
                         len(self.hmm_object.hidden_states)))
        path[0,:] = [hidden_state_index for hidden_state_index in range(len(self.hmm_object.hidden_states))]

        best_path = np.zeros((len(decode_observation_states), 
                         len(self.hmm_object.hidden_states)))        
        
        # Compute initial delta:
        # 1. Calculate the product of the prior and emission probabilities. This will be used to decode the first observation state.
        # 2. Scale      
        delta = np.multiply()

        # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
        for trellis_node in range(1, len(decode_observation_states)):

            # TODO: comment the initialization, recursion, and termination steps

            product_of_delta_and_transition_emission =  np.multiply()
            
            # Update delta and scale

            # Select the hidden state sequence with the maximum probability

            # Update best path
            for hidden_state in range(len(self.hmm_object.hidden_states)):
                pass
                # Set best hidden state sequence in the best_path np.ndarray THEN copy the best_path to path

            path = best_path.copy()

        # Select the last hidden state, given the best path (i.e., maximum probability)

        best_hidden_state_path = np.array([])

        return best_hidden_state_path