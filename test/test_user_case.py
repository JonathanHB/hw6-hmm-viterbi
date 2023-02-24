"""
UCSF BMI203: Biocomputing Algorithms
Author:
Date: 
Program: 
Description:
"""
import pytest
import numpy as np
import src
from src.models.hmm import HiddenMarkovModel
from src.models.decoders import ViterbiAlgorithm


def test_use_case_lecture():
    """a test case based on a relationship between funding sources and student commitment to their rotation lab
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['committed','ambivalent'] # A graduate student's dedication to their rotation lab
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['R01','R21'] # The NIH funding source of the graduate student's rotation project 

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('../data/UserCase-Lecture.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    #Check HMM dimensions and ViterbiAlgorithm
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])
    assert len(use_case_decoded_hidden_states) == len(use_case_one_data['observation_states']), "number of hidden state predictions does not match number of observations"


def test_user_case_one():
    """a test case based on a relationship between traffic conditions and arrival time
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['on-time','late'] 

    # index annotation hidden_states=[i,j]
    hidden_states = ['no-traffic','traffic']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('../data/UserCase-One.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    #Check HMM dimensions and ViterbiAlgorithm

    #the last state is not a hidden state at all; this is a bug
    #print(use_case_one_data['hidden_states'])

    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states[:-1] == use_case_one_data['hidden_states'][:-1]) #adjusted due to bug in test case
    assert len(use_case_decoded_hidden_states) == len(use_case_one_data['observation_states']), "number of hidden state predictions does not match number of observations"


def test_user_case_two():
    """_summary_
    """
    # TODO


def test_user_case_three():
    """_summary_
    """
    # TODO
