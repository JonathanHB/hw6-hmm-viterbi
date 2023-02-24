"""
UCSF BMI203: Biocomputing Algorithms
Author:
Date: 
Program: 
Description:
"""
import numpy as np
from src.hmm import HiddenMarkovModel
from src.decoders import ViterbiAlgorithm

pathprefix=""

def test_use_case_lecture():
    """a test case based on a relationship between funding sources and student commitment to their rotation lab
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['committed','ambivalent'] # A graduate student's dedication to their rotation lab
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['R01','R21'] # The NIH funding source of the graduate student's rotation project 

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load(f'{pathprefix}./data/UserCase-Lecture.npz')

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
    use_case_one_data = np.load(f'{pathprefix}./data/UserCase-One.npz')

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
    """a test case based on a relationship between traffic conditions and arrival time
    """
    # index annotation observation_states=[i,j]
    observation_states = ['early', 'on time', 'late']

    # index annotation hidden_states=[i,j]
    # It's not really clear how these would be hidden, but let's just assume that the model user avoids both windows
    # and weather forecasts and always uses an umbrella.
    hidden_states = ['sunny', 'foggy', 'rainy']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_2_data = {'prior_probabilities': [1/3, 1/3, 1/3],
                       "transition_probabilities": np.array([[0.65, 0.1, 0.3],[0.05, 0.8, 0.05],[0.3, 0.1, 0.65]]),
                       "emission_probabilities": np.array([[0.8, 0.3, 0.2],[0.1, 0.4, 0.3],[0.1, 0.3, 0.5]]),
                       "observation_states": ['early', 'early', 'on time', 'on time', 'on time', 'on time' , 'late', 'on time', 'on time', 'on time', 'on time', 'on time', 'on time', 'late', 'on time', 'early', 'late', 'late', 'late', 'early', 'late', 'early'],
                       "hidden_states": ['sunny', 'sunny', 'foggy', 'foggy', 'foggy', 'foggy', 'foggy', 'foggy', 'foggy', 'foggy', 'foggy', 'foggy', 'foggy', 'foggy', 'foggy', 'sunny', 'rainy', 'rainy', 'rainy', 'sunny', 'sunny', 'sunny']}
    #np.load('../data/UserCase-One.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_2_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                                         use_case_2_data['prior_probabilities'],
                                         # prior probabilities of hidden states in the order specified in the hidden_states list
                                         use_case_2_data['transition_probabilities'],
                                         # transition_probabilities[:,hidden_states[i]]
                                         use_case_2_data['emission_probabilities'])
                                         # emission_probabilities[hidden_states[i],:][:,observation_states[j]]

    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM
    use_case_one_viterbi = ViterbiAlgorithm(use_case_2_hmm)

    # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_2_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_2_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_2_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities,
                       use_case_2_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_2_hmm.emission_probabilities)

    # Check HMM dimensions and ViterbiAlgorithm

    # print(use_case_one_data['hidden_states'])

    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(
        use_case_2_data['observation_states'])

    assert np.alltrue(use_case_decoded_hidden_states == use_case_2_data['hidden_states'])
    assert len(use_case_decoded_hidden_states) == len(use_case_2_data['observation_states']), "number of hidden state predictions does not match number of observations"


def test_user_case_three():
    """an edge case test with a non-ergodic system represented by a non-transposable matrix
    """
    # index annotation observation_states=[i,j]
    observation_states = ['larry', 'curly', 'moe']

    # index annotation hidden_states=[i,j]
    #"But the Elves fled from him; and three of their rings they saved, and bore them away, and hid them. ...
    # But Sauron could not discover them, for they were given into the hands of the Wise, who concealed them and
    # never again used them openly while Sauron kept the Ruling Ring."
    hidden_states = ['vilya', 'nenya', 'narya']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_2_data = {'prior_probabilities': [1/3, 1/3, 1/3],
                       "transition_probabilities": np.array([[0.5, 0.5, 0],[0, 0.5, 0.5],[0.5, 0, 0.5]]),
                       "emission_probabilities": np.array([[.9, .05, .05],[.05, .9, .05],[.05, .05, .9]]),
                       "observation_states": ['larry', 'larry', 'curly', 'moe', 'larry', 'curly' , 'curly', 'moe', 'moe', 'moe', 'moe', 'moe', 'moe', 'larry'],
                       "hidden_states": ['vilya', 'vilya', 'nenya', 'narya', 'vilya', 'nenya', 'nenya', 'narya', 'narya', 'narya', 'narya', 'narya', 'narya', 'vilya']
}
    #np.load('../data/UserCase-One.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_2_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                                         use_case_2_data['prior_probabilities'],
                                         # prior probabilities of hidden states in the order specified in the hidden_states list
                                         use_case_2_data['transition_probabilities'],
                                         # transition_probabilities[:,hidden_states[i]]
                                         use_case_2_data['emission_probabilities'])
                                         # emission_probabilities[hidden_states[i],:][:,observation_states[j]]

    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM
    use_case_one_viterbi = ViterbiAlgorithm(use_case_2_hmm)

    # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_2_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_2_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_2_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities,
                       use_case_2_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_2_hmm.emission_probabilities)

    # Check HMM dimensions and ViterbiAlgorithm

    # print(use_case_one_data['hidden_states'])

    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(
        use_case_2_data['observation_states'])


    #print(use_case_decoded_hidden_states)
    #print(use_case_2_data['observation_states'])
    #print([hidden_states.index(i) for i in use_case_decoded_hidden_states])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_2_data['hidden_states'])
    assert len(use_case_decoded_hidden_states) == len(use_case_2_data['observation_states']), "number of hidden state predictions does not match number of observations"

    #It somehow worked out with hobbits, but giving the three stooges three rings of power was not Elrond's wisest decision...