"""
UCSF BMI203: Biocomputing Algorithms
Author: HW assignment by Andrew Blair, completed by Jonathan Borowsky
Date: 2/24/23
Program: BMI/biophysics
Description: hidden markov models/viterbi algorithm homework
"""

import numpy as np
import src.hmm
from src.hmm import HiddenMarkovModel
from src.decoders import ViterbiAlgorithm

pathprefix=""

def test_deliverable():
    """compare implementations
    """
    # index annotation observation_states=[i,j] 
    observation_states = ['regulatory', 'regulatory-potential'] # observed regulatory activity in the TBX5 TAD of cardiomyocytes
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['encode-atac', 'atac'] # In order of the two cCRE selection strategies (encode_atac, atac)

    # Import the HMM input data for progenitor cardiomyocytes (prefix: prog_cm)
    prog_cm_data = np.load(f'{pathprefix}./data/ProjectDeliverable-ProgenitorCMs.npz')

    # Instantiate submodule class models.HiddenMarkovModel with progenitor cardiomyocytes
    # observation and hidden states and prior, transition, and emission probabilities.
    prog_cm_hmm_object = HiddenMarkovModel(observation_states,
                                           hidden_states,
                                           prog_cm_data['prior_probabilities'], #  prior probabilities of hidden states in the order specified in the hidden_states list
                                           prog_cm_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                                           prog_cm_data['emission_probabilities'])  # emission_probabilities[hidden_states[i],:][:,observation_states[j]]

    # Instantiate submodule class models.ViterbiAlgorithm with the progenitor cardiomyocyte's HMM
    prog_cm_viterbi_instance = ViterbiAlgorithm(prog_cm_hmm_object)

    # Decode the hidden states (i.e., CRE selection strategy) for the progenitor CMs and evaluate the model performace
    evaluate_viterbi_decoder_using_observation_states_of_prog_cm = prog_cm_viterbi_instance.best_hidden_state_sequence(prog_cm_data['observation_states'])

    #Rabiner's implementation (accuracy 0.8) does a better job of predicting TBX5 TAD regulatory activity from cCRE
    # selection strategy than this implementation (accuracy 0.6).
    expected_accuracy = 0.6

    # Evaluate the accuracy of using the progenitor cardiomyocyte HMM and Viterbi algorithm to decode the progenitor CM's CRE selection strategies
    # NOTE: Model is expected to perform with 80% accuracy
    assert np.sum(prog_cm_data['hidden_states'] == evaluate_viterbi_decoder_using_observation_states_of_prog_cm)/len(prog_cm_data['observation_states']) == expected_accuracy, \
        f"the viterbi algorithm suggests that either the TBX5 TAD regulatory activity does not predict the cCRE selection strategy much better than chance " \
        f"({np.sum(prog_cm_data['hidden_states'] == evaluate_viterbi_decoder_using_observation_states_of_prog_cm)/len(prog_cm_data['observation_states'])} vs 0.5) " \
        f"or this hmm implementation is not the best way of extracting this information."

    ### Evaluate Primitive Cardiomyocyte Regulatory Observation Sequence ###
    # Import primitive cardiomyocyte data (prefix: prim_cm)
    prim_cm_data = np.load(f'{pathprefix}./data/ProjectDeliverable-PrimitiveCMs.npz')

    # Instantiate submodule class models.ViterbiAlgorithm with the progenitor cardiomyocyte's HMM
    prim_cm_viterbi_instance = ViterbiAlgorithm(prog_cm_hmm_object)

    # Decode the hidden states of the primitive cardiomyocyte's regulatory observation states
    decoded_hidden_states_for_observed_states_of_prim_cm = prim_cm_viterbi_instance.best_hidden_state_sequence(prim_cm_data['observation_states'])
    assert np.sum(prim_cm_data['hidden_states'] == decoded_hidden_states_for_observed_states_of_prim_cm)/len(prim_cm_data['observation_states']) == expected_accuracy, \
        f"the viterbi algorithm suggests that either the TBX5 TAD regulatory activity does not predict the cCRE selection strategy much better than chance " \
        f"({np.sum(prog_cm_data['hidden_states'] == evaluate_viterbi_decoder_using_observation_states_of_prog_cm) / len(prog_cm_data['observation_states'])} vs 0.5) " \
        f"or this hmm implementation is not the best way of extracting this information."