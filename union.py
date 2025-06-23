import numpy as np
import nupack as nu
import multiprocessing as mp
from orthogonal import *
from toeholds import *
from tqdm import tqdm
from functools import partial
from scipy.stats import rankdata
from seqwalk import design
from math import comb
import itertools
import random

def get_substrings_set(seq, k):
    string_set = []
    for i in range(len(seq)-k+1):
        string_set.append(seq[i:i+k])
    return string_set

def get_substrings_set_with_reverse_complement(seq, k):
    string_set = []
    for i in range(len(seq)-k+1):
        string_set.append(seq[i:i+k])
        string_set.append(nu.reverse_complement(seq[i:i+k]))
    return string_set

def toeholds_get_substrings_set_with_rc(toeholds, k):
    string_set = []
    for i in range(len(toeholds)):
        string_set = string_set + get_substrings_set_with_reverse_complement(toeholds[i], k)
    return string_set

def generate_putative_barcodes(toeholds, q, k, len_barcode, num_sets):
    num_barcodes = len(toeholds)
    string_set = toeholds_get_substrings_set_with_rc(toeholds, q)
    barcodes = design.max_size(len_barcode, k, alphabet="ACGT", RCfree=1,prevented_patterns=string_set)
    print(len(barcodes))
    print(num_barcodes)
    if len(barcodes) < num_barcodes:
        raise ValueError(f"q,k choice too stringnet, can only generate {len(barcodes)} possible barcodes")
    
    barcode_sets = []
    
    max_possible_barcodes = comb(len(barcodes),num_barcodes)
    if max_possible_barcodes < num_sets:
        print(f"can only generate {max_possible_barcodes} barcodes")
        for combination in itertools.combinations(barcodes, num_barcodes):
            barcode_sets.append(combination)
    else:
        while len(barcode_sets) < num_sets:
            # Generate a random subset
            random_subset = random.sample(barcodes, num_barcodes)
            
            # Check if we've already generated this subset
            if random_subset not in barcode_sets:
                barcode_sets.append(random_subset)
                
    return barcode_sets

def bar_and_toe_evaluation(data, model, conc):
    library, id, num_bar, num_toe = data
    duplex = 1 # since barcodes inherently form duplexes
    
    # generate evaluation metric data
    a_matrix = alignment_matrix(library, duplex)
    np.fill_diagonal(a_matrix, np.inf)
    nu_mat, on_t = nupack_matrix_no_mp_barcodes(library, model, conc, duplex, num_bar)
    # only calcuate ensemble data for the barcode domains
    off_target_ensemble, on_target_ensemble = ensemble_matrix_no_mp_barcodes(library[:num_bar], model, 
                                                                   duplex, num_bar)
    tm_mat = tm_no_mp_barcodes(library, 50, 100, 1, conc, num_bar)
    
    min_edit_distance = np.min(a_matrix)
    min_ontarget_probability = np.min(on_t)
    max_offtarget_probability = np.max(nu_mat)
    max_on_target_defect = np.max(on_target_ensemble)
    min_off_target_defect = np.min(off_target_ensemble)
    on_indices = [(i, i+1) for i in range(0, num_bar - 1, 2)]
    tm_on = [tm_mat[i, j] for i, j in on_indices]
    tm_off = [tm_mat[i, j] for i in range(num_bar) for j in range(len(tm_mat)) if 
              (j != i+1)]
    min_tmdelta = np.min(tm_on) - np.max(tm_off)
    
    return id, (min_edit_distance, min_ontarget_probability, max_offtarget_probability, 
           max_on_target_defect, min_off_target_defect, min_tmdelta)

def evaluate_barcodes(barcode_sets, toeholds, model, ncores, conc):
    evaluation_func = partial(bar_and_toe_evaluation, 
                            model=model, 
                            conc=conc)
    bars_and_toes = barcode_sets.copy()
    for i in range(len(bars_and_toes)):
        bars_and_toes[i] = bars_and_toes[i] + toeholds
    assert(len(bars_and_toes[0]) != len(barcode_sets[0]))
    results = np.empty(len(barcode_sets), dtype=object)
    tasks = [(bars_and_toes[i], i, len(barcode_sets[i]), len(toeholds)) for i in range(len(bars_and_toes))]
    assert(len(bars_and_toes) == len(barcode_sets))
    with Pool(ncores) as pool:
        for i, result in tqdm(pool.imap(evaluation_func, tasks), total=len(bars_and_toes)):
            results[i] = result
    return results
    
def ensemble_matrix_no_mp_barcodes(library, model, duplex,num_barcodes):
    size = len(library)
    off_target_ensemble = np.ones((size, size)) #high is good so 1 is best; ignore toehold self interactions
    on_target_ensemble = np.zeros(size) # low is good so 0 is best; ignore toehold self interactions

    for i in range(num_barcodes):
        i, row_slice, on_t = ensemble_row_worker((library, model, i, duplex))
        off_target_ensemble[i, i:] = row_slice
        off_target_ensemble[i+1:, i] = row_slice[1:]  #fill lower triangle by symmetry
        on_target_ensemble[i] = on_t

    return off_target_ensemble, on_target_ensemble

def nupack_matrix_no_mp_barcodes(library, model, conc, duplex, num_barcodes):
    size = len(library)
    np_probs = np.zeros((size, size)) # low is good so 0 is best; ignore toehold self interactions
    on_probs = np.ones(size) # high is good so 1 is best; ignore toehold self interactions

    for i in range(num_barcodes):
        i, row_slice, on_t = row_worker((library, model, conc, i, duplex))
        np_probs[i, i:] = row_slice
        np_probs[i+1:, i] = row_slice[1:]  # fill lower triangle by symmetry
        on_probs[i] = on_t
    return np_probs, on_probs

def tm_no_mp_barcodes(library, low, high, grain, conc, num_barcodes):
    size = len(library)
    tm_mat = np.zeros((size, size)) # just set all toehold self tm to 0
    for i in range(num_barcodes):
        i, row_slice = row_tm_worker((i, library, low, high, grain, conc))
        tm_mat[i, i:] = row_slice
        tm_mat[i+1:, i] = row_slice[1:]  # mirror lower triangle

    return tm_mat

def get_domains(toeholds, full_seq, toehold_length):
    num_domains = len(toeholds) + 1
    domains = np.empty(num_domains, dtype=object)
    domains[0] = full_seq[:toeholds[0][1]]
    for i in range(0, len(toeholds)-1):
        domains[i+1] = full_seq[(toeholds[i][1]+toehold_length):toeholds[i+1][1]]
    domains[num_domains - 1] = full_seq[(toeholds[len(toeholds)-1][1]+toehold_length):]
    
    return domains
    
    

