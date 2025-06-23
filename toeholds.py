import numpy as np
import csv
import nupack as nu
import multiprocessing as mp
from orthogonal import *
import itertools
import random
from tqdm import tqdm
from functools import partial
from scipy.stats import rankdata

def generate_putative_toeholds(seq, args):
    oligo_length, barcode_length, fragment_length, max_final_c_domain_length, toehold_length, toehold_search_range, identity_threshold = args
    putative_toeholds = []
    position = fragment_length - toehold_length 
    while position < len(seq):
        toehold_subcandidates = []
        # for each possible toehold position, add the toehold to the list
        for i in range(toehold_search_range):
            toehold_subcandidates.append((seq[position:position+toehold_length].upper(),position))
            position += 1 
        putative_toeholds.append(toehold_subcandidates)
        # if the remaining sequence is less than the max final c domain length, break
        if len(seq) - position <= max_final_c_domain_length:
            break
        position = position - toehold_search_range + fragment_length - toehold_length
    return putative_toeholds

def toehold_combinations(putative_toeholds, args, how_many):
    oligo_length, barcode_length, fragment_length, max_final_c_domain_length, toehold_length, toehold_search_range, identity_threshold = args
    
    # if how_many = 0, we evaluate all toehold combinations 
    # else we evaluate how_many random toehold combinations
    if how_many == 0:        
        toehold_combinations = list(itertools.product(*putative_toeholds))
    else:
        toehold_combinations = np.empty(how_many, dtype=object)
        
        for i in range(how_many):
            # Pick one random tuple from each sublist
            random_combination = tuple(random.choice(sublist) for sublist in putative_toeholds)
            toehold_combinations[i] = random_combination
        
    return toehold_combinations


def evaluation_combination(data, model, conc):
    library, id = data
    duplex = 1 # since toeholds inherently form duplexes
    
    # generate evaluation metric data
    a_matrix = alignment_matrix(library, duplex)
    np.fill_diagonal(a_matrix, np.inf)
    nu_mat, on_t = nupack_matrix_no_mp(library, model, conc, duplex)
    off_target_ensemble, on_target_ensemble = ensemble_matrix_no_mp(library, model, 
                                                                   duplex)
    tm_mat = tm_no_mp(library, 50, 100, 1, conc)
    
    min_edit_distance = np.min(a_matrix)
    min_ontarget_probability = np.min(on_t)
    max_offtarget_probability = np.max(nu_mat)
    max_on_target_defect = np.max(on_target_ensemble)
    min_off_target_defect = np.min(off_target_ensemble)
    on_indices = [(i, i+1) for i in range(0, len(tm_mat) - 1, 2)]
    tm_on = [tm_mat[i, j] for i, j in on_indices]
    tm_off = [tm_mat[i, j] for i in range(len(tm_mat)) for j in range(len(tm_mat)) if j != i+1]
    min_tmdelta = np.min(tm_on) - np.max(tm_off)
    
    return id, (min_edit_distance, min_ontarget_probability, max_offtarget_probability, 
           max_on_target_defect, min_off_target_defect, min_tmdelta)

def evaluate_combinations(combinations, model, ncores, conc):
    evaluation_func = partial(evaluation_combination, 
                            model=model, 
                            conc=conc)
    results = np.empty(len(combinations), dtype=object)
    tasks = [(list(zip(*combinations[i]))[0], i) for i in range(len(combinations))]
    with Pool(ncores) as pool:
        for i, result in tqdm(pool.imap(evaluation_func, tasks), total=len(combinations)):
            results[i] = result
    return results

def rank_combinations_weighted(results, weights=None):
    
    if weights is None:
        # Defaul to equal weights
        weights = {
            'edit_distance': 1.0,
            'ontarget_prob': 1.0, 
            'offtarget_prob': 1.0,
            'on_target_defect': 1.0,
            'off_target_defect': 1.0,
            'tmdelta': 1.0
        }
    
    if len(results) == 0:
        return None, {}
    
    # Extract metrics from results
    valid_results = [(i, metrics) for i, metrics in enumerate(results)]
    indices, metrics_list = zip(*valid_results)
    indices = list(indices)
    metrics_array = np.array(metrics_list)
    
    # Extract individual metrics
    edit_distances = metrics_array[:, 0]
    min_ontarget_probs = metrics_array[:, 1] 
    max_offtarget_probs = metrics_array[:, 2]
    max_on_target_defects = metrics_array[:, 3]
    min_off_target_defects = metrics_array[:, 4]
    min_tmdeltas = metrics_array[:, 5]
    
    # Rank each metric
    edit_distance_ranks = rankdata(-edit_distances, method='min')  #higher is better so negate
    ontarget_prob_ranks = rankdata(-min_ontarget_probs, method='min')  #higher is better
    offtarget_prob_ranks = rankdata(max_offtarget_probs, method='min')  #lower is better
    on_target_defect_ranks = rankdata(max_on_target_defects, method='min')  #lower is better  
    off_target_defect_ranks = rankdata(-min_off_target_defects, method='min')  #higher is better
    tmdelta_ranks = rankdata(-min_tmdeltas, method='min')  #higher is better
    
    # Calculate weighted average rank
    weighted_ranks = (
        edit_distance_ranks * weights['edit_distance'] +
        ontarget_prob_ranks * weights['ontarget_prob'] +
        offtarget_prob_ranks * weights['offtarget_prob'] +
        on_target_defect_ranks * weights['on_target_defect'] +
        off_target_defect_ranks * weights['off_target_defect'] +
        tmdelta_ranks * weights['tmdelta']
    ) / sum(weights.values())
    
    # Find best combination
    best_idx = np.argmin(weighted_ranks)
    best_combination_id = indices[best_idx]
    
    print(best_combination_id)
    # print individual ranks
    print("edit_distance_ranks")
    print(edit_distance_ranks[best_combination_id])
    print("ontarget_prob_ranks")
    print(ontarget_prob_ranks[best_combination_id])
    print("offtarget_prob_ranks")
    print(offtarget_prob_ranks[best_combination_id])
    print("on_target_defect_ranks")
    print(on_target_defect_ranks[best_combination_id])
    print("off_target_defect_ranks")
    print(off_target_defect_ranks[best_combination_id]) 
    print("tmdelta_ranks")
    print(tmdelta_ranks[best_combination_id])
    print("raw data")
    print(results[best_combination_id])
    
    return best_combination_id    
    
    
                
                
                
    