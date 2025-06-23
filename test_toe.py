import nupack as nu 
import numpy as np 
from seqwalk import design
import math
import sys
import random 
from toeholds import *
from union import *
import pickle 
import multiprocessing as mp
random.seed(69420)

n1 = 100
n2 = 100

n=2000
random_dna_seq = "".join(random.choices("ACGT", k=n))

oligo_length = 150 
barcode_length = 18 
fragment_length = oligo_length - 2 * barcode_length
max_final_c_domain_length = oligo_length - barcode_length
toehold_length = 10
toehold_search_range = 15
identity_threshold = 0.6
args = oligo_length, barcode_length, fragment_length, max_final_c_domain_length, toehold_length, toehold_search_range, identity_threshold


if __name__ == "__main__":
    mp.set_start_method("forkserver") 
    
    # generate toehold library 
    putative_toeholds = generate_putative_toeholds(random_dna_seq, args)
    combinations = toehold_combinations(putative_toeholds, args, n1)
    model =nu.Model(material="dna", celsius=50)
    ncores = mp.cpu_count()
    conc = 1e-8
    data = evaluate_combinations(combinations, model, ncores, conc)
    bestid = rank_combinations_weighted(data)
    toeholds = combinations[bestid]
    with open('_toeholds.pkl','wb') as f:
        pickle.dump(toeholds, f)

    # generate barcode library 
    combination = combinations[1]
    library = list(zip(*combination))[0]
    nu_mat, on_t = nupack_matrix_no_mp(library, model, conc, 1)
    toeholds_seqs = [toeholds[i][0] for i in range(len(toeholds))]
    barcode_sets = generate_putative_barcodes(toeholds_seqs, 5,9,barcode_length, n2)
    barcode_data = evaluate_barcodes(barcode_sets, toeholds_seqs, model, ncores, conc)
    best_barcodeid = rank_combinations_weighted(barcode_data)
    barcode_sets[best_barcodeid]
    with open('_barcodes.pkl','wb') as f:
        pickle.dump(barcode_sets, f)
        
    # generate domains 
    domains =get_domains(toeholds, random_dna_seq, toehold_length)
    with open('_domains.pkl','wb') as f:
        pickle.dump(domains, f)