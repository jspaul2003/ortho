# Template duplex pool generator code
import nupack as nu
from seqwalk import design
import multiprocessing
from orthogonal import *

# denote single stranded setting
duplex = 0

#<MODIFIABLE> 0: Library save file
save_file = "single_library.csv"

#<MODIFIABLE> 0: minimal intermediate reporting, 1: full reporting
reporting = 1 

#<MODIFIABLE> Number of cores for parallel processing
ncores = multiprocessing.cpu_count() 

#<MODIFIABLE> Working concentration of strands in molar
conc = 1e-6 

#<MODIFIABLE> Length of sequences
l = 16

#<MODIFIABLE> SSM K parameter 
k = 6

#<MODIFIABLE> character similarity threshold
threshold_SIM = 12 

#<MODIFIABLE> Standard operating temperature of DNA reaction in celsius
rxn_temp = 37 

#<MODIFIABLE> Nupack model, can change salt conditions here
my_model = nu.Model(material='dna', celsius=rxn_temp) 

#<MODIFIABLE> On-target probability threshold
threshold_ON = 0.7

#<MODIFIABLE> Off-target probability threshold
threshold_OFF = 0.1

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver") 
    # in case we use linux in which case the default start method is fork
    # note that fork yields issues due to pickle in multiprocessing
    print("STEP 1/6: Generating SSM Hamiltonian Set\n")
    library = design.max_size(l, k, alphabet="ACGT", RCfree=duplex)
    print_library_size(library)

    print("STEP 2/6: Similarity Optimization\n")
    library = sim_optimization(library, threshold_SIM, reporting, duplex)
    print_library_size(library)
                
    print("STEP 3/6: Generating Thermodynamic Complex Probabilities\n")
    nu_mat, on_t = nupack_matrix_mp(library, my_model, conc, ncores, duplex)

    print("STEP 4/6: ON-Target Optimization\n")
    library, on_t, nu_mat = on_target_optimization(on_t, library, threshold_ON, 
                                                   reporting, nu_mat)
    print_library_size(library)

    print("STEP 5/6: OFF-Target Optimization\n")
    library, on_t, nu_mat = off_target_optimization(nu_mat, library, threshold_OFF, 
                                                    reporting, on_t)
    print_library_size(library)

    print("STEP 6/6: Saving Library\n")
    save_lib(library, save_file)