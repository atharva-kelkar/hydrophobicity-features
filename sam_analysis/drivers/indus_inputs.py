"""
indus_inputs.py
This script contains the indus analysis tool input parameters

NOTE: to be update each time a different analysis condition is desired
For example, if the temperature should be larger change temperature = 300 to 
temperature = 350.

"""
##############################################################################
## ANALYSIS VARIABLES
##############################################################################
## DICTIONARY OBJECT CONTAINING UNBIASED ANALYSIS ATTRIBUTES
ANALYSIS_ATTR = { 
                  "sim_working_dir"    : None,
                  "input_prefix"       : "sam_indus",
                  "verbose"            : True,
                  ## WHAM/HFE SPECIFIC VARIABLES
                  "temperature"        : 300,
                  ## WHAM
                  "start"              : 2000.,
                  "end"                : -1,
                  "spring_weak"        : 2.0,
                  "spring_strong"      : 8.5,
                  "tol"                : 0.00001,
                  "recompute_wham"     : False,
                  ## HYDRATION FREE ENERGY
                  "recompute_fe"       : False,
                  ## CONVERGENCE/EQUILBRATION TIME SPECIFIC VARIABLES
                  "time_range"         : ( 0., 5000., ),
                  "time_step"          : 200.,
                  ## CONVERGENCE
                  "equilibration_time" : 2000.,
                  "recompute_converge" : True,
                  ## EQUILIBRATION
                  "convergence_time"   : 3000.,
                  "recompute_equil"    : True,
                  }
