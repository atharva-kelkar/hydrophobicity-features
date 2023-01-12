"""
unbiased_compile.py
This script compiles unbiased simulation data to be plotted or 
further analyzed

CREATED ON: 12/15/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
## IMPORT OS
import os
## IMPORT NUMPY
import numpy as np

## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl, save_pkl
## IMPORT CHECK SERVER PATH
from sam_analysis.core.check_tools import check_server_path

##############################################################################
## FUNCTIONS
##############################################################################
## PATH TO UNBIASED DIRECTORIES
PATH_UNBIASED = r"/mnt/r/simulations/polar_sams/unbiased"

## LIST CONTAINING DIRECTORY PATHS
## UNBIASED SUBDIRECTORIES
DIRECTORIES = {
                "BULK-250K-TIP4P"  : "bulk_250K_tip4p",
                "BULK-250K-AMOEBA" : "bulk_250K_amoeba",
                "BULK-300K-TIP4P"  : "bulk_300K_tip4p",
                "BULK-IDEAL"       : "bulk_ideal_gas",
                "BULK-LJ"          : "bulk_lennard_jones",
                "CH3"        : "sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                "NH2"        : "sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",
                "CONH2"      : "sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",
                "OH"         : "sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",
                "MIX25NH2"   : "sam_single_12x12_mixed_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",
                "MIX40NH2"   : "sam_single_12x12_mixed_300K_dodecanethiol0.6_C13NH20.4_tip4p_nvt_CHARMM36",
                "MIX50NH2"   : "sam_single_12x12_mixed_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",
                "MIX75NH2"   : "sam_single_12x12_mixed_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",
                "MIX25CONH2" : "sam_single_12x12_mixed_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36",
                "MIX40CONH2" : "sam_single_12x12_mixed_300K_dodecanethiol0.6_C12CONH20.4_tip4p_nvt_CHARMM36",
                "MIX50CONH2" : "sam_single_12x12_mixed_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",
                "MIX75CONH2" : "sam_single_12x12_mixed_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36",
                "MIX25OH"    : "sam_single_12x12_mixed_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",
                "MIX40OH"    : "sam_single_12x12_mixed_300K_dodecanethiol0.6_C13OH0.4_tip4p_nvt_CHARMM36", 
                "MIX50OH"    : "sam_single_12x12_mixed_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",
                "MIX75OH"    : "sam_single_12x12_mixed_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",
                "SEP25NH2"   : "sam_single_12x12_separated_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",
                "SEP40NH2"   : "sam_single_12x12_separated_300K_dodecanethiol0.58_C13NH20.42_tip4p_nvt_CHARMM36",
                "SEP50NH2"   : "sam_single_12x12_separated_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",
                "SEP75NH2"   : "sam_single_12x12_separated_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",
                "SEP25CONH2" : "sam_single_12x12_separated_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36",
                "SEP40CONH2" : "sam_single_12x12_separated_300K_dodecanethiol0.58_C12CONH20.42_tip4p_nvt_CHARMM36",
                "SEP50CONH2" : "sam_single_12x12_separated_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",
                "SEP75CONH2" : "sam_single_12x12_separated_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36",
                "SEP25OH"    : "sam_single_12x12_separated_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",
                "SEP40OH"    : "sam_single_12x12_separated_300K_dodecanethiol0.58_C13OH0.42_tip4p_nvt_CHARMM36",
                "SEP50OH"    : "sam_single_12x12_separated_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",
                "SEP75OH"    : "sam_single_12x12_separated_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",
                "CS0.0NH2"   : "sam_single_12x12_300K_C13NH2_k0.0_tip4p_nvt_CHARMM36",
                "CS0.1NH2"   : "sam_single_12x12_300K_C13NH2_k0.1_tip4p_nvt_CHARMM36",
                "CS0.2NH2"   : "sam_single_12x12_300K_C13NH2_k0.2_tip4p_nvt_CHARMM36",
                "CS0.3NH2"   : "sam_single_12x12_300K_C13NH2_k0.3_tip4p_nvt_CHARMM36",
                "CS0.4NH2"   : "sam_single_12x12_300K_C13NH2_k0.4_tip4p_nvt_CHARMM36",
                "CS0.5NH2"   : "sam_single_12x12_300K_C13NH2_k0.5_tip4p_nvt_CHARMM36",
                "CS0.6NH2"   : "sam_single_12x12_300K_C13NH2_k0.6_tip4p_nvt_CHARMM36",
                "CS0.7NH2"   : "sam_single_12x12_300K_C13NH2_k0.7_tip4p_nvt_CHARMM36",
                "CS0.8NH2"   : "sam_single_12x12_300K_C13NH2_k0.8_tip4p_nvt_CHARMM36",
                "CS0.9NH2"   : "sam_single_12x12_300K_C13NH2_k0.9_tip4p_nvt_CHARMM36",
                "CS0.0CONH2" : "sam_single_12x12_300K_C12CONH2_k0.0_tip4p_nvt_CHARMM36",
                "CS0.1CONH2" : "sam_single_12x12_300K_C12CONH2_k0.1_tip4p_nvt_CHARMM36",
                "CS0.2CONH2" : "sam_single_12x12_300K_C12CONH2_k0.2_tip4p_nvt_CHARMM36",
                "CS0.3CONH2" : "sam_single_12x12_300K_C12CONH2_k0.3_tip4p_nvt_CHARMM36",
                "CS0.4CONH2" : "sam_single_12x12_300K_C12CONH2_k0.4_tip4p_nvt_CHARMM36",
                "CS0.5CONH2" : "sam_single_12x12_300K_C12CONH2_k0.5_tip4p_nvt_CHARMM36",
                "CS0.6CONH2" : "sam_single_12x12_300K_C12CONH2_k0.6_tip4p_nvt_CHARMM36",
                "CS0.7CONH2" : "sam_single_12x12_300K_C12CONH2_k0.7_tip4p_nvt_CHARMM36",
                "CS0.8CONH2" : "sam_single_12x12_300K_C12CONH2_k0.8_tip4p_nvt_CHARMM36",
                "CS0.9CONH2" : "sam_single_12x12_300K_C12CONH2_k0.9_tip4p_nvt_CHARMM36",
                "CS0.0OH"    : "sam_single_12x12_300K_C13OH_k0.0_tip4p_nvt_CHARMM36",
                "CS0.1OH"    : "sam_single_12x12_300K_C13OH_k0.1_tip4p_nvt_CHARMM36",
                "CS0.2OH"    : "sam_single_12x12_300K_C13OH_k0.2_tip4p_nvt_CHARMM36",
                "CS0.3OH"    : "sam_single_12x12_300K_C13OH_k0.3_tip4p_nvt_CHARMM36",
                "CS0.4OH"    : "sam_single_12x12_300K_C13OH_k0.4_tip4p_nvt_CHARMM36",
                "CS0.5OH"    : "sam_single_12x12_300K_C13OH_k0.5_tip4p_nvt_CHARMM36",
                "CS0.6OH"    : "sam_single_12x12_300K_C13OH_k0.6_tip4p_nvt_CHARMM36",
                "CS0.7OH"    : "sam_single_12x12_300K_C13OH_k0.7_tip4p_nvt_CHARMM36",
                "CS0.8OH"    : "sam_single_12x12_300K_C13OH_k0.8_tip4p_nvt_CHARMM36",
                "CS0.9OH"    : "sam_single_12x12_300K_C13OH_k0.9_tip4p_nvt_CHARMM36",
                }

## LIST CONTAINING UNBIASED SAMPLE DIRECTORIES
PATH_SAMPLES = [ 
                 "sample1", 
                 "sample2", 
                 "sample3" 
                 ]

## ANALYSIS KEYS
ANALYSIS_KEYS = [ 
                  "wc_grid",
                  "wc_fluctuation",
                  "height_instantaneous",
                  "density_z",
                  "water_density",
                  "triplet_angle_theta",
                  "water_water_water_distribution",
                  "orientation_phi",
                  "oh_bond_angle_distribution",
                  "hbonds_sam_sam",
                  "hbonds_sam_water_per_water",
                  "hbonds_water_water",
                  "hbonds_total",
                  "hbonds_dist_n",
                  "hbonds_dist_sam_sam",
                  "hbonds_dist_sam_water_per_water",
                  "hbonds_dist_water_water",
                  "hbonds_dist_total",
                  "embedded_waters",
                  "coord",
                  "coord_dist",
                  "coord_target_dist",
                  "coord_target_dist_water",
                  "coord_target_dist_sam",
                  "triplet_coord_5",
                  "triplet_coord_6",
                  "theta90_x",
                  "theta90_dist",
                  ]

#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## DATAFILE STRUCTURE
    """ DICTIONARY
    First level is simulation type (e.g., CH3, MIX40NH2, etc)
    Second level is datatype (e.g., density_profile, triplet_angle_distribution, etc)
    Third level is independent samples (e.g., sample1, etc) 
    """
    unbiased_data = {}
    
    ## RESULTS PKL NAME
    data_pkl = r"unbiased_results_data.pkl"
    
    ## OUT PATH
    out_path = r"/home/bdallin/python_projects/sam_analysis/sam_analysis/raw_data"
    
    ## LOOP THROUGH DIRECTORIES
    for sim_key, sim_dir in DIRECTORIES.items():
        ## CREATE SAMPLE PLACEHOLDER
        samples_data = {}
        
        ## LOOP THROUGH SAMPLES
        for ii, sample in enumerate( PATH_SAMPLES ):
            ## PATH TO ANALYSIS DATA
            path_data = os.path.join( PATH_UNBIASED, sim_dir, sample, "output_files", data_pkl )
            path_data = check_server_path( path_data )
            
            ## LOAD DATA IF EXISTS
            if os.path.exists( path_data ) is True:
                data_obj = load_pkl( path_data )
                
                ## LOOP THROUGH DATA TYPES
                for data_key in ANALYSIS_KEYS:
                    ## CHECK IF KEY IN DATA FILE
                    if data_key in data_obj.__dict__.keys():
                        ## STORE DATA 
                        add_data = data_obj.__dict__[data_key]
 
                        ## CONVERT SCALARS TO ARRAYS
                        if type(add_data) is not np.ndarray:
                            add_data = np.array([ add_data ])
 
                        ## IF FIRST SAMPLE CREATE OBJECT
                        if ii < 1:
                            samples_data[data_key] = add_data[np.newaxis,:]
                        else:
                            samples_data[data_key] = np.concatenate(( samples_data[data_key], add_data[np.newaxis,:] ), axis = 0 )
        
        ## UPDATE UNBIASED DATA
        unbiased_data[sim_key] = samples_data
                            
    ## SAVE DATA
    path_pkl = os.path.join( out_path, "unbiased_data.pkl" )
    path_pkl = check_server_path( path_pkl )
    save_pkl( unbiased_data, path_pkl )
           
