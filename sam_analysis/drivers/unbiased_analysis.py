"""
unbiased_analysis.py
This is the main script that drives the analysis of the unbaised SAM trajectories

CREATED ON: 12/03/2020

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

## FUNCTION TO SAVE AND LOAD PICKLE FILES
from sam_analysis.core.pickles import load_pkl, save_pkl
## IMPORT CHECK SERVER PATH
from sam_analysis.core.check_tools import check_server_path
## IMPORT FUNCTION TO STORE ANALYSIS VARIABLES
from sam_analysis.drivers.unbiased_inputs import ANALYSIS_ATTR
## IMPORT BUILDER OBJECTS
from sam_analysis.drivers.unbiased_builders import Director, analysis_builder

##############################################################################
## LISTS
##############################################################################
## PATH TO UNBIASED DIRECTORIES
PATH_UNBIASED = r"/mnt/r/simulations/polar_sams/unbiased"
### TESTING DIRECTORY
#PATH_UNBIASED = r"/mnt/r/python_projects/sam_analysis/sam_analysis/testing"

## LIST CONTAINING UNBIASED ANALYSIS TYPES
ANALYSIS_TYPES = [
                   "willard_chandler",
                   "sam_height",
                   "sam_order",
                   "sam_voronoi",
                   "chain_length",
                   "traj_time",
                   "density",
                   "water_orientation",
                   "triplet_angle",
                   "hbonds",
                   "embedded_waters",
                   "coordination",
                   "theta90_position",
                #    "hydration_dynamics",
                #    "hydration_fe_maps",
                   ]

## LIST CONTAINING COMPLETED UNBIASED ANALYSIS TYPES
COMPLETED_TYPES = [
                    "willard_chandler",
                    "sam_height",
                    "sam_order",
                    "sam_voronoi",
                    "chain_length",
                    "traj_time",
                    "density",
                    "water_orientation",
                    "triplet_angle",
                    "hbonds",
                    "embedded_waters",
                    "coordination",
                    "theta90_position",
                    # "hydration_dynamics",
                    # "hydration_fe_maps",
                    ]

## LIST CONTAINING UNBIASED ANALYSIS TYPES TO RERUN
RERUN_TYPES = [
                # "willard_chandler",
                # "sam_height",
                # "sam_order",
                # "sam_voronoi",
                # "chain_length",
                # "traj_time",
                # "density",
                # "water_orientation",
                # "triplet_angle",
                # "hbonds",
                # "embedded_waters",
                # "coordination",
                "theta90_position",
                # "hydration_dynamics",
                # "hydration_fe_maps",
                ]        

## LIST CONTAINING DIRECTORY PATHS
## UNBIASED SUBDIRECTORIES
DIRECTORIES = [
                # "bulk_300K_tip4p",
                # "bulk_250K_tip4p",
                # "bulk_250K_amoeba",
                # "bulk_ideal_gas",
                # "bulk_lennard_jones",
                "sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",
                "sam_single_12x12_mixed_300K_dodecanethiol0.6_C13NH20.4_tip4p_nvt_CHARMM36",
                "sam_single_12x12_mixed_300K_dodecanethiol0.6_C12CONH20.4_tip4p_nvt_CHARMM36",
                "sam_single_12x12_mixed_300K_dodecanethiol0.6_C13OH0.4_tip4p_nvt_CHARMM36", 
                "sam_single_12x12_separated_300K_dodecanethiol0.58_C13NH20.42_tip4p_nvt_CHARMM36",
                "sam_single_12x12_separated_300K_dodecanethiol0.58_C12CONH20.42_tip4p_nvt_CHARMM36",
                "sam_single_12x12_separated_300K_dodecanethiol0.58_C13OH0.42_tip4p_nvt_CHARMM36",
                "sam_single_12x12_mixed_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",
                "sam_single_12x12_mixed_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",
                "sam_single_12x12_mixed_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",
                "sam_single_12x12_mixed_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36",
                "sam_single_12x12_mixed_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",
                "sam_single_12x12_mixed_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36",
                "sam_single_12x12_mixed_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",
                "sam_single_12x12_mixed_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",
                "sam_single_12x12_mixed_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",
                "sam_single_12x12_separated_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",
                "sam_single_12x12_separated_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",
                "sam_single_12x12_separated_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",
                "sam_single_12x12_separated_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36",
                "sam_single_12x12_separated_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",
                "sam_single_12x12_separated_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36",
                "sam_single_12x12_separated_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",
                "sam_single_12x12_separated_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",
                "sam_single_12x12_separated_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13NH2_k0.0_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13NH2_k0.1_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13NH2_k0.2_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13NH2_k0.3_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13NH2_k0.4_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13NH2_k0.5_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13NH2_k0.6_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13NH2_k0.7_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13NH2_k0.8_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13NH2_k0.9_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C12CONH2_k0.0_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C12CONH2_k0.1_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C12CONH2_k0.2_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C12CONH2_k0.3_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C12CONH2_k0.4_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C12CONH2_k0.5_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C12CONH2_k0.6_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C12CONH2_k0.7_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C12CONH2_k0.8_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C12CONH2_k0.9_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13OH_k0.0_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13OH_k0.1_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13OH_k0.2_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13OH_k0.3_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13OH_k0.4_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13OH_k0.5_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13OH_k0.6_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13OH_k0.7_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13OH_k0.8_tip4p_nvt_CHARMM36",
                "sam_single_12x12_300K_C13OH_k0.9_tip4p_nvt_CHARMM36",
                ]
### TESTING DIRECTORIES
#DIRECTORIES = [
#                "ordered_ch3_sam",
#                "ordered_oh_sam",
#                "mixed_oh_sam",
#                "mixed_conh2_sam",
#                ]

## LIST CONTAINING UNBIASED SAMPLE DIRECTORIES
PATH_SAMPLES = [ 
                 "sample1", 
                 "sample2", 
                 "sample3"
                 ]
### TESTING
#PATH_SAMPLES = [ "" ]

## MAIN FUNCTION
def main( analysis_type, analysis_attr ):
    """This function drives the analysis"""
    ## CREATE CONCRETE BUILDER
    builder = analysis_builder( analysis_type )
    
    ## CREATE DIRECTORY
    director = Director( builder, analysis_attr )
    director.construct_analysis_obj()
    analysis_obj = director.get_analysis_obj()
    
    ## RETURN ANALYSIS OBJECT
    return analysis_obj
            
#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## IMPORT ANALYSIS RESULTS CLASS
    from sam_analysis.drivers.analysis_results import AnalysisResults
    
    ## RESULTS PKL NAME
    data_pkl = r"unbiased_results_data.pkl"
        
    ## LOOP THROUGH SAMPLES
    for sample in PATH_SAMPLES:    
        ## LOOP THROUGH DIRECTORIES
        for sim_dir in DIRECTORIES:
            ## UPDATE SIMULATION WORKING DIRECTORY
            wd = os.path.join( PATH_UNBIASED, sim_dir, sample )
            ANALYSIS_ATTR["sim_working_dir"] = check_server_path( wd )
            
            ## PATH TO DATA DIR
            path_pkl = os.path.join( wd, "output_files", data_pkl )
            path_pkl = check_server_path( path_pkl )
            
            ## CHECK IF PKL EXISTS
            if os.path.exists( path_pkl ) is True:
                ## LOAD RESULTS FROM FILE
                results = load_pkl( path_pkl )
            else:
                ## INITIALIZE RESULTS CLASS
                results = AnalysisResults()                
            
            ## LOOP THROUGH ANALYSIS TYPES
            for analysis in ANALYSIS_TYPES:
                ## PRINTING
                print_string = "   RUNNING {} IN ./{}".format( analysis.upper(), os.path.join( sim_dir, sample ) )
                print( "\n\n{}".format( "-" * len(print_string) ) )
                print( print_string )
                print( "{}".format( "-" * len(print_string) ) )
                
                ## RUN ANALYSIS FOR FEATURES NOT COMPLETED OR SELECTED TO RURUN
                if analysis not in COMPLETED_TYPES:
                    ## EXECUTE MAIN SCRIPT
                    results.add_results( main( analysis, ANALYSIS_ATTR ) )
                    
                elif analysis in RERUN_TYPES:
                    ## EXECUTE MAIN SCRIPT
                    results.add_results( main( analysis, ANALYSIS_ATTR ), overwrite = True )
                
                else:
                    ## ALREADY ADDED
                    print( "FEATURES ALREADY EXIST" )
                
            ## SAVE FINAL UPDATED RESULTS
            save_pkl( results, path_pkl )
            
            ## DELETE OBJECT
            del results
            
