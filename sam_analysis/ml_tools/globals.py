"""
globals.py
This script contains global variables for machine learning applications

CREATED ON: 10/06/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
### IMPORT MODULES
##############################################################################
import numpy as np

##############################################################################
### GLOBAL INPUTS
##############################################################################
## SIMULATION GROUPS
GROUP_LIST = [ 
               ## SINGLE COMPONENT
               "CH3", 
               "NH2", 
               "CONH2", 
               "OH",
               ## NH2 CHARGE SCALED
               "CS0.0NH2", "CS0.1NH2", "CS0.2NH2", "CS0.3NH2", "CS0.4NH2",
               "CS0.5NH2", "CS0.6NH2", "CS0.7NH2", "CS0.8NH2", "CS0.9NH2",
               ## CONH2 CHARGE SCALED
               "CS0.0CONH2", "CS0.1CONH2", "CS0.2CONH2", "CS0.3CONH2", "CS0.4CONH2",
               "CS0.5CONH2", "CS0.6CONH2", "CS0.7CONH2", "CS0.8CONH2", "CS0.9CONH2",
               ## OH CHARGE SCALED
               "CS0.0OH", "CS0.1OH", "CS0.2OH", "CS0.3OH", "CS0.4OH",
               "CS0.5OH", "CS0.6OH", "CS0.7OH", "CS0.8OH", "CS0.9OH",
               ## NH2 MIXED COMPOSITION
               "MIX25NH2", "MIX40NH2", "MIX50NH2", "MIX75NH2",
               ## CONH2 MIXED COMPOSITION
               "MIX25CONH2", "MIX40CONH2", "MIX50CONH2", "MIX75CONH2",
               # OH MIXED COMPOSITION
               "MIX25OH", "MIX40OH", "MIX50OH", "MIX75OH",
               ## NH2 SEPARATED COMPOSITION
               "SEP25NH2", "SEP40NH2", "SEP50NH2", "SEP75NH2",
               ## CONH2 SEPARATED COMPOSITION
               "SEP25CONH2", "SEP40CONH2", "SEP50CONH2", "SEP75CONH2",
               # OH SEPARATED COMPOSITION
               "SEP25OH", "SEP40OH", "SEP50OH", "SEP75OH",
               ]

## DATA TO ADD
X_DATA_LIST = [ 
                "water_water_water_distribution",
                "oh_bond_angle_distribution",
                "hbonds_total",
                "hbonds_sam_sam",
                "hbonds_sam_water_per_water",
                "hbonds_water_water",
                "hbonds_dist_total",
                "hbonds_dist_sam_sam",
                "hbonds_dist_sam_water_per_water",
                "hbonds_dist_water_water",
                "embedded_waters",
                ]                        

## READABLE LABELS FOR DATA
DATA_LABELS = [ "theta_{:d}".format(ii) for ii in np.arange( 0, 180, 2 ) ]            + \
              [ "phi_{:d}".format(ii) for ii in np.arange( 5, 185, 10 ) ]             + \
              [ "num_hbonds_all" ]                                                    + \
              [ "num_hbonds_sam_sam" ]                                                + \
              [ "num_hbonds_sam_water" ]                                              + \
              [ "num_hbonds_water_water" ]                                            + \
              [ "hbond_all_{:d}".format(ii) for ii in np.arange( 0, 10, 1 ) ]         + \
              [ "hbond_sam_sam_{:d}".format(ii) for ii in np.arange( 0, 10, 1 ) ]     + \
              [ "hbond_sam_water_{:d}".format(ii) for ii in np.arange( 0, 10, 1 ) ]   + \
              [ "hbond_water_water_{:d}".format(ii) for ii in np.arange( 0, 10, 1 ) ] + \
              [ "embedded_waters" ]

## GROUP INDICES
NH2_GROUPS       = np.array([  0,  1,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 34, 35, 36, 37, 46, 47, 48, 49 ])
NH2_CS_GROUPS    = np.array([  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  1 ])
NH2_MIX_GROUPS   = np.array([  0, 34, 35, 36, 37,  1 ])
NH2_SEP_GROUPS   = np.array([  0, 46, 47, 48, 49,  1 ])
CONH2_GROUPS     = np.array([  0,  2, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 38, 39, 40, 41, 50, 51, 52, 53 ])
CONH2_CS_GROUPS  = np.array([ 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  2 ])
CONH2_MIX_GROUPS = np.array([  0, 38, 39, 40, 41,  2 ])
CONH2_SEP_GROUPS = np.array([  0, 50, 51, 52, 53,  2 ])
OH_GROUPS        = np.array([  0,  3, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 42, 43, 44, 45, 54, 55, 56, 57 ])
OH_CS_GROUPS     = np.array([ 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,  3 ])
OH_MIX_GROUPS    = np.array([  0, 42, 43, 44, 45,  3 ])
OH_SEP_GROUPS    = np.array([  0, 54, 55, 56, 57,  3 ])

## CONSTANT TRAINING AND TEST CROSS-VALIDATION SETS
RANDOM_SET = np.array([ 44, 30, 19, 51, 33, 31, 39, 10, 20, 28,  4, 16,  0, 56, 48, 32, 36,  
                         7, 45,  6, 15, 38,  9, 27, 24,  8,  2, 42, 43,  1, 55, 41, 46, 18, 
                        54, 22, 14, 21, 13, 29, 52, 40,  3, 17, 49, 11, 34,  5, 37, 26, 23, 
                        25, 57, 12, 53, 35, 50, 47 ])
                        
TRAINING_GROUPS = [ np.array([  0, 56, 48, 32, 36,  7, 45,  6, 15, 38,  9, 27, 24,  8,  2, 42, 43,
                                1, 55, 41, 46, 18, 54, 22, 14, 21, 13, 29, 52, 40,  3, 17, 49, 11,
                               34,  5, 37, 26, 23, 25, 57, 12, 53, 35, 50, 47 ]),
                    np.array([ 44, 30, 19, 51, 33, 31, 39, 10, 20, 28,  4, 16, 24,  8,  2, 42, 43,
                                1, 55, 41, 46, 18, 54, 22, 14, 21, 13, 29, 52, 40,  3, 17, 49, 11,
                               34,  5, 37, 26, 23, 25, 57, 12, 53, 35, 50, 47 ]),
                    np.array([ 44, 30, 19, 51, 33, 31, 39, 10, 20, 28,  4, 16,  0, 56, 48, 32, 36,
                                7, 45,  6, 15, 38,  9, 27, 14, 21, 13, 29, 52, 40,  3, 17, 49, 11,
                               34,  5, 37, 26, 23, 25, 57, 12, 53, 35, 50, 47 ]),
                    np.array([ 44, 30, 19, 51, 33, 31, 39, 10, 20, 28,  4, 16,  0, 56, 48, 32, 36,
                                7, 45,  6, 15, 38,  9, 27, 24,  8,  2, 42, 43,  1, 55, 41, 46, 18,
                               54, 22,  5, 37, 26, 23, 25, 57, 12, 53, 35, 50, 47 ]),
                    np.array([ 44, 30, 19, 51, 33, 31, 39, 10, 20, 28,  4, 16,  0, 56, 48, 32, 36,
                                7, 45,  6, 15, 38,  9, 27, 24,  8,  2, 42, 43,  1, 55, 41, 46, 18,
                               54, 22, 14, 21, 13, 29, 52, 40,  3, 17, 49, 11, 34 ]), ]

TESTING_GROUPS = [ np.array([ 44, 30, 19, 51, 33, 31, 39, 10, 20, 28,  4, 16 ]),
                   np.array([  0, 56, 48, 32, 36,  7, 45,  6, 15, 38,  9, 27 ]),
                   np.array([ 24,  8,  2, 42, 43,  1, 55, 41, 46, 18, 54, 22 ]),
                   np.array([ 14, 21, 13, 29, 52, 40,  3, 17, 49, 11, 34 ]),
                   np.array([  5, 37, 26, 23, 25, 57, 12, 53, 35, 50, 47 ]), ]

## NH2 TRAINING AND TEST SETS
NH2_TRAINING_GROUPS = [ np.array([ 36,  9, 10,  8, 11, 49, 48,  1, 37, 47,  6,  4, 13, 46,  0 ]),
                        np.array([ 36,  9, 10,  8, 11, 49, 48,  1, 37, 47,  7,  5, 35, 12, 34 ]),
                        np.array([ 36,  9, 10,  8, 11,  6,  4, 13, 46,  0,  7,  5, 35, 12, 34 ]),
                        np.array([ 49, 48,  1, 37, 47,  6,  4, 13, 46,  0,  7,  5, 35, 12, 34 ]), ]

NH2_TESTING_GROUPS = [ np.array([  7,  5, 35, 12, 34 ]),
                       np.array([  6,  4, 13, 46,  0 ]),
                       np.array([ 49, 48,  1, 37, 47 ]),
                       np.array([ 36,  9, 10,  8, 11 ]), ]

## CONH2 TRAINING AND TEST SETS
CONH2_TRAINING_GROUPS = [ np.array([  0, 39, 18,  2, 14, 38, 19, 17, 21, 41, 40, 20, 50, 23, 15 ]),
                          np.array([  0, 39, 18,  2, 14, 38, 19, 17, 21, 41, 51, 16, 52, 22, 53 ]),
                          np.array([  0, 39, 18,  2, 14, 40, 20, 50, 23, 15, 51, 16, 52, 22, 53 ]),
                          np.array([ 38, 19, 17, 21, 41, 40, 20, 50, 23, 15, 51, 16, 52, 22, 53 ]), ]

CONH2_TESTING_GROUPS = [ np.array([ 51, 16, 52, 22, 53 ]),
                         np.array([ 40, 20, 50, 23, 15 ]),
                         np.array([ 38, 19, 17, 21, 41 ]),
                         np.array([  0, 39, 18,  2, 14 ]), ]

## OH TRAINING AND TEST SETS
OH_TRAINING_GROUPS = [ np.array([ 29, 26, 42, 33, 45, 31,  0, 56,  3, 32, 30, 43, 54, 28, 55 ]),
                       np.array([ 29, 26, 42, 33, 45, 31,  0, 56,  3, 32, 27, 44, 25, 24, 57 ]),
                       np.array([ 29, 26, 42, 33, 45, 30, 43, 54, 28, 55, 27, 44, 25, 24, 57 ]),
                       np.array([ 31,  0, 56,  3, 32, 30, 43, 54, 28, 55, 27, 44, 25, 24, 57 ]), ]

OH_TESTING_GROUPS = [ np.array([ 27, 44, 25, 24, 57 ]),
                      np.array([ 30, 43, 54, 28, 55 ]),
                      np.array([ 31,  0, 56,  3, 32 ]),
                      np.array([ 29, 26, 42, 33, 45 ]), ]
