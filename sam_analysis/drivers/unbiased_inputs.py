"""
unbiased_inputs.py
This script contains the unbiased analysis tool input parameters

NOTE: to be update each time a different analysis condition is desired
For example, if the z cutoff distance should be larger change z_cutoff = 0.3 to 
z_cutoff = 0.5.

"""
##############################################################################
## ANALYSIS VARIABLES
##############################################################################
## DICTIONARY OBJECT CONTAINING UNBIASED ANALYSIS ATTRIBUTES
ANALYSIS_ATTR = { 
                  "traj"                     : None,
                  "sim_working_dir"          : None,
                  "input_prefix"             : "sam_prod",
                  "z_ref"                    : "willard_chandler",
                  "z_cutoff"                 : 0.3,
                  "r_cutoff"                 : 0.33,
                  "use_com"                  : True,
                  "n_procs"                  : 12,
                  "iter_size"                : 1000,
                  "periodic"                 : True,
                  "water_residues"           : [ "SOL", "HOH" ],
                  "verbose"                  : True,
                  "print_freq"               : 100,
                  ## WILLARD-CHANDLER INTERFACE SPECIFIC VARIABLES
                  "alpha"                    : 0.24,
                  "contour"                  : 16.0,
                  "mesh"                     : [0.1, 0.1, 0.1],
                  "recompute_interface"      : False,
                  ## ORDER PARAMETER SPECIFIC VARIABLE
                  "recompute_order"          : False,
                  ## SAM HEIGHT SPECIFIC VARIALBE
                  "recompute_height"         : False,
                  ## VORONOI TESSELLATION SPECIFIC VARIABLE
                  "dimensions"               : 2,
                  "recompute_voronoi"        : False,
                  ## DENSITY PROFILE SPECIFIC VARIABLES
                  "z_range"                  : ( -2., 3. ),
                  "z_bin_width"              : 0.005,
                  "recompute_density"        : False,
                  ## WATER ORIENTATION ANGLE SPECIFIC VARIABLES
                  "phi_range"                : ( 0, 180. ),
                  "phi_bin_width"            : 10.,
                  "recompute_orientation"    : False,                  
                  ## TRIPLET ANGLE SPECIFIC VARIABLES
                  "theta_theta_range"        : ( 0., 180. ),
                  "theta_bin_width"          : 2.,
                  "split_traj"               : False,
                  "recompute_triplet_angles" : False,
                  ## HBOND SPECIFIC VARIABLES
                  "hbond_r_cutoff"           : 0.35,
                  "hbond_angle_cutoff"       : 0.523598,
                  "recompute_hbonds"         : False,
                  "recompute_hbond_triplets" : False,
                  ## EMBEDDED WATERS VARIABLES
                  "embedded_ref"             : "sam_height",
                  "recompute_embedded_water" : False,
                  ## COORDINATION NUMBER VARIABLES
                  "target_angle"             : 48,
                  "recompute_coord"          : False,
                  ## PROPERTY POSITION VARIABLES
                  "property_value"           : 90,
                  "recompute_coords"         : True,
                  }
