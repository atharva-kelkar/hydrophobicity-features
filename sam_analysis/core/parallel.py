# -*- coding: utf-8 -*-
"""
parallel.py
This script contains the parallelization functions to speed up analysis

CREATED ON: 09/26/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
## IMPORTING OS
import os
## IMPORTING MULTIPROCESSING FOR PARALLIZATION
import multiprocessing as mp

## IMPORTING SPLIT LIST
from sam_analysis.core.misc_tools import split_list

##############################################################################
## FUNCTIONS AND CLASSES
##############################################################################
## ANALYSIS DRIVER FUNCTION
def run_parallel( analysis_obj, iter_obj, n_procs, split_traj = True, verbose = False, append = True ):
    """Chooses best analysis type based on requested num. of processes"""
    ## INITIALIZE PARALLEL CLASS
    par = parallel( analysis_obj, n_procs, verbose = verbose, append = append )
    
    if split_traj is True:
        if os.cpu_count() < n_procs:
            print( " Too many cores requested for your system hardware" )
            print( " Switching to {} cores".format( int( os.cpu_count() / 2. ) ) )
            n_procs = int( os.cpu_count() / 2. )
        if verbose is True:
            ## PRINTING
            print( "Running {} analysis in parallel".format( analysis_obj ) )
            print( "{} cores requested".format( n_procs ) )
            print( "{} cores available".format( os.cpu_count() ) )
            
        ## RUN ANALYSIS IN PARALLEL
        results = par.par_analysis( iter_obj )
        
    else:
        if os.cpu_count() < n_procs:
            print( " Too many cores requested for your system hardware" )
            print( " Switching to {} cores".format( int( os.cpu_count() / 2. ) ) )
            n_procs = int( os.cpu_count() / 2. )
        if verbose is True:
            ## PRINTING
            print( "Running {} analysis in parallel".format( analysis_obj ) )
            print( "{} cores requested".format( n_procs ) )
            print( "{} cores available".format( os.cpu_count() ) )
            
        ## RUN ANALYSIS IN PARALLEL
        results = par.par_analysis_no_split( iter_obj )
        
    ## RETURN RESULTS
    return results     

## PARALLEL CLASS 
class parallel:
    """Parallel class"""
    def __init__( self, analysis_obj, n_procs, verbose = False, append = True ):
        """Initialize class by storing analysis_obj and n_procs"""
        self._method  = analysis_obj.compute
        self._n_procs = n_procs
        self.verbose  = verbose
        self.append   = append
    
    ## PARALLEL IMPLEMENTATION OF ANALYSIS TOOL
    def par_analysis( self, iter_obj ):
        """Runs analysis in parallel. Note: variables should be saved into analysis_obj
        class"""   
        ## SPLIT ITER_OBJ INTO CHUNKS (DOMAIN DECOMPOSITION PARALLELIZATION)
        iter_obj_chunks = split_list( alist = iter_obj, wanted_parts = self._n_procs )
        if self.verbose is True:
            print( "  Object successfully split: {} processes with {} frames".format( self._n_procs, len(iter_obj_chunks[0]) ) )
        ## DELETE ITER_OBJ ONCE SPLIT TO SAVE RAM
        del iter_obj
        ## RUNNING MULTIPROCESSING
        with mp.Pool( processes = self._n_procs ) as pool:
            ## RUNNING FUNCTION
            par_results = pool.map( self._method, iter_obj_chunks )
        
        if self.append is not True:
            ## RETURN RAW RESULTS
            return par_results
        else:
            ## COMBINE RESULTS PARALLEL RESULTS (NPROCS LIST ELEMENTS IN A LIST)
            results = par_results[0]
            for rr in par_results[1:]:
                results += rr
        
        ## RETURN RESULTS
        return results
    
    ## PARALLEL IMPLEMENTATION OF ANALYSIS TOOL
    def par_analysis_no_split( self, iter_obj ):
        """Runs analysis in parallel. Assume iter_obj already split Note: variables should be saved into analysis_obj
        class"""
        ## RUNNING MULTIPROCESSING
        with mp.Pool( processes = self._n_procs ) as pool:
            ## RUNNING FUNCTION
            par_results = pool.map( self._method, iter_obj )
            
        ## COMBINE RESULTS PARALLEL RESULTS (NPROCS LIST ELEMENTS IN A LIST)
        results = par_results[0]
        for rr in par_results[1:]:
            results += rr
        
        ## RETURN RESULTS
        return results