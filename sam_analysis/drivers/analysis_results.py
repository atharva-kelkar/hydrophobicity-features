"""
analysis_results.py
This script contains the analysis results object structure.

CREATED ON: 12/14/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## CLASS
##############################################################################
## ANALYSIS CLASS OBJECT
class AnalysisResults:
    """Class compiles all unbiased analysis results into a single class object"""
    def __str__( self ):
        return "Unbiased analysis results"
    
    ## ADD RESULTS INSTANCE
    def add_results( self, obj, overwrite = False ):
        """Instance updates class with new results. Incoming results must be 
        stored as a dictionary under the results attribute"""
        ## PRINTING
        print( "Updating class! {} added\n".format( obj.type ) )
#        print( obj.results )
        ## CREATE PLACEHOLDERS
        to_add    = {}
        not_added = []
        
        ## LOOP THROUGH NEW RESULTS AND CHECK THAT NOT OVERWRITING CURRENT RESULTS
        for rr, vv in obj.results.items():
            ## CHECK IF IN CURRENT RESULTS
            if rr not in self.__dict__.keys() \
             or overwrite is True:
                ## UPDATE TO ADD
                to_add[rr] = vv
            else:
                not_added.append( rr )
        
        ## UPDATE CLASS RESULTS
        self.__dict__.update( **to_add )
                
        ## PRINT VALUES NOT ADD
        if len(not_added) > 0:
            print( "Results not added! {} already exist in class.\n".format( not_added ) )
            