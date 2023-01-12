"""
indus_builders.py
This script contains the class builder which constructs the analysis tool object
when adding an new tool must add a builder object for that tool. Follow the example
of one of the classed under analysis classes

CREATED ON: 12/10/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################


##############################################################################
## ANALYSIS FUNCTIONS
##############################################################################
## FUNCTION CREATES DICTIONARY OF BUILDER OBJECTS
def analysis_builder_dict():
    ## DICTIONARY CONTAIN AVAILABLE ANALYSIS TYPES
    return {
             "wham"          : WhamBuilder(),
             "hydration_fe"  : HydrationFreeEnergyBuilder(),
             "convergence"   : ConvergenceBuilder(),
             "equilibration" : EquilibrationBuilder(),
             }

## FUNCTION TO RUN BUILDERS
def analysis_builder( analysis_type ):
    """Provided the analysis type, this function provides the builder"""
    # TO DO: create method to load class if pkl already exists
    analysis_dict = analysis_builder_dict()
    if analysis_type in analysis_dict.keys():
        return analysis_dict[analysis_type]
    else:
        print( "{} not in available analysis types".format( analysis_type ) )
        
##############################################################################
## BUILDER CLASSES
##############################################################################
class Director():
    """Director"""
    def __init__( self, builder, kwargs ):
        ## INITIALIZE DIRECTOR OBJECT
        self._builder = builder
        ## UPDATE VARIABLES TO CONTAIN ANALYSIS INPUTS
        self._builder.__dict__.update(**kwargs)
        
    def construct_analysis_obj( self ):
        """Method to construct the analysis object"""
        ## CREATE NEW ANALYSIS OBJECT
        self._builder.create_new_analysis_obj()
        ## ADD ANALYSIS TYPE (E.G., DENSITY PROFILE, TRIPLET ANGLE DIST, ETC.)
        self._builder.add_type()
        # ADD RESULTS TO OBJECT
        self._builder.add_results() 
        
    def get_analysis_obj( self ):
        """Function to get the analysis object from the Abstract Builder Class"""
        return self._builder.analysis_obj
        
class Builder():
    """Abstract Builder"""
    def __init__( self ):
        ## SET INITIAL OBJECT TO NONE
        self.analysis_obj = None
        
    def create_new_analysis_obj( self ):
        """Method creates a new analysis object"""
        self.analysis_obj = Analysis()
        
class Analysis():
    """Product"""
    def __init__( self ):
        self.type      = None
        self.results   = None
                
    def __str__( self ):
        return 'ANALYSIS DETAILS\n{}\n\n'.format( self.type )
    
##############################################################################
## ANALYSIS BUILDERS
##############################################################################
## WHAM BUILDER
class WhamBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "INDUS WHAM"
    
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.indus.wham import main
        self.analysis_obj.results = main(**self.__dict__)
        
## HYDRATION FREE ENERGY BUILDER
class HydrationFreeEnergyBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "Hydration Free Energy"
            
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.indus.hydration_fe import main
        self.analysis_obj.results = main(**self.__dict__)
        
## CONVERGENCE TIME BUILDER
class ConvergenceBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "INDUS Convergence Time"
            
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.indus.convergence import main
        self.analysis_obj.results = main(**self.__dict__)

## EQUILIBRATION TIME BUILDER
class EquilibrationBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "INDUS Equilibration Time"
            
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.indus.equilibration import main
        self.analysis_obj.results = main(**self.__dict__)
        