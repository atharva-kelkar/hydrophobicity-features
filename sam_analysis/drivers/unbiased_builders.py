"""
unbiased_builders.py
This script contains the class builder which constructs the analysis tool object
when adding an new tool must add a builder object for that tool. Follow the example
of one of the classed under analysis classes

CREATED ON: 12/03/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## ANALYSIS FUNCTIONS
##############################################################################
## FUNCTION CREATES DICTIONARY OF BUILDER OBJECTS
def analysis_builder_dict():
    ## DICTIONARY CONTAIN AVAILABLE ANALYSIS TYPES
    return {
             "willard_chandler"  : WillardChandlerBuilder(),
             "sam_height"        : SamHeightBuilder(),
             "sam_order"         : SamOrderBuilder(),
             "sam_voronoi"       : SamVoronoiBuilder(),
             "density"           : DensityBuilder(),
             "water_orientation" : WaterOrientationBuilder(),
             "triplet_angle"     : TripletAngleBuilder(),
             "hbonds"            : HydrogenBondsBuilder(),
             "embedded_waters"   : EmbeddedWatersBuilder(),
             "coordination"      : CoordinationBuilder(),
             "theta90_position"  : Theta90PositionBuilder(),
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
## WILLARD-CHANDLER BUILDER
class WillardChandlerBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "Willard-Chandler Interface Grid"
    
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.water.willard_chandler import main
        self.analysis_obj.results = main(**self.__dict__)
        
## SAM-HEIGHT BUILDER
class SamHeightBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "SAM height"
            
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.sam.sam_height import main
        self.analysis_obj.results = main(**self.__dict__)
        
## SAM ORDER BUILDER
class SamOrderBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "SAM order"
            
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.sam.sam_order import main
        self.analysis_obj.results = main(**self.__dict__)
        
## SAM VORONOI BUILDER
class SamVoronoiBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "SAM voronoi"
            
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.sam.voronoi import main
        self.analysis_obj.results = main(**self.__dict__)
        
## DENSITY BUILDER
class DensityBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "Density"
            
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.water.density import main
        self.analysis_obj.results = main(**self.__dict__)

## WATER ORIENTATION BUILDER
class WaterOrientationBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "Water orientation"
            
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.water.water_orientation import main
        self.analysis_obj.results = main(**self.__dict__)

## TRIPLET ANGLE BUILDER
class TripletAngleBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "Water triplet angle"
            
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.water.triplet_angle import main
        self.analysis_obj.results = main(**self.__dict__)
        
## TRIPLET ANGLE BUILDER
class HydrogenBondsBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "Hydrogen bonds"
            
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.water.h_bond_types import main
        self.analysis_obj.results = main(**self.__dict__)

## EMBEDDED WATERS BUILDER
class EmbeddedWatersBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "Number Embedded Waters"
            
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.water.embedded_water import main
        self.analysis_obj.results = main(**self.__dict__)

## COORDINATION NUMBER BUILDER
class CoordinationBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "Water Coordination Number"
            
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.water.coordination_number import main
        self.analysis_obj.results = main(**self.__dict__)

## THETA 90 POSITION BUILDER
class Theta90PositionBuilder( Builder ):
    """Concrete Builder --> provides parts and tools to work on the parts """        
    ## FUNCTION ADDS ANALYSIS TYPE
    def add_type( self ):
        self.analysis_obj.type = "Water Theta90 Positions"
            
    ## FUNCTION ADDS RESULTS FROM COMPLETED ANALYSIS
    def add_results( self ):
        ## IMPORT ANALYSIS FUNCTION
        from sam_analysis.water.property_position import main
        self.analysis_obj.results = main(**self.__dict__)
