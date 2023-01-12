"""
check_tools.py
script contains methods used to check that paths/files/data exists

CREATED ON: 09/25/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
import os, sys

##############################################################################
## FUNCTIONS
##############################################################################
## FUNCTION TO CHECK IF YOU ARE RUNNING ON SPYDER
def check_spyder():
    ''' This function checks if you are running on spyder '''
    if any('SPYDER' in name for name in os.environ):
        return True
    else:
        return False

## FUNCTION TO CHECK IF SERVER PATH
def check_server_path(path):
    '''
    The purpose of this function is to change the path of analysis based on the current operating system. 
    INPUTS:
        path: [str] Path to analysis directory
    OUTPUTS:
        path: [str] Corrected path (if necessary)
    '''
    ## IMPORTING MODULES
    import getpass

    ## CHECKING THE USER NAME
    user_name = getpass.getuser() # Outputs $USER, e.g. akchew, or bdallin
    
    ## CHANGING BACK SLASHES TO FORWARD SLASHES
    backSlash2Forward = path.replace('\\','/')
    
    ## CHECK OS OF IN
    in_prefix = None
    # AT SWARM
    if '/usr' in path:
        in_prefix = "/home/" + user_name
    # WINDOWS
    elif "R:" in path:
        in_prefix = "R:"
    # WSL
    elif "/mnt/r" in path:
        in_prefix = "/mnt/r"
    # WSL LOCAL
    elif "/home" in path:
        in_prefix = "/home/" + user_name
    # MAC
    elif "/Volumes" in path:
        in_prefix = "/Volumes"
    
    ## CHECK OS OF OUT
    out_prefix = None
    # SWARM
    if '/usr' in sys.prefix:
        out_prefix = "/home/" + user_name
    # WINDOWS
    elif "R:" in sys.prefix:
        out_prefix = "R:"
    # LINUX
    elif "/home" in sys.prefix:
        if "miniconda3" in sys.prefix:
            out_prefix = "/home/" + user_name
        else:
            out_prefix = "/mnt/r"
    # MAC
    elif "/Volumes" in sys.prefix:
        out_prefix = "/Volumes"
        
    ## CHECK THAT IN AND OUT EXIST
    if in_prefix is None or out_prefix is None:
        ## SOMETHING BROKE
        print( "ERROR HERE" )
    else:
        ## UPDATE PATH
        path = backSlash2Forward.replace(in_prefix, out_prefix)
    
    return path

#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## TESTING DIRECTORY
    test_dir = r"mnt/r/python_projects/sam_analysis/sam_analysis/testing"
    
    ## SAM DIRECTORY
    sam_dir = r"mixed_polar_sam"
    
    ## WORKING DIR
    working_dir = os.path.join(test_dir, sam_dir)

    ## CHECK IF SPYDER
    is_spyder = check_spyder()
    print( "IS SPYDER: {}".format( str(is_spyder) ))
    
    ## PATH
    path = check_server_path( working_dir )
    print( "{} -> {}".format( working_dir, path ) )
    