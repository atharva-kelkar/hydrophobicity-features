"""
ligands.py
This script contains the the available ligands

NOTE: add a ligand to be considered by updating the dictionary below. The key 
is are strings as seen by GMX and the elements the chemical id

"""
##############################################################################
## ANALYSIS VARIABLES
##############################################################################
## DICTIONARY OBJECT CONTAINING POTENTIAL LIGANDS (ONLY CONTAINS LIGANDS USED
## IN THIS PROJECT)
LIGANDS = { 
            "DOD" : "CH3",
            "TAM" : "NH2",
            "DAD" : "CONH2",
            "TOH" : "OH",
            }

LIGAND_END_GROUPS = {
                      "DOD" : [ "C35", "H36", "H37", "H38" ],
                      "TAM" : [ "N41", "H42", "H43" ],
                      "DAD" : [ "C38", "O39", "N40", "H41", "H42" ],
                      "TOH" : [ "O41", "H42" ],
                      }
