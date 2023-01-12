"""
solvents.py
This script contains the the available solvents

NOTE: add a solvent to be considered by updating the dictionary below. The key
should be the string recognized by GMX and MDTRAJ

"""
##############################################################################
## ANALYSIS VARIABLES
##############################################################################
## DICTIONARY OBJECT CONTAINING POTENTIAL LIGANDS (ONLY CONTAINS LIGANDS USED
## IN THIS PROJECT)
SOLVENTS = { 
             "SOL" : "water",
             "HOH" : "water",
             "MET" : "methanol",
             "CL"  : "chloride",
            }