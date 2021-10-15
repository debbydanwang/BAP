#%reset -f

import itertools
import numpy as np
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable


def plec_pairing(plec_degrees):
    """
    Pairs ecfp radii of the two molecular fragments in protein and ligand.
    Parameters:
        plec_degrees - ECFP radii for a pair of molecules
    Returns a list of ECFP-radius pair (e.g. [(0, 0), (1, 1), (2, 1), (3, 1)] if ecfp_protein = 1 and ecfp_ligand = 3)
        note the first index indicates the protein and the second the ligand
    """    
    dg1 = min(plec_degrees)
    dg2 = max(plec_degrees)
    pairings = []
    if plec_degrees[1] == plec_degrees[0]:
        for dg in range(plec_degrees[1] + 1):
            pairings.append((dg, dg))
    else:
        for dg in range(dg1):
            pairings.append((dg, dg))
        pairings += list(itertools.product([dg1], np.arange(dg1, dg2 + 1)) if plec_degrees[0] == dg1
                         else itertools.product(np.arange(dg1, dg2 + 1), [dg1]))   
    pairings = [(int(i), int(j)) for (i,j) in pairings]         
    return pairings


def getOriginalIdentifiers(mol, 
                           prop = ['AtomicNumber', 'AtomicMass', 'TotalConnections', 'HCount', 'HeavyNeighborCount', 'FormalCharge', 
                                   'DeltaMass', 'IsTerminalAtom'],
                           includeAtoms = None,
                           radius = 2):
    """Compute the original identifiers for atoms in a molecule based on atomic properties. 
       Note it only includes HEAVY atoms.
    Parameters:
        mol - rdkit.Chem.rdchem.Mol molecule
        prop - atomic property list
               'AtomicNumber': the atomic number of atom
               'AtomicMass': the mass of atom
               'TotalConnections': the degree of the atom in the molecule including Hs
               'HeavyNeighborCount': the number of heavy (non-hydrogen) neighbor atoms
               'HCount': the number of attached hydrogens (both implicit and explicit)
               'FormalCharge': the formal charge of atom
               'DeltaMass': the difference between atomic mass and atomic weight (weighted average of atomic masses)
               'IsTerminalAtom': indicates whether the atom is a terminal atom
        includeAtoms - atom indices for getting identifiers
        radius - ECFP radius, only calculates the identifiers of atoms in the neighborhoods (of radius) of included atoms (includeAtoms)
    Returns an dictionary mapping each heavy-atom index to an integer representing the atomic properties
    """
    tbl = GetPeriodicTable()
    idf_dict = {}
    nAtoms = mol.GetNumAtoms()
    if includeAtoms is None:
        indices = range(nAtoms)
    else:
        indices = includeAtoms
    indices = [int(i) for i in set(indices)]
    for index in indices:
        env = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, index, useHs=True))
        if len(env) == 0:
            env = list(Chem.FindAtomEnvironmentOfRadiusN(mol, mol.GetAtomWithIdx(index).GetDegree(), index, useHs=True))        
        env_aids = set([mol.GetBondWithIdx(bid).GetBeginAtomIdx() for bid in env] + [mol.GetBondWithIdx(bid).GetEndAtomIdx() for bid in env]) 
        for aid in env_aids:
            if (aid, 0) not in idf_dict:
                atom = mol.GetAtomWithIdx(aid)        
                if atom.GetAtomicNum() > 1:
                    properties = []
                    if 'AtomicNumber' in prop:
                        properties.append(atom.GetAtomicNum())
                    if 'AtomicMass' in prop:
                        properties.append(atom.GetMass())
                    if 'TotalConnections' in prop:
                        properties.append(atom.GetDegree())
                    if 'HCount' in prop:
                        properties.append(atom.GetNumExplicitHs())
                    if 'HeavyNeighborCount' in prop:
                        properties.append(len([bond.GetOtherAtom(atom) for bond in atom.GetBonds() if bond.GetOtherAtom(atom).GetAtomicNum() > 1]))
                    if 'FormalCharge' in prop:
                        properties.append(atom.GetFormalCharge())
                    if 'DeltaMass' in prop:
                        properties.append(atom.GetMass() - tbl.GetAtomicWeight(atom.GetAtomicNum()))
                    if 'IsTerminalAtom' in prop:
                        is_terminal_atom = 1 if atom.GetDegree() == 1 else 0
                        properties.append(is_terminal_atom)
                    
                    idf_dict[(aid, 0)] = hash(tuple(properties))  

    return idf_dict

def getIdentifiersRadiusN(molinfo, 
                          prop = ['AtomicNumber', 'AtomicMass', 'TotalConnections', 'HCount', 'HeavyNeighborCount', 'FormalCharge', 
                                  'DeltaMass', 'IsTerminalAtom'],
                          includeAtoms = None,
                          radius = 2):
    """Calculate the Identifiers of molecular fragments (each originated from an atom, of radius N) in a molecule.
    Parameters:
        molinfo - a tuple describing a molecule (coordinates, rdkit.Chem.rdchem.Mol molecule, weights), weights = None for non-weighted alpha shapes
        prop, radius and includeAtoms - same as in getOriginalIdentifiers
    Returns the identifiers
    """
    res_idfs = {}
    mol = molinfo[1]
    nAtoms = mol.GetNumAtoms()
    neighborhoods = []
    deadAtoms = [0] * nAtoms
    if includeAtoms is not None:
        selatoms = list(set(includeAtoms))
    else:
        selatoms = range(nAtoms)
    
    # get original identifiers (of radius 0) of included atoms and their neighbors (in neighborhood of radius)
    idf_dict = getOriginalIdentifiers(mol = mol, 
                                      prop = prop,
                                      includeAtoms = selatoms,
                                      radius = radius)
    ids_fil = set([u[0] for u in idf_dict])

    # get atom orders
    if includeAtoms is not None:
        idfs = {u:v for (u,v) in idf_dict.items() if u[0] in selatoms}
        # put the query atoms in front positions (access first)
        atomOrder = selatoms + [i for i in ids_fil if i not in selatoms]
    else:
        idfs = idf_dict
        atomOrder = range(nAtoms)
    # initialize res_idfs
    res_idfs = {k: (v, [-k[0]-1]) for (k, v) in idfs.items()} # when radius is 0, the neighborhood only contains the center atom (use -aid-1 to distinguish from bond ids)
    atomOrder = [int(i) for i in atomOrder]
    
    # iteratively calculate the identifiers of larger radius
    if radius == 0:
        return res_idfs
    else:
        for layer in range(radius):
            round_idfs = {}
            neighborhoodThisRound = []
            for index in atomOrder:
                if not deadAtoms[index]:
                    atom = mol.GetAtomWithIdx(index)
                    if atom.GetAtomicNum() == 1 or atom.GetDegree() == 0:
                        deadAtoms[index] = 1
                        continue

                    env = list(Chem.FindAtomEnvironmentOfRadiusN(mol, layer + 1, index, useHs=True))
                    env.sort()
                    nbrs = []
                    bonds = atom.GetBonds()
                    for bond in bonds:
                        oth_index = bond.GetOtherAtomIdx(index)
                        if (oth_index, layer) in idf_dict:
                            bt = bond.GetBondTypeAsDouble()
                            nbrs.append((bt, idf_dict[(oth_index, layer)]))
                    nbrs.sort()
                    nbrhd = [layer, idf_dict[(index, layer)]]
                    for nbr in nbrs:                        
                        nbrhd.append(nbr)
                    # use [layer, idf, (nbr1_bondtype, nbr1_idf), ..., (nbrN_bondtype, nbrN_idf)] to represent an atomic neighborhood of a specific radius (layer)
                    idf = hash(tuple(nbrhd))
                    
                    round_idfs[(index, layer + 1)] = idf
                    neighborhoodThisRound.append((env, idf, index))
                    if env in neighborhoods:
                        deadAtoms[index] = 1
            
            neighborhoodThisRound.sort()
            for candi in neighborhoodThisRound:
                if candi[0] not in neighborhoods:
                    neighborhoods.append(candi[0])
                    if includeAtoms == None or candi[2] in selatoms:
                        res_idfs[(candi[2], layer + 1)] = (candi[1], candi[0])
                else:
                    deadAtoms[candi[2]] = 1
            idf_dict = round_idfs
        
        return res_idfs
    

#def getIdentifiersRadiusN_all(molinfo, 
#                              prop = ['AtomicNumber', 'AtomicMass', 'TotalConnections', 'HCount', 'HeavyNeighborCount', 'FormalCharge', 
#                                      'DeltaMass', 'IsTerminalAtom'],
#                              includeAtoms = None,
#                              radius = 2):
#    """Calculate the Identifiers of all molecular fragments (each originated from an atom, of radius N, can be redundant) in a molecule.
#    Parameters:
#        molinfo - a tuple describing a molecule (coordinates, rdkit.Chem.rdchem.Mol molecule, weights), weights = None for non-weighted alpha shapes
#        prop, radius and includeAtoms - same as in getOriginalIdentifiers
#    Returns the identifiers
#    """
#    idfs_all = {}
#    mol = molinfo[1]
#    nAtoms = mol.GetNumAtoms()
#    deadAtoms = [0] * nAtoms
#    
#    # get original identifiers (of radius 0) of included atoms and their neighbors (in neighborhood of radius)
#    idf_dict = getOriginalIdentifiers(mol = mol, 
#                                      prop = prop,
#                                      includeAtoms = includeAtoms,
#                                      radius = radius)
#    ids_fil = set([u[0] for u in idf_dict])
#    idfs_all = {k: (v, [k[0]]) for (k, v) in idf_dict.items()}
#
#    # get atom orders
#    if includeAtoms is not None:
#        # put the query atoms in front positions (access first)
#        atomOrder = includeAtoms + [i for i in ids_fil if i not in includeAtoms]
#    else:
#        atomOrder = range(nAtoms)
#    atomOrder = [int(i) for i in atomOrder]
#    
#    # iteratively calculate the identifiers of larger radius
#    if radius == 0:
#        sel = {k:v for (k, v) in idfs_all if k[0] in includeAtoms}
#        return sel
#    else:
#        for layer in range(radius):
#            for index in atomOrder:
#                if not deadAtoms[index]:
#                    atom = mol.GetAtomWithIdx(index)
#                    env = list(Chem.FindAtomEnvironmentOfRadiusN(mol, layer + 1, index, useHs=True))
#                    env.sort()
#                    if atom.GetAtomicNum() == 1 or atom.GetDegree == 0:
#                        deadAtoms[index] = 1
#                        continue
#                    nbrs = []
#                    bonds = atom.GetBonds()
#                    for bond in bonds:
#                        oth_index = bond.GetOtherAtomIdx(index)
#                        if (oth_index, layer) in idfs_all:
#                            bt = bond.GetBondTypeAsDouble()
#                            nbrs.append((bt, idfs_all[(oth_index, layer)][0]))
#                    nbrs.sort()
#                    nbrhd = [layer, idfs_all[(index, layer)][0]]
#                    for nbr in nbrs:                        
#                        nbrhd.append(nbr)
#                    # use [layer, idf, (nbr1_bondtype, nbr1_idf), ..., (nbrN_bondtype, nbrN_idf)] to represent an atomic neighborhood of a specific radius (layer)
#                    idf = hash(tuple(nbrhd))
#                    
#                    if index in includeAtoms:
#                        idfs_all[(index, layer + 1)] = (idf, env)
#        
#        return idfs_all



def getIdentifiersRadiusN_ifp(mols_info, 
                              prop = ['AtomicNumber', 'AtomicMass', 'TotalConnections', 'HCount', 'HeavyNeighborCount', 'FormalCharge', 
                                      'DeltaMass', 'IsTerminalAtom'],
                              contacts = [[], []],
                              degrees = [1, 1],
                              ifptype = 'splif'):
    """Computes SPLIF identifiers for a pair of molecular fragments (e.g. protein-binding pocket and ligand).
    Parameters:
        mols_info - a list of two molecules (coordinates, rdkit.Chem.rdchem.Mol molecule)
        contacts - a list of two sets, each indicating the indices of atoms to be considered in a molecule
        degrees - ecfp radii
        prop - same as above
        ifptype - either 'splif' or 'plec'
    """
    idf_dicts = [{}, {}]
    envlsts = [[], []]
    idflsts = [[], []]
    res_idfs = []
    mols = [mols_info[0][1], mols_info[1][1]]
    nPairs = len(contacts[0])
    if nPairs == 0:
        print('Wrong contact list!')
        return res_idfs
    else:
        neighborhoods = []
        deadAtomPairs = {}        
            
        if ifptype == 'splif':
            dg_pairs = [(degrees[0], degrees[1])]
        elif ifptype == 'plec':            
            dg_pairs = plec_pairing(plec_degrees = degrees)
        else:
            print('Wrong ifp type!')
            return res_idfs
        
        # get original identifiers of included atoms
        for k in [0, 1]:
            idf_dicts[k] = getIdentifiersRadiusN(molinfo = mols_info[k],
                                                     prop = prop,
                                                     includeAtoms = contacts[k],
                                                     radius = degrees[k])     
            envlsts[k] = [i[1] for i in list(idf_dicts[k].values())]
            idflsts[k] = [i[0] for i in list(idf_dicts[k].values())]
        
        for dgs in dg_pairs:
            for (a1, a2) in zip(contacts[0], contacts[1]):
                inds = (int(a1), int(a2))
                if inds not in deadAtomPairs:
                    atoms = (mols[0].GetAtomWithIdx(inds[0]), mols[1].GetAtomWithIdx(inds[1]))
                    sign1 = (atoms[0].GetAtomicNum() == 1 or atoms[1].GetAtomicNum() == 1)
                    sign2 = (atoms[0].GetDegree() == 0 or atoms[1].GetDegree() == 0)
                    if sign1 or sign2:
                        deadAtomPairs[inds] = 1
                        continue
                    envs = [[], []]
                    idfinds = [-1, -1]
                    for m in [0, 1]:
                        env = list(Chem.FindAtomEnvironmentOfRadiusN(mols[m], dgs[m], inds[m], useHs=True)) if dgs[m] > 0 else str(inds[m])
                        if dgs[m] == 0:
                            env = [-inds[m]-1]
                        env.sort()
                        envs[m] = env
                        if env in envlsts[m]:
                            idfinds[m] = envlsts[m].index(env)
                    if idfinds[0] > -1 and idfinds[1] > -1:
                        if envs not in neighborhoods:
                            neighborhoods.append(envs)
                            res_idfs.append(hash(tuple((idflsts[0][idfinds[0]], idflsts[1][idfinds[1]]))))
            
            return res_idfs


def getECFPidentifiers_molpair(mols_info, 
                               prop = ['AtomicNumber', 'AtomicMass', 'TotalConnections', 'HCount', 'HeavyNeighborCount', 'FormalCharge', 
                                       'DeltaMass', 'IsTerminalAtom'],
                               contacts = [[], []],
                               degrees = [1, 1],
                               ifptype = 'splif'):
    """Obtain the integer identifers of molecular fragments.
    """
    idf_list = [[], []]
    idfs = []
    if ifptype in ['ecfp']:
        for i in [0, 1]:
            tmp = getIdentifiersRadiusN(molinfo = mols_info[i],
                                        prop = prop,
                                        includeAtoms = contacts[i],
                                        radius = degrees[i])
            idf_list[i] = [i[0] for i in list(tmp.values())]
        return idf_list
    elif ifptype in ['splif', 'plec']:
        idfs = getIdentifiersRadiusN_ifp(mols_info = mols_info, 
                                         prop = prop,
                                         contacts = contacts,
                                         degrees = degrees,
                                         ifptype = ifptype)
        return idfs
    else:
        print('Wrong ifptype!')
        return []
             

