#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:53:50 2021

@author: debbywang
"""

from deepchem.utils.rdkit_utils import load_molecule
from scipy.spatial.distance import cdist
import numpy as np
import pandas
from math import sqrt, acos, pi
#from Bio.PDB import PDBParser
from biopandas.mol2 import PandasMol2
from itertools import combinations
from rdkit.Chem import GetPeriodicTable
from ECFP_util import getECFPidentifiers_molpair
import itertools
from statistics import mean


def get_atmind_withTpReq(mol2df, colname, filter_lst = None, coor = None, req = None):
    """
    Get atom indices that fulfill some requirements.
    Parameters:
        mol2df - a PandasMol dataframe for a molecule
        colname - the column name in mol2df for filtering atoms
        filter_list - a list of atom types or names for filtering atoms in mol2df (None: consider all atoms)
        coor - atom coordinates of a molecule
        req - supplementary requirement for atom filtering
              None: no action
              'poscharge': formal charge >= 0
              'negcharge': formal charge <= 0
    """
    if filter_lst is not None:
        tmp = mol2df.loc[mol2df[colname].isin(filter_lst), ['x', 'y', 'z', 'charge']]
    else:
        tmp = mol2df[['x', 'y', 'z', 'charge']]
    if req is not None:
        tmp2 = tmp.loc[tmp['charge'] >= 0, ['x', 'y', 'z']] if req == 'poscharge' else tmp.loc[tmp['charge'] <= 0, ['x', 'y', 'z']]
    else:
        tmp2 = tmp
    if len(tmp2) > 0:
        tmp_coord = tmp2.apply(lambda row: [float(round(row.x, 3)), float(round(row.y, 3)), float(round(row.z, 3))], axis=1)
        tmp_coord2 = [i for i in tmp_coord if i in coor]
        inds = [coor.index(i) for i in tmp_coord2]
    else:
        inds = []
    return inds

def fil_atmind_withBondReq(inds, mol, req):
    """
    Filter a list of atom indices according to bond requirements.
    Parameters:
        inds - a list of atom indices
        mol - an rdkit.Chem.rdchem.Mol molecule
        req - supplementary bond requirement for atom filtering
              'hydro': no double bond to 'O' and 'N'
              'no_aromatic': no aromatic bond
              'donor': connect to an hydrogen atom
    """
    supp_inds = []
    for ind in inds:
        atm = mol.GetAtomWithIdx(ind)
        for bond in atm.GetBonds():
            other_atm_symb = mol.GetAtomWithIdx(bond.GetOtherAtomIdx(ind)).GetSymbol()
            if req == 'hydro':
                cond = (other_atm_symb in ['O', 'N'])
            elif req == 'no_aromatic':
                cond = (bond.GetIsAromatic())
            elif req == 'donor':
                cond = (other_atm_symb == 'H')
            if cond:
                supp_inds += [ind]
                break
    fil_inds = [i for i in inds if i not in supp_inds]
    
    return (fil_inds, supp_inds)

def identify_pharmacophore_formol(mol2df, coor, mol):
    """
    Find pharmacophoric atoms in a molecule, and return different types of pharmacophoric atoms.
    Parameter:
        mol2df - a PandasMol dataframe for a molecule
        coor - atom coordinates of a molecule
        mol - an rdkit.Chem.rdchem.Mol molecule
    """
    pharmacophorelst = {'hydrophobic':[], 'aromatic':[], 'hbond_acceptor':[], 'hbond_donor':[], 'cation':[], 'anion':[], 'metal':[], 'backbone':[]}

    print('hydrophobic and aromatic...')
    tmp_inds1 = get_atmind_withTpReq(mol2df = mol2df, colname = 'atom_type', filter_lst = ['C.1', 'C.2', 'C.3', 'S.2', 'S.3'], coor = coor, req = None)
    tmp_inds2 = get_atmind_withTpReq(mol2df = mol2df, colname = 'atom_type', filter_lst = ['C.ar', 'N.ar'], coor = coor, req = None)
    supp_inds1 = fil_atmind_withBondReq(inds = tmp_inds1, mol = mol, req = 'hydro')
    supp_inds2 = fil_atmind_withBondReq(inds = tmp_inds2, mol = mol, req = 'no_aromatic')
    print('h-bond acceptor and donor...')
    tmp_inds3 = get_atmind_withTpReq(mol2df = mol2df, 
                                     colname = 'atom_type', 
                                     filter_lst = ['N.1', 'N.2', 'N.3', 'N.pl3', 'O.2', 'O.3', 'O.spc', 'O.t3p'], 
                                     coor = coor, req = 'negcharge')
    tmp_inds4 = get_atmind_withTpReq(mol2df = mol2df, 
                                     colname = 'atom_type', 
                                     filter_lst = ['N.am', 'N.pl3', 'N.4', 'O.3', 'O.spc', 'O.t3p'], 
                                     coor = coor, req = None)
    supp_inds3 = fil_atmind_withBondReq(inds = tmp_inds4, mol = mol, req = 'donor')   
    print('cation and anion...')
    tmp_inds5 = get_atmind_withTpReq(mol2df = mol2df, colname = 'atom_type', filter_lst = ['C.cat', 'N.4'], coor = coor, req = None)
    tmp_inds6 = get_atmind_withTpReq(mol2df = mol2df, colname = 'atom_type', filter_lst = None, coor = coor, req = 'poscharge')
    tmp_inds7 = get_atmind_withTpReq(mol2df = mol2df, 
                                     colname = 'atom_type', 
                                     filter_lst = ['O.co2', 'Mg', 'Zn', 'Mn', 'Ca', 'Fe', 'Cu'], 
                                     coor = coor, req = None)
    tmp_inds8 = get_atmind_withTpReq(mol2df = mol2df, colname = 'atom_type', filter_lst = None, coor = coor, req = 'negcharge')
    print('metal...')
    tmp_inds9 = get_atmind_withTpReq(mol2df = mol2df, 
                                     colname = 'atom_type', 
                                     filter_lst = ['Mg', 'Zn', 'Mn', 'Ca', 'Fe', 'Cu'], 
                                     coor = coor, req = None)
    tmp_inds10 = get_atmind_withTpReq(mol2df = mol2df, colname = 'atom_name', filter_lst = ['N', 'CA', 'C', 'O', 'H', 'HNCAP'], coor = coor, req = None)
    pharmacophorelst['hydrophobic'] = supp_inds1[0] + supp_inds2[0]
#    pharmacophorelst['aromatic'] = supp_inds2[1]
    pharmacophorelst['aromatic'] = [list(ring) for ring in mol.GetRingInfo().AtomRings()]
    pharmacophorelst['hbond_acceptor'] = tmp_inds3
    pharmacophorelst['hbond_donor'] = supp_inds3[1]
    pharmacophorelst['cation'] = tmp_inds5 + tmp_inds6
    pharmacophorelst['anion'] = tmp_inds7 + tmp_inds8
    pharmacophorelst['metal'] = tmp_inds9
    pharmacophorelst['backbone'] = tmp_inds10
    
    return pharmacophorelst

# definitions of functions and classes
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def getangle(v1, v2):
    len1 = sqrt(dotproduct(v1, v1))
    len2 = sqrt(dotproduct(v2, v2))
    ang = acos(dotproduct(v1, v2) / len1 / len2)
    return ang

# Hbond donors and acceptors referred to the following situations:
# Donor: O, N, S; Acceptor: N, O; rule: dis <= 3.5 and ang <= pi/4 and ang >= -pi/4 (ang: <dh,ha)
#                       ("Encoding protein-ligand interaction patterns in fingerprints and graphs")
def check_Hbond(donor_idx, mol_donor, molcoor_donor, acceptor_idx, mol_acceptor, molcoor_acceptor):
    """
    Check whether a hydrogen bond exists.
    Parameters:
        donor_idx - the index of donor in mol_donor
        mol_donor - an rdkit.Chem.rdchem.Mol molecule that provides the donor
        molcoor_donor - atom coordinates of the molecule that provides that donor
        acceptor_idx - the index of acceptor in mol_donor
        mol_acceptor - an rdkit.Chem.rdchem.Mol molecule that provides the acceptor        
        molcoor_acceptor - atom coordinates of the molecule that provides that acceptor
    """
    donor = mol_donor.GetAtomWithIdx(donor_idx)
    donor_coor = molcoor_donor[donor_idx]
    acceptor = mol_acceptor.GetAtomWithIdx(acceptor_idx)
    acceptor_coor = molcoor_acceptor[acceptor_idx]
    dis = cdist(donor_coor.reshape(-1,3), acceptor_coor.reshape(-1,3), metric = 'euclidean')[0,0]
    if dis > 3.5:
        return (0,)
    else:
        for bond in donor.GetBonds():
            nbr_idx = bond.GetOtherAtomIdx(donor_idx)
            nbr_coor = molcoor_donor[nbr_idx]
            nbr = mol_donor.GetAtomWithIdx(nbr_idx)
            if nbr.GetSymbol() == 'H':
                DH = nbr_coor - donor_coor
                HA = acceptor_coor -  nbr_coor
                ang = getangle(DH, HA)
                if abs(ang) <= pi/4:
                    return (1, (donor.GetSymbol(), acceptor.GetSymbol(), dis, ang))
        return (0,)

def get_plane_normal(points):
    """
    return the normal of the plane formed by three points.
    """
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    return np.cross(v1, v2)
    

def check_aromatic(sets1, coor1, sets2, coor2):
    """
    Check whether a face-to-face aromatic interaction exists.
    Parameters:
        sets1: a set of indices corresponding to the aromatic atoms in coor1.
        coor1: atom coordinates of the molecule providing aromatic ring of sets1.
        sets2: a set of indices corresponding to the aromatic atoms in coor2.
        coor2: atom coordinates of the molecule providing aromatic ring of sets2.
    Return 1 if a face-to-face aromatic interaction exisits, 2 if a edge-to-face aromatic interaction exists, 0 if no aromatic interaction
    """
    # compute the distance between the centers of the two aromatic rings
    center1 = np.mean(coor1, axis = 0)
    center2 = np.mean(coor2, axis = 0)
    dis_c2c = cdist(center1.reshape(-1,3), center2.reshape(-1,3), metric = 'euclidean')[0,0]
    cond1 = (dis_c2c <= 4)
    # compute the pairwise distances between any two atoms belonging to the two aromatic rings
    ring1 = coor1[sets1]
    ring2 = coor2[sets2]
    dismat = np.triu(cdist(ring1, ring2, metric = 'euclidean'))
    pwdis = dismat[np.nonzero(dismat)]
    cond2 = all(pwdis <= 12)
    # compute the angle between the two aromatic rings
    normal1 = get_plane_normal(points = ring1[:3])
    normal2 = get_plane_normal(points = ring2[:3])
    ang = getangle(normal1, normal2)
    if ang > (pi*5/ 6):
        ang = ang - pi 
    cond3 = (abs(ang) <= (pi/6))
    cond4 = (ang <= (pi*5/6) and ang > (pi/6))
    
    if cond1 and cond2 and cond3:
        return 1
    elif cond1 and cond4:
        return 2
    else:
        return 0
    

def get_index(dis):
    if dis > 0 and dis <= 2.5:
        return 0
    elif dis > 2.5 and dis <= 4:
        return 1
    elif dis > 4 and dis <= 6:
        return 2
    elif dis > 6 and dis <= 9:
        return 3
    elif dis > 9 and dis <= 13:
        return 4
    elif dis > 13 and dis <= 18:
        return 5
    else:
        return 6
    
def get_pwintpos(int1, pro_coor, int2, lig_coor, pwint_type):
    """
    For a pairwise interaction (e.g. hydrophobic-hydrophobic), get the position of this interaction in the APIF
    Parameters:
        int1 - the first interaction for comprising a pairwise interaction 
               the format for an interaction is like:
               (index of protein atom, index of ligand atom, symbol of protein atom, sysmbol of ligand atom,
               index of protein residue, number of protein residue),
               e.g. (4238, 14, 'C', 'Br', 262, 'GLN')
        pro_coor - the coordinates of protein atoms
        int2 - the second interaction for comprising a pairwise interaction
        lig_coor - the coordinates of ligand atoms
        pwint_type - type of the pairwise interaction, including:
                     0: hydrophobic-hydrophobic
                     1: hydrophobic-hbond(acceptor from protein)
                     2: hydrophobic-hbond(donor from protein)
                     3: hbond(acceptor from protein)-hbond(donor from protein)
                     4: hbond(acceptor from protein)-hbond(acceptor from protein)
                     5: hbond(donor from protein)-hbond(donor from protein)
    """
    if pwint_type not in range(6):
        print('Wrong type of pairwise interaction!')
        return (-1, -1, -1)
    proat_ind1 = int1[0]
    proat_ind2 = int2[0]
    ligat_ind1 = int1[1]
    ligat_ind2 = int2[1]
    
    dis_proat = cdist(pro_coor[proat_ind1].reshape(-1,3), pro_coor[proat_ind2].reshape(-1,3), metric = 'euclidean')[0,0]
    dis_ligat = cdist(lig_coor[ligat_ind1].reshape(-1,3), lig_coor[ligat_ind2].reshape(-1,3), metric = 'euclidean')[0,0]
    # distance between two protein atoms divided into 7 bins (0-2.5, 2.5-4, 4-6, 6-9, 9-13, 13-18, >18)
    # distance between two ligand atoms divided into 7 bins (0-2.5, 2.5-4, 4-6, 6-9, 9-13, 13-18, >18)
    dis_ind1 = get_index(dis_proat)
    dis_ind2 = get_index(dis_ligat)
    return (pwint_type, dis_ind1, dis_ind2)   

def get_pwintpos2(int1, int2, lig_coor, pwint_type):
    """
    For a pairwise interaction (e.g. hydrophobic-hydrophobic), get the position of this interaction in the Pharm-IF,
    here only consider the distance bins of 1, 2, ..., 18
    Parameters:
        int1 - the first interaction for comprising a pairwise interaction 
               the format for an interaction is like:
               (index of protein atom, index of ligand atom, symbol of protein atom, sysmbol of ligand atom,
               index of protein residue, number of protein residue),
               e.g. (4238, 14, 'C', 'Br', 262, 'GLN')
        int2 - the second interaction for comprising a pairwise interaction
        lig_coor - the coordinates of ligand atoms
        pwint_type - type of the pairwise interaction, including:
                     0: hydrophobic-hydrophobic
                     1: hydrophobic-hbond(acceptor from protein)
                     2: hydrophobic-hbond(donor from protein)
                     3: hbond(acceptor from protein)-hbond(donor from protein)
                     4: hbond(acceptor from protein)-hbond(acceptor from protein)
                     5: hbond(donor from protein)-hbond(donor from protein)
    """
    if pwint_type not in range(6):
        print('Wrong type of pairwise interaction!')
        return (-1, -1)
    ligat_ind1 = int1[1]
    ligat_ind2 = int2[1]
    
    dis_ligat = cdist(lig_coor[ligat_ind1].reshape(-1,3), lig_coor[ligat_ind2].reshape(-1,3), metric = 'euclidean')[0,0]
    if dis_ligat <= 1:
        return [((int(pwint_type), int(0)), 1 - abs(1 - dis_ligat))]
    elif dis_ligat >= 18:
        return [((int(pwint_type), int(17)),  1 - abs(18 - dis_ligat))]
    else:
        bin1 = np.floor(dis_ligat)
        bin2 = np.ceil(dis_ligat)
        if bin1 == bin2:
            return [((int(pwint_type), int(bin1 - 1)),  1 - abs(bin1 - dis_ligat))]
        else:
            return [((int(pwint_type), int(bin1 - 1)),  1 - abs(bin1 - dis_ligat)), ((int(pwint_type), int(bin2 - 1)),  1 - abs(bin2 - dis_ligat))]

def ifp_folding(identifiers = [],
                channel_power = 10,
                counts = 0):
    """
    Folds a list of integer identifiers to a bit vector of fixed-length.
    Parameters:
        identifiers - a list of integers
        channel_power - decides the length of feature vector
        counts - use occurences of identifiers (1) or not (0)
    Returns a final feature vector of length 2^channel_power
    """
    feature_vector = np.zeros(2**channel_power)
    on_channels = [int(idf % 2**channel_power) for idf in identifiers]
    if counts:
        for ch in on_channels:
            feature_vector[ch] += 1
    else:
        feature_vector[on_channels] += 1

    return feature_vector


class SIFt(object):
    def __init__(self, fn_pro_PDB, fn_lig_PDB, fn_pro_MOL2, fn_lig_MOL2, ID = None, addH = True, sant = True, int_cutoff = 4.5):
        """
        Initialize an SIFt class.
        Parameters:
            fn_pro_PDB - PDB file name of the protein
            fn_lig_PDB - PDB file name of the ligand
            fn_pro_MOL2 - MOL2 file name of the protein
            fn_lig_MOL2 - MOL2 file name of the ligand
            ID - ID of the complex
            addH - whether to add hydrogen atoms when reading in the structure file
            sant - whether to sanitize the molecule when reading in the structure file
            int_cutoff - distance threshold for identifying protein-ligand interacting atoms 
        """
        self.ID = ID if ID is not None else "PL"
        print('Constructing an SIFt object for %s.........\n' % self.ID)
        # read in pdb coordinates and topology
        self.lig = (load_molecule(fn_lig_PDB, add_hydrogens=addH, calc_charges=False, sanitize=sant)) 
        self.pro = (load_molecule(fn_pro_PDB, add_hydrogens=addH, calc_charges=False, sanitize=sant))
        # read in mol2 file for identifying specific atom types
        self.lig_mol2 = PandasMol2().read_mol2(fn_lig_MOL2).df
        self.pro_mol2 = PandasMol2().read_mol2(fn_pro_MOL2).df
        # parse protein pdb file for identifying sidechain/mainchain atoms
#        parser = PDBParser()
#        self.structure = parser.get_structure(self.ID, fn_pro_PDB)
#        self.chid = self.pro[1].GetAtomWithIdx(0).GetPDBResidueInfo().GetChainId()
        # identify interacting area
        self.contact_bins = [(0, int_cutoff)]
        self.pd = cdist(self.pro[0], self.lig[0], metric = 'euclidean')
        contacts = np.nonzero(self.pd < int_cutoff)
        tmpcont = [(int(i), int(j)) for (i, j) in zip(contacts[0], contacts[1])]
        self.cont = [[int(i) for i in contacts[0]], [int(j) for j in contacts[1]]]
        self.contacts = []
        tmp = self.pro_mol2[['x', 'y', 'z']]
        mol2coor = list(tmp.apply(lambda row: [float(round(row.x, 3)), float(round(row.y, 3)), float(round(row.z, 3))], axis=1))
#        excludeATs = ['ZN', 'Zn', 'CU', 'Cu', 'FE', 'Fe', 'CO', 'Co', 'MN', 'Mn', 'CA', 'Ca', 'MG', 'Mg', 'K', 'NA', 'Na', 'CL', 'Cl']
        for cont in tmpcont:
            proat = self.pro[1].GetAtomWithIdx(cont[0])
            ligat = self.lig[1].GetAtomWithIdx(cont[1])
#            procoor = self.pro[0][cont[0]]
#            ligcoor = self.lig[0][cont[1]]
            if proat.GetPDBResidueInfo().GetResidueName() != 'HOH' and proat.GetSymbol() != 'H' and ligat.GetSymbol() != 'H':
                proat_chid = proat.GetPDBResidueInfo().GetChainId()
                proat_resno = proat.GetPDBResidueInfo().GetResidueNumber()
                proat_resname = proat.GetPDBResidueInfo().GetResidueName()
                resid = '%c_%d_%s' % (proat_chid, proat_resno, proat_resname)
                ck_coor = self.pro[0][cont[0]].tolist()
                if ck_coor in mol2coor:
                    i_in_mol2 = mol2coor.index(ck_coor)
                    proat_name = self.pro_mol2['atom_name'][i_in_mol2]
                    atmid = '%s_%s' % (resid, proat_name)
                    self.contacts += [(cont[0], cont[1], resid, atmid)]       
        self.reslst = []
        self.atmlst = []
        self.pro_pharmacophorelst = {}
        self.lig_pharmacophorelst = {}
        self.intlst = {'contact': [], 'donor': [], 'acceptor': [], 'hydrophobic': [], 'backbone': [], 'sidechain': [], 
                       'polar': [], 'nonpolar': [], 'metalacceptor':[], 'closecont':[], 'aromaticF2F':[], 'aromaticE2F':[]}
    
    def output_bld_file(self, radius_res = 2, radius_atm = 1):
        """
        Construct .bld file of the binding-site residues/atoms, for visulization.
        """
        res = {}
        atm = []
        prs = []
        self.resposbildstrs = ['.color blue\n']
        self.atmposbildstrs = ['.color green\n']
        self.contbildstrs = []
        for (i, j, resid, atmid) in self.contacts:
            proat = self.pro[1].GetAtomWithIdx(i)
            proatcoor = self.pro[0][i].tolist()
#            proat_resname = proat.GetPDBResidueInfo().GetResidueName()
            ligat = self.lig[1].GetAtomWithIdx(j)    
            ligatcoor = self.lig[0][j].tolist()
            if proat.GetSymbol() != 'H' and ligat.GetSymbol() != 'H':
                prs.append((proatcoor, ligatcoor))
            if resid not in res:
                res[resid] = proatcoor
            if proatcoor not in atm:
                atm.append(proatcoor)
        for item in list(res.values()):
            self.resposbildstrs.append('.sphere %.2f %.2f %.2f %.2f\n' % (item[0], item[1], item[2], radius_res))
        for item in atm:
            self.atmposbildstrs.append('.sphere %.2f %.2f %.2f %.2f\n' % (item[0], item[1], item[2], radius_atm))
        tmp1 = []
        tmp2 = []
        tmp3 = []
        for item in prs:
            tmp1.append('.sphere %.2f %.2f %.2f %.2f\n' % (item[0][0], item[0][1], item[0][2], radius_atm))
            tmp2.append('.sphere %.2f %.2f %.2f %.2f\n' % (item[1][0], item[1][1], item[1][2], radius_atm))
            tmp3.append('.vector %.2f %.2f %.2f %.2f %.2f %.2f\n' % (item[0][0], item[0][1], item[0][2], item[1][0], item[1][1], item[1][2]))
        self.contbildstrs = ['.color green\n'] + tmp1 + ['.color red\n'] + tmp2 + ['.color purple\n'] + tmp3  
    
    def output_interaction_bld_file(self, radius_atm = 1):
        """
        Construct .bld file of the interacting atoms, for visulization.
        """
        eletoclr = {'C': '.color 1 1 0\n', 'N': '.color 0 1 1\n', 'O': '.color 1 0 1\n', 'S': '.color 0.5 0.5 1\n'}
        # yellow, blue, pink, purple
        bld = {eletoclr['C']: [], eletoclr['N']: [], eletoclr['O']: [], eletoclr['S']: []}
        for (i, j, resid, atmid) in self.contacts:
            proat = self.pro[1].GetAtomWithIdx(i).GetSymbol()
            proatcoor = self.pro[0][i].tolist()
            ligat = self.lig[1].GetAtomWithIdx(j).GetSymbol()   
            ligatcoor = self.lig[0][j].tolist()
            if proat in eletoclr:
                probldstr = '.sphere %.2f %.2f %.2f %.2f\n' % (proatcoor[0], proatcoor[1], proatcoor[2], radius_atm)
                if probldstr not in bld[eletoclr[proat]]:
                    bld[eletoclr[proat]].append(probldstr)
            if ligat in eletoclr:
                ligbldstr = '.sphere %.2f %.2f %.2f %.2f\n' % (ligatcoor[0], ligatcoor[1], ligatcoor[2], radius_atm)
                if ligbldstr not in bld[eletoclr[ligat]]:
                    bld[eletoclr[ligat]].append(ligbldstr)
                    
        bldstrs = []
        for char in bld:
            bldstrs += ([char] + bld[char])
        
        return bldstrs           

    def find_contacts(self):
        """
        Find all contacts with atom distance < 4.5A, record info of atom/res
        """
        for (i, j, resid, atmid) in self.contacts:
            proat = self.pro[1].GetAtomWithIdx(i)
            ligat = self.lig[1].GetAtomWithIdx(j)            
            if atmid not in self.atmlst:
                self.atmlst += [atmid]     
            if resid not in self.reslst:
                self.reslst += [resid]
            self.intlst['contact'] += [(i, j, proat.GetSymbol(), ligat.GetSymbol(), resid, atmid)]
    
    def find_pharmacophoric_atoms(self):
        """
        Assign pharmacophoric properties to protein/ligand atoms, including:
            hydrophobic, aromatic, h-bond acceptor, h-bond donor, positive ionizable, negative ionizable, metal
        Reference: "Encoding protein-ligand interaction patterns in fingerprints and graphs"
        """
        # 1. find pharmacophoric atoms in protein
        coor_pro = self.pro[0].tolist()
        self.pro_pharmacophorelst = identify_pharmacophore_formol(mol2df = self.pro_mol2, coor = coor_pro, mol = self.pro[1])
            
        # 2. find pharmacophoric atoms in ligand
        coor_lig = self.lig[0].tolist()
        self.lig_pharmacophorelst = identify_pharmacophore_formol(mol2df = self.lig_mol2, coor = coor_lig, mol = self.lig[1])

        
    def find_hydrophobic_int(self):
        """
        Find hydrophobic interactions: atoms are hydrophobic and atom distance < 4.5A 
        """
        ligphlst = self.lig_pharmacophorelst['hydrophobic']
        prophlst = self.pro_pharmacophorelst['hydrophobic'] 
        ligcoor = self.lig[0]
        procoor = self.pro[0]
        for (i, j, resid, atmid) in self.contacts:
            if j in ligphlst and i in prophlst:
                proat = self.pro[1].GetAtomWithIdx(i)
                ligat = self.lig[1].GetAtomWithIdx(j)
                dis = cdist(ligcoor[j].reshape(-1,3), procoor[i].reshape(-1,3), metric = 'euclidean')[0,0]
                if dis <= 4.5:
                    self.intlst['hydrophobic'] += [(i, j, proat.GetSymbol(), ligat.GetSymbol(), resid, atmid)]
    
    def find_aromatic_int(self):
        """
        Find face-to-face or edge-to-face aromatic interactions.
        """
        ligphlst = self.lig_pharmacophorelst['aromatic']
        prophlst = self.pro_pharmacophorelst['aromatic'] 
        ligcoor = self.lig[0]
        procoor = self.pro[0]
        for (i, j, resid, atmid) in self.contacts:
            cur_proring = [ring for ring in prophlst if i in ring]
            cur_ligring = [ring for ring in ligphlst if j in ring]
            if len(cur_proring) > 0 and len(cur_ligring) > 0:
                proat = self.pro[1].GetAtomWithIdx(i)
                ligat = self.lig[1].GetAtomWithIdx(j)
                ck = check_aromatic(sets1 = cur_proring[0], coor1 = procoor, sets2 = cur_ligring[0], coor2 = ligcoor)
                if ck == 1:
                    self.intlst['aromaticF2F'] += [(i, j, proat.GetSymbol(), ligat.GetSymbol(), resid, atmid)]
                elif ck == 2:
                    self.intlst['aromaticE2F'] += [(i, j, proat.GetSymbol(), ligat.GetSymbol(), resid, atmid)]
   
    def find_backboneORsidechain_int(self):
        """
        Find interactions involving either backbone atoms or sidechain atoms
        """
        for (i, j, resid, atmid) in self.contacts:
            proat = self.pro[1].GetAtomWithIdx(i)
            ligat = self.lig[1].GetAtomWithIdx(j)
            if i in self.pro_pharmacophorelst['backbone']:
                self.intlst['backbone'] += [(i, j, proat.GetSymbol(), ligat.GetSymbol(), resid, atmid)]
            else:
                self.intlst['sidechain'] += [(i, j, proat.GetSymbol(), ligat.GetSymbol(), resid, atmid)]
    
    def find_hydrogen_bond(self):
        """
        Find hydrogen bonds D-H...A: 
            atoms are N, O; DA distance <= 3.5A; <DH,HA> E [-pi/4, pi/4]
        """
        for (i, j, resid, atmid) in self.contacts:
            proat = self.pro[1].GetAtomWithIdx(i)
            ligat = self.lig[1].GetAtomWithIdx(j)
            if j in self.lig_pharmacophorelst['hbond_acceptor'] and i in self.pro_pharmacophorelst['hbond_donor']:
#                print((i, j))
                ck = check_Hbond(donor_idx = i, mol_donor = self.pro[1], molcoor_donor = self.pro[0], 
                                 acceptor_idx = j, mol_acceptor = self.lig[1], molcoor_acceptor = self.lig[0])
                if ck[0] == 1:
                    self.intlst['donor'] += [(i, j, proat.GetSymbol(), ligat.GetSymbol(), resid, atmid, ck[1])]                    
            elif j in self.lig_pharmacophorelst['hbond_donor'] and i in self.pro_pharmacophorelst['hbond_acceptor']:
#                print((i, j))
                ck = check_Hbond(donor_idx = j, mol_donor = self.lig[1], molcoor_donor = self.lig[0], 
                                 acceptor_idx = i, mol_acceptor = self.pro[1], molcoor_acceptor = self.pro[0])
                if ck[0] == 1:
                    self.intlst['acceptor'] += [(i, j, proat.GetSymbol(), ligat.GetSymbol(), resid, atmid, ck[1])]             
                 
    def find_polar_int(self):
        """
        Find polar (dipole-dipole) or non-polar interactions
        """
        ligcationlst = self.lig_pharmacophorelst['cation']
        liganionlst = self.lig_pharmacophorelst['anion']
        procationlst = self.pro_pharmacophorelst['cation'] 
        proanionlst = self.pro_pharmacophorelst['anion'] 
        for (i, j, resid, atmid) in self.contacts:
            if (i in procationlst and j in liganionlst) or (i in proanionlst and j in ligcationlst):
                proat = self.pro[1].GetAtomWithIdx(i)
                ligat = self.lig[1].GetAtomWithIdx(j)
                dis = cdist(self.pro[0][i].reshape(-1,3), self.lig[0][j].reshape(-1,3), metric = 'euclidean')[0,0]
                if dis <= 4.0:
                    self.intlst['polar'] += [(i, j, proat.GetSymbol(), ligat.GetSymbol(), resid, atmid, dis)]  

    def find_nonpolar_int(self):
        """
        Find polar (dipole-dipole) or non-polar interactions
        """
        for (i, j, resid, atmid) in self.contacts:
            proat = self.pro[1].GetAtomWithIdx(i)
            ligat = self.lig[1].GetAtomWithIdx(j)
            if proat.GetSymbol() == ligat.GetSymbol():
                self.intlst['nonpolar'] += [(i, j, proat.GetSymbol(), ligat.GetSymbol(), atmid, resid)]  
    
    def find_metalacceptor_int(self):
        """
        Find polar (dipole-dipole) or non-polar interactions
        """
        ligacptlst = self.lig_pharmacophorelst['hbond_acceptor']
        prometallst = self.pro_pharmacophorelst['metal'] 
        for (i, j, resid, atmid) in self.contacts:
            if i in prometallst and j in ligacptlst:
                proat = self.pro[1].GetAtomWithIdx(i)
                ligat = self.lig[1].GetAtomWithIdx(j)
                dis = cdist(self.pro[0][i].reshape(-1,3), self.lig[0][j].reshape(-1,3), metric = 'euclidean')[0,0]
                if dis <= 2.8:
                    self.intlst['metalacceptor'] += [(i, j, proat.GetSymbol(), ligat.GetSymbol(), resid, atmid, dis)]  
   
    def find_closecont(self):
        """
        Find close contacts whose interacting distance is shorter than the sum of VDW radii of non-hbond atom pairs.
        This funtion should be executed after find_hydrogen_bond() function.
        """
        hbonds = [(item[0], item[1]) for item in self.intlst['donor']] + [(item[0], item[1]) for item in self.intlst['acceptor']] 
        tbl = GetPeriodicTable()
        ligcoor = self.lig[0]
        procoor = self.pro[0]
        for (i, j, resid, atmid) in self.contacts:
            proat = self.pro[1].GetAtomWithIdx(i)
            ligat = self.lig[1].GetAtomWithIdx(j)
            vdw_radii = tbl.GetRvdw(proat.GetAtomicNum()) + tbl.GetRvdw(ligat.GetAtomicNum())
            if (i, j) not in hbonds:
                dis = cdist(ligcoor[j].reshape(-1,3), procoor[i].reshape(-1,3), metric = 'euclidean')[0,0]
                if dis < vdw_radii:
                    self.intlst['closecont'] += [(i, j, proat.GetSymbol(), ligat.GetSymbol(), resid, atmid)]  
                 
    def find_int(self):
        """
        Find different types of protein-ligand interactions and save them in self.intlst.
        """
        self.find_contacts()
        self.find_pharmacophoric_atoms()
        self.find_hydrophobic_int()
        self.find_backboneORsidechain_int()
        self.find_hydrogen_bond()
        self.find_closecont()
        self.find_polar_int()
        self.find_nonpolar_int()
        self.find_metalacceptor_int()
    
    def construct_SIFt(self, 
                       reslst = None, 
                       intlst = ['contact', 'backbone', 'sidechain', 'polar', 'nonpolar', 'donor', 'acceptor'], 
                       count = 0):
        """
        Construct SIFt using the list of residues in reslst. 
        By default, each residue corresponds to 7 bits of interactions:
            bit1 - contact or not
            bit2 - backbone atom involved or not
            bit3 - sidechain atom involved or not
            bit4 - polar interaction involved or not
            bit5 - nonpolar interaction involved or not
            bit6 - providing h-bond donor or not
            bit7 - providing h-bond acceptor or not
        Parameters:
            reslst - a list of interacting-residue numbers for constructing SIFt (if None, using self.reslst)
            intlst - a list of interaction types for constructing IFPs
            count - whether to count the occurences of each interaction when constructing SIFt (1: count, 0: not count)
        """
        tmpdict = {}
        sift = []
        lst = reslst if reslst is not None else self.reslst
        lst.sort()
        for res in lst:
            tmpdict[res] = [0] * len(intlst)
        # bits 1 ~ 7
        for ind in range(len(intlst)):
            tmplst = self.intlst[intlst[ind]]
            if len(tmplst) > 0:
                for item in tmplst:
                    if item[4] in tmpdict:
                        if count == 0:
                            tmpdict[item[4]][ind] = 1
                        else:
                            tmpdict[item[4]][ind] += 1
        
        for tmp in list(tmpdict.values()):
            sift += tmp
        
        return np.array(sift)
        
    def construct_KB_IFP(self, atmlst = None, inttype = 'CH', count = 0):
        """
        Construct knowledge-based IFPs using the list of atoms in atmlst. 
        Parameters:
            atmlst - a list of interacting-atom numbers for constructing SIFt (if None, using self.atmlst)
            inttype - type of KB_IFP
                      'C': each bit corresponds to whether a close contact exists
                      'H': each bit corresponds to wether a hbond exists
                      'CH': each bit corresponds to whether a close contact or hbond exists
            count - whether to count the occurences of each interaction when constructing SIFt (1: count, 0: not count)
        """
        intdict = {'C': ['closecont'], 'H': ['donor', 'acceptor'], 'CH': ['closecont', 'donor', 'acceptor']}
        tmpdict = {}
        lst = atmlst if atmlst is not None else self.atmlst
        lst.sort()
        for atm in lst:
            tmpdict[atm] = 0
        ints = intdict.get(inttype, 'error')
        if ints == 'error':
            print('Error interaction type!!!')
            return []
        for ind in range(len(ints)):
            tmplst = self.intlst[ints[ind]]
            if len(tmplst) > 0:
                for item in tmplst:
                    if item[5] in tmpdict:
                        if count == 0:
                            tmpdict[item[5]] = 1
                        else:
                            tmpdict[item[5]] += 1
               
        return np.array(list(tmpdict.values()))
    
    def construct_APIF(self, count = 0):
        """
        Construct APIF based on pairwise interactions. 
        Types of pairwise interactions include:
            0: hydrophobic-hydrophobic
            1: hydrophobic-hbond(acceptor from protein)
            2: hydrophobic-hbond(donor from protein)
            3: hbond(acceptor from protein)-hbond(donor from protein)
            4: hbond(acceptor from protein)-hbond(acceptor from protein)
            5: hbond(donor from protein)-hbond(donor from protein)
        Parameters:
            count - whether to count the occurences of each pairwise interaction when constructing APIF (1: count, 0: not count)
        """
        apif = np.zeros((6, 7, 7))
        pwints = [('hydrophobic', 'hydrophobic'), ('hydrophobic', 'acceptor'), ('hydrophobic', 'donor'),
                  ('acceptor', 'donor'), ('acceptor', 'acceptor'), ('donor', 'donor')]
        for ind in range(len(pwints)):
            lst1 = self.intlst[pwints[ind][0]]
            lst2 = self.intlst[pwints[ind][1]]
            if len(lst1) > 0 and len(lst2) > 0:
                if ind in [1, 2, 3]:
                    for int1 in lst1:
                        for int2 in lst2:
                            pos = get_pwintpos(int1, self.pro[0], int2, self.lig[0], ind)
                            if count == 0:
                                apif[pos] = 1
                            else:
                                apif[pos] += 1
                else:
                    for int1, int2 in list(combinations(lst1, 2)):
                        pos = get_pwintpos(int1, self.pro[0], int2, self.lig[0], ind)
                        if count == 0:
                            apif[pos] = 1
                        else:
                            apif[pos] += 1                    
        
        return apif.flatten()     

    def construct_PharmIF(self):
        """
        Construct PharmIF based on pairwise interactions. 
        Types of pairwise interactions include:
            0: hydrophobic-hydrophobic
            1: hydrophobic-hbond(acceptor from protein)
            2: hydrophobic-hbond(donor from protein)
            3: hbond(acceptor from protein)-hbond(donor from protein)
            4: hbond(acceptor from protein)-hbond(acceptor from protein)
            5: hbond(donor from protein)-hbond(donor from protein)
        """
        pharmif = np.zeros((6, 18))
        pwints = [('hydrophobic', 'hydrophobic'), ('hydrophobic', 'acceptor'), ('hydrophobic', 'donor'),
                  ('acceptor', 'donor'), ('acceptor', 'acceptor'), ('donor', 'donor')]
        for ind in range(len(pwints)):
            lst1 = self.intlst[pwints[ind][0]]
            lst2 = self.intlst[pwints[ind][1]]
            if len(lst1) > 0 and len(lst2) > 0:
                if ind in [1, 2, 3]:
                    for int1 in lst1:
                        for int2 in lst2:
                            pos = get_pwintpos2(int1, int2, self.lig[0], ind)
                            for item in pos:
                                pharmif[item[0]] += item[1]
                else:
                    for int1, int2 in list(combinations(lst1, 2)):
                        pos = get_pwintpos2(int1, int2, self.lig[0], ind)
                        for item in pos:
                            pharmif[item[0]] += item[1]
        
        return pharmif.flatten()     
    
    def get_quasi_fragmental_desp(self, 
                                  bins = None,
                                  md_type = 'rfscore',
                                  reftypes = None,
                                  solo = 0):
        """
        Compute quasi fragmental descriptors.
        Parameters:
            bins - distance bins for extracting qf descriptors
            md_type - descriptor type ('rfscore': <12A, 'onionnet': < 30.5A or 'ecif': <6A)   
            reftypes - lists of atom types (solo = 1, e.g. [prolist = [tp1,...], liglist = [tp1,...]]) or atom-pair types (solo = 0, e.g. [(atp_pro1, atp_lig1),...]) for extracting features
            solo - whether to calculate the features for pro and lig separately (1 - yes, 0 - no)
        """
        if bins is not None:
            self.contact_bins = bins
        else:
            if md_type == 'rfscore':
                self.contact_bins = [(0, 12)]
            elif md_type == 'onionnet':
                self.contact_bins = [(0, 1)]
                for shell in np.arange(2, 61):
                    self.contact_bins += [(1 + (shell - 2) * 0.5, 1 + (shell - 1) * 0.5)]     
            elif md_type == 'ecif':
                self.contact_bins = [(0, 6)]
            else:
                return []

        # list atom types ----------------------------------------------------
        atNumDict_onnt = {6: -1, 7: -2, 8: -3, 1: -4, 15: -5, 16: -6, 9: -7, 17: -7, 35: -7, 53: -7} 
        at_filt_ecif = [6, 7, 8, 9, 15, 16, 17, 35, 53]
        if md_type == 'rfscore':
            pro_pool = [6, 7, 8, 16]
            lig_pool = [6, 7, 8, 9, 15, 16, 17, 35, 53]
        elif md_type == 'onionnet':
            pro_pool = [-a for a in np.arange(1, 9)]
            lig_pool = pro_pool
            # use atNumDict_onnt(orinum, default) to map the atom number to new (default is -8)
        else:
            if reftypes is not None:
                if solo == 0:
                    pro_pool = []
                    lig_pool = []
                elif solo in [1, 100]:
                    pro_pool = reftypes[0]
                    lig_pool = reftypes[1] 
                elif solo == -1:
                    pro_pool = reftypes[1][0]
                    lig_pool = reftypes[1][1] 
                else:
                    print('Wrong parameter (solo)!')
                    return [], []
            else:
                pro_pool = []
                lig_pool = []
        # get list of atom-pair types ----------------------------------------
        if md_type == 'ecif' and reftypes is not None and solo == 0:
            quasi_type = reftypes
        elif md_type == 'ecif' and reftypes is not None and solo == -1:
            quasi_type = reftypes[0]
        else:
            quasi_type = list(itertools.product(pro_pool, lig_pool))
        
        quasi_feat = []
        quasi_feat_types = []
        quasi_feat_types_pro = []
        quasi_feat_types_lig = []
        if md_type == 'ecif':
            prorings = [list(ring) for ring in self.pro[1].GetRingInfo().AtomRings()]
            pro_ringatoms_dict = {} if len(prorings) == 0 else {i:1 for i in np.unique(np.concatenate(prorings))}
            ligrings = [list(ring) for ring in self.lig[1].GetRingInfo().AtomRings()]
            lig_ringatoms_dict = {} if len(ligrings) == 0 else {i:1 for i in np.unique(np.concatenate(ligrings))}

        for cbin in self.contact_bins:
            # initiate dictionaries for atom counts (solo = 1) and atom-pair counts (solo = 0)
            occur = {}
            occur_pro = {}
            occur_lig = {}
            for tp in quasi_type:
                occur[tp] = 0
            for tp in pro_pool:
                occur_pro[tp] = 0
            for tp in lig_pool:
                occur_lig[tp] = 0
            # check the contacts one by one ----------------------------------
            contacts = np.nonzero((self.pd >= cbin[0]) & (self.pd < cbin[1]))
            conts = [(int(i), int(j)) for (i, j) in zip(contacts[0], contacts[1])]
            for cont in conts:
                atm1 = self.pro[1].GetAtomWithIdx(cont[0])
                atm2 = self.lig[1].GetAtomWithIdx(cont[1])                
                if md_type == 'onionnet':
                    atm1_an = atNumDict_onnt.get(atm1.GetAtomicNum(), -8)
                    atm2_an = atNumDict_onnt.get(atm2.GetAtomicNum(), -8)
                elif md_type == 'ecif':
                    atm1_an = (atm1.GetAtomicNum(), atm1.GetExplicitValence(), atm1.GetTotalDegree() - atm1.GetTotalNumHs(), 
                               atm1.GetTotalNumHs(), int(atm1.GetIsAromatic()), pro_ringatoms_dict.get(cont[0], 0))
                    atm2_an = (atm2.GetAtomicNum(), atm2.GetExplicitValence(), atm2.GetTotalDegree() - atm2.GetTotalNumHs(), 
                               atm2.GetTotalNumHs(), int(atm2.GetIsAromatic()), lig_ringatoms_dict.get(cont[1], 0))  
                else:
                    atm1_an = atm1.GetAtomicNum()
                    atm2_an = atm2.GetAtomicNum()
                
                if solo == 0:
                    tmp = (atm1_an, atm2_an)
                    if md_type == 'ecif' and reftypes is None:
                        if atm1_an[0] in at_filt_ecif and atm2_an[0] in at_filt_ecif:
                            if tmp not in occur:
                                occur[tmp] = 1
                            else:
                                occur[tmp] += 1
                    else:
                        # for rfscore and oninonnet and ecif with reftypes -----------------------------------------------
                        if tmp in quasi_type:
                            occur[tmp] += 1
                elif solo in [1, 100]:
                    if md_type == 'ecif' and reftypes is None:
                        if atm1_an[0] in at_filt_ecif:
                            if atm1_an not in occur_pro:
                                occur_pro[atm1_an] = 1
                            else:
                                occur_pro[atm1_an] += 1
                        
                        if atm2_an[0] in at_filt_ecif:
                            if atm2_an not in occur_lig:
                                occur_lig[atm2_an] = 1
                            else:
                                occur_lig[atm2_an] += 1
                    else:
                        if atm1_an in pro_pool:
                            occur_pro[atm1_an] += 1
                        if atm2_an in lig_pool:
                            occur_lig[atm2_an] += 1
                else:
                    tmp = (atm1_an, atm2_an)
                    if md_type == 'ecif' and reftypes is None:
                        if atm1_an[0] in at_filt_ecif and atm2_an[0] in at_filt_ecif:
                            if tmp not in occur:
                                occur[tmp] = 1
                            else:
                                occur[tmp] += 1
                                
                        if atm1_an[0] in at_filt_ecif:
                            if atm1_an not in occur_pro:
                                occur_pro[atm1_an] = 1
                            else:
                                occur_pro[atm1_an] += 1
                        
                        if atm2_an[0] in at_filt_ecif:
                            if atm2_an not in occur_lig:
                                occur_lig[atm2_an] = 1
                            else:
                                occur_lig[atm2_an] += 1
                    else:
                        if tmp in quasi_type:
                            occur[tmp] += 1
                        if atm1_an in pro_pool:
                            occur_pro[atm1_an] += 1
                        if atm2_an in lig_pool:
                            occur_lig[atm2_an] += 1
                    
            if solo == 0:
                quasi_feat += list(occur.values()) 
                quasi_feat_types += list(occur.keys())
            elif solo == 1:
                quasi_feat += (list(occur_pro.values()) + list(occur_lig.values()))
                quasi_feat_types_pro += list(occur_pro.keys())
                quasi_feat_types_lig += list(occur_lig.keys())
            elif solo == 100:
                quasi_feat += [(list(occur_pro.values()) + list(occur_lig.values()))]
                quasi_feat_types_pro += list(occur_pro.keys())
                quasi_feat_types_lig += list(occur_lig.keys())
            else:
                quasi_feat += (list(occur.values()) + list(occur_pro.values()) + list(occur_lig.values()))
                quasi_feat_types += list(occur.keys())
                quasi_feat_types_pro += list(occur_pro.keys())
                quasi_feat_types_lig += list(occur_lig.keys())                
        
        rtn_reftypes = quasi_feat_types if solo == 0 else ([quasi_feat_types_pro, quasi_feat_types_lig] if solo in [1, 100] else [quasi_feat_types, [quasi_feat_types_pro, quasi_feat_types_lig]])
        
        return quasi_feat, rtn_reftypes  

    def get_quasi_fragmental_desp_ext(self, bins = None):
        """
        Compute extended quasi fragmental descriptors.
        Parameters:
            bins - distance bins for extracting qf descriptors
        """
        if bins is not None:
            self.contact_bins = bins
        else:
            self.contact_bins = [(0, 12)]

        # list atom types ----------------------------------------------------
        pro_pool = [6, 7, 8, 16]
        lig_pool = [6, 7, 8, 9, 15, 16, 17, 35, 53]
        # get list of atom-pair types ----------------------------------------
        quasi_type = list(itertools.product(pro_pool, lig_pool))        
        quasi_feat = []

        for cbin in self.contact_bins:
            occur = {}
            for tp in quasi_type:
                occur[tp] = [0, []]
            # check the contacts one by one ----------------------------------
            contacts = np.nonzero((self.pd >= cbin[0]) & (self.pd < cbin[1]))
            conts = [(int(i), int(j)) for (i, j) in zip(contacts[0], contacts[1])]
            distances = [self.pd[i, j] for (i, j) in conts]
            for ind in range(len(conts)):
                cont = conts[ind]
                cur_dist = distances[ind]
                atm1 = self.pro[1].GetAtomWithIdx(cont[0])
                atm2 = self.lig[1].GetAtomWithIdx(cont[1])                
                atm1_an = atm1.GetAtomicNum()
                atm2_an = atm2.GetAtomicNum()
                
                tmp = (atm1_an, atm2_an)
                if tmp in quasi_type:
                    occur[tmp][0] += 1
                    occur[tmp][1] += [cur_dist]
            
            for tp in quasi_type:
                if occur[tp][0] == 0:
                    quasi_feat += [0, 0]
                else:
                    quasi_feat += [occur[tp][0], mean(occur[tp][1])]
        
        return quasi_feat  

    def get_quasi_fragmental_desp_ext2(self, bins = None):
        """
        Compute extended quasi fragmental descriptors.
        Parameters:
            bins - distance bins for extracting qf descriptors
        """
        if bins is not None:
            self.contact_bins = bins
        else:
            self.contact_bins = [(0, 12)]

        # list atom types ----------------------------------------------------
        atNumDict_onnt = {6: -1, 7: -2, 8: -3, 1: -4, 15: -5, 16: -6, 9: -7, 17: -7, 35: -7, 53: -7} 
        pro_pool = [-a for a in np.arange(1, 9)]
        lig_pool = pro_pool
        # get list of atom-pair types ----------------------------------------
        quasi_type = list(itertools.product(pro_pool, lig_pool))        
        quasi_feat = []

        for cbin in self.contact_bins:
            occur = {}
            for tp in quasi_type:
                occur[tp] = [0, []]
            # check the contacts one by one ----------------------------------
            contacts = np.nonzero((self.pd >= cbin[0]) & (self.pd < cbin[1]))
            conts = [(int(i), int(j)) for (i, j) in zip(contacts[0], contacts[1])]
            distances = [self.pd[i, j] for (i, j) in conts]
            for ind in range(len(conts)):
                cont = conts[ind]
                cur_dist = distances[ind]
                atm1 = self.pro[1].GetAtomWithIdx(cont[0])
                atm2 = self.lig[1].GetAtomWithIdx(cont[1])                
                atm1_an = atNumDict_onnt.get(atm1.GetAtomicNum(), -8)
                atm2_an = atNumDict_onnt.get(atm2.GetAtomicNum(), -8)
                
                tmp = (atm1_an, atm2_an)
                if tmp in quasi_type:
                    occur[tmp][0] += 1
                    occur[tmp][1] += [cur_dist]
            
            for tp in quasi_type:
                if occur[tp][0] == 0:
                    quasi_feat += [0, 0]
                else:
                    quasi_feat += [occur[tp][0], mean(occur[tp][1])]
        
        return quasi_feat  
      
    def compute_ifp_idfs(self,
                         ifptype = 'ecfp', 
                         ecfp_radius = [1, 1],
                         base_prop = ['AtomicNumber', 'TotalConnections', 'HCount', 'HeavyNeighborCount', 'FormalCharge'],
                         folding_para = {'power': np.arange(6, 7, 1), 'counts': 0}):
        """
        Computes interaction fingerprint identifiers for protein-ligand complexes, in order to construct IFPs.
        Parameters:
            ifptype - type of interaction fingerprint ('ecfp', 'splif' or 'plec')
            ecfp_radius - radii for generating ECFP fragments for the protein and ligand
            base_prop - a list of atom properties for calculating the initial identifiers of heavy atoms
            folding_para - parameters for folding the identifiers, with 'power' indicating the length and 'counts' indicating whether to use occurences or not
        """
        if ifptype not in ['ecfp', 'splif', 'plec']:
            print('Wrong interaction fingerprint type! Please provide mode for computing ifp!')
            return []
        print('Compute ifp identifiers for the protein-ligand complex...')
        ifp = {}
        mols = (self.pro, self.lig)   
        identifiers = getECFPidentifiers_molpair(mols_info = mols, 
                                                 prop = base_prop,
                                                 contacts = self.cont,
                                                 degrees = ecfp_radius,
                                                 ifptype = ifptype)          

        if ifptype in ['ecfp']:
            for pr in folding_para['power']:
                ifp[pr] = np.concatenate([ifp_folding(identifiers = tmp, 
                                                      channel_power = pr, 
                                                      counts = folding_para['counts']) for tmp in identifiers])  
        else:
            for pr in folding_para['power']:
                ifp[pr] = ifp_folding(identifiers = identifiers,
                                  channel_power = pr,
                                  counts = folding_para['counts'])
                         
        return ifp

