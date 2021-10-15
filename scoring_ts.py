#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 20:58:12 2021

@author: debbywang
"""
import logging
import numpy as np
import pandas as pd
import multiprocessing
from SIFt import SIFt
import random
import os
import time
import deepchem as dc
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE
import tensorflow as tf


def _featurize_complex(pdbinfo, log_message, para):
    """Featurizes a complex.
    First initializes an SIFt object, and then calculates the intlst.
    """   
    logging.info(log_message)
    try:
        sift = SIFt(fn_pro_PDB = pdbinfo['fn_pro_PDB'], fn_lig_PDB = pdbinfo['fn_lig_PDB'],
                    fn_pro_MOL2 = pdbinfo['fn_pro_MOL2'], fn_lig_MOL2 = pdbinfo['fn_lig_MOL2'], 
                    ID = pdbinfo['pdbid'], addH = para['addH'], sant = para['sant'], 
                    int_cutoff = para['cutoff'])
        if para['ifp_type'] in ['sift', 'kbifp', 'apif', 'pharmif']:
            sift.find_int()
            if len(sift.intlst['contact']) == 0:
                return None
            else:
                if para['ifp_type'] == 'apif':
                    return sift.construct_APIF(count = para['count'])
                elif para['ifp_type'] == 'pharmif':
                    return sift.construct_PharmIF()
                else:
                    return sift
        elif para['ifp_type'] in ['rfscore', 'onionnet']:
            return sift.get_quasi_fragmental_desp(bins = para['bins'], 
                                                  md_type = para['ifp_type'], 
                                                  reftypes = None, 
                                                  solo = para['solo'])[0]
        elif para['ifp_type'] == 'ecif':
            if len(para['lst']) == 0:
                return sift.get_quasi_fragmental_desp(bins = para['bins'],
                                                      md_type = para['ifp_type'],
                                                      reftypes = None,
                                                      solo = para['solo'])
            else:
                return sift.get_quasi_fragmental_desp(bins = para['bins'], 
                                                      md_type = para['ifp_type'], 
                                                      reftypes = para['lst'], 
                                                      solo = para['solo'])[0]
        elif para['ifp_type'] == 'rfscore_ext':
            return sift.get_quasi_fragmental_desp_ext(bins = para['bins'])
        elif para['ifp_type'] == 'rfscore_ext2':
            return sift.get_quasi_fragmental_desp_ext2(bins = para['bins'])
        else:
            ifp = sift.compute_ifp_idfs(ifptype = para['ifp_type'], 
                                        ecfp_radius = para['ecfp_radius'],
                                        base_prop = para['base_prop'],
                                        folding_para = para['folding_para'])    

            return ifp
    except:
        return None

  
def featurize_complexes(ligand_pdbfiles, 
                        protein_pdbfiles,
                        ligand_mol2files, 
                        protein_mol2files,
                        pdbids,
                        para = {'addH': True, 'sant': True,
                                'cutoff': 4.5, 'includeH': False, 'ifp_type': 'sift', 'count': 1,
                                'intlst': ['contact', 'backbone', 'sidechain', 'polar', 'nonpolar', 'donor', 'acceptor'],
                                'inttype': 'CH'}
#                        para = {'addH': True, 'sant': True,
#                                'cutoff': 4.5, 'ifp_type': 'rfscore',
#                                'bins': None, # use default contact bins
#                                'solo': 0, 'lst': []}
#                        para = {'addH': True, 'sant': False,
#                                'cutoff': 4.5, 'ifp_type': 'splif',
#                                'ecfp_radius': [1, 1],
#                                'base_prop': ['AtomicNumber', 'TotalConnections', 'HCount', 'HeavyNeighborCount', 'FormalCharge'],
#                                'folding_para': {'power': np.arange(6, 8, 1), 'counts': 1}}
                        ):
    """Obtains SIFts of a group of complexes.
    Parameters:
        ligand_pdbfiles - a list of ligand pdb files
        protein_pdbfiles - a list of protein pdb files
        ligand_mol2files - a list of ligand mol2 files
        protein_mol2files - a list of protein mol2 files
        pdbids - a list of pdb ids for the complexes under processing
        para - parameters for constructing SIFts
    Returns a list of the SIFts and the indices of failed complexes.
    """
    pool = multiprocessing.Pool(processes = 15)
    results = []
    feat = []
    failures = []
    
    info = zip(ligand_pdbfiles, protein_pdbfiles, ligand_mol2files, protein_mol2files, pdbids)
    for i, (lig_pdbfile, pro_pdbfile, lig_mol2file, pro_mol2file, pdbid) in enumerate(info):
        log_message = "Featurizing %d / %d complex..." % (i, len(pdbids))
        pdbinfo = {'fn_pro_PDB': pro_pdbfile, 'fn_lig_PDB': lig_pdbfile, 
                   'fn_pro_MOL2': pro_mol2file, 'fn_lig_MOL2': lig_mol2file, 'pdbid': pdbid}
        results.append(pool.apply_async(_featurize_complex, (pdbinfo, log_message, para)))      
    pool.close()  
    
    for ind, result in enumerate(results):
        new_sift = result.get()
        if new_sift is None:
            failures.append(ind)
        else:            
            feat.append(new_sift)
    
    return feat, failures
    
def generate_dt(feat,
                para = {'addH': True, 'sant': True,
                        'cutoff': 4.5, 'includeH': False, 'ifp_type': 'sift', 'count': 1,
                        'intlst': ['contact', 'backbone', 'sidechain', 'polar', 'nonpolar', 'donor', 'acceptor'],
                        'inttype': 'CH'},
#                para = {'addH': True, 'sant': True,
#                        'cutoff': 4.5, 'ifp_type': 'rfscore',
#                        'bins': None, # use default contact bins
#                        'solo': 0. 'lst': []}
#                para = {'addH': True, 'sant': False,
#                        'cutoff': 4.5, 'ifp_type': 'splif',
#                        'ecfp_radius': [1, 1],
#                        'base_prop': ['AtomicNumber', 'TotalConnections', 'HCount', 'HeavyNeighborCount', 'FormalCharge'],
#                        'folding_para': {'power': np.arange(6, 8, 1), 'counts': 1}},
                lst = []):
    """Featurizes a group of complexes using SIFts.
    Parameters:
        feat - a list of SIFt objects
        cutoff - distance cutoff for protein-ligand interactions, default is 4.5 Angstrom
        includeH - whether to include hydrogen atoms when identifying protein-ligand interactions
        ifp_type - type of SIFt, including 'sift', 'kbifp', 'apif', 'pharmif', 'rfscore', 'onionnet', 'ecif', 'splif', 'plec' and 'ecfp'
        count - whether to use the occurences of bits in ifps (applicable for 'sift', 'kbifp' and 'apif')
        intlst - a list of interaction types for constructing 'sift'
        inttype - type of 'kbifp'
        lst - a list of residues or atoms for generating ifps (applicable for 'sift' and 'kbifp') or a list of atom types for generating ecif fp
    Returns an array of computed features (n' x m, where n' = n - f and m = length of sift) 
    and an index list of failed complexes (length of f)
    """
    reslst = []
    atmlst = []
    atprtplst = []
    attplst = [[], []]
    reflst = []
    features = []
    features_dict = {}
    if len(feat) == 0:
        return features, []
    
    if para['ifp_type'] in ['sift', 'kbifp']:
        if len(lst) == 0:
            for sift in feat:
                if para['ifp_type'] == 'sift':
                    reslst += sift.reslst
                else:
                    atmlst += sift.atmlst
            
            if para['ifp_type'] == 'sift':
                reflst = list(set(reslst))
            else:
                reflst = list(set(atmlst))
        else:
            reflst = lst
        
        for sift in feat:
            if para['ifp_type'] == 'sift':
                features.append(sift.construct_SIFt(reslst = reflst, intlst = para['intlst'], count = para['count']))
            else:
                features.append(sift.construct_KB_IFP(atmlst = reflst, inttype = para['inttype'], count = para['count']))

        if len(features) > 0:
            return np.vstack(features), reflst
        else:
            return features, reflst
    elif para['ifp_type'] == 'ecif':
        if len(para['lst']) == 0:
            for sift in feat:
                if para['solo'] == 0:
                    atprtplst += sift[1]
                elif para['solo'] == 1:
                    attplst[0] += sift[1][0]
                    attplst[1] += sift[1][1]          
                else:
                    atprtplst += sift[1][0]
                    attplst[0] += sift[1][1][0]
                    attplst[1] += sift[1][1][1]                       
            
            reflst = list(set(atprtplst)) if para['solo'] == 0 else ([list(set(attplst[0])), list(set(attplst[1]))] if para['solo'] == 1 else [list(set(atprtplst)), [list(set(attplst[0])), list(set(attplst[1]))]])
            return [], reflst
        else:       
            for sift in feat:
                features.append(sift)
            if len(features) > 0:
                return np.vstack(features), reflst
            else:
                return features, reflst
    elif para['ifp_type'] in ['apif', 'pharmif', 'rfscore', 'onionnet', 'acfm', 'aifm', 'rfscore_ext', 'rfscore_ext2']:
        for sift in feat:
            features.append(sift)
        if len(features) > 0:
#            return np.vstack(features), reflst
            return np.array(features), reflst
        else:
            return features, reflst   
    else:
        for pr in para['folding_para']['power']:
            features_dict[pr] = []
        for sift in feat:
            for pr in para['folding_para']['power']:
                features_dict[pr].append(sift[pr])
        for pr in para['folding_para']['power']:
            features_dict[pr] = np.vstack(features_dict[pr])
        return features_dict, []       


def loaddata_fromPDBbind_v2020(data_dir,
                               subset = "refined",
                               select_target = ['HIV-1 PROTEASE'],
                               randsplit_ratio = [0.5, 0.25, 0.25],
                               para = {'addH': True, 'sant': True,
                                       'cutoff': 4.5, 'ifp_type': 'rfscore',
                                       'bins': None, 'lst': [], # use default contact bins
                                       'solo': 0},
                               rand_seed = 123):
    """Load data from PDBbind sets or subsets.
    Parameters:
        data_dir - folder that stores PDBbind data
        subset - supports PDBbind refined set ('refined'), core set ('cs') and refined minus core set ('rs-cs')
        select_target - whether to select specific targets from the subset (['HIV-1 PROTEASE'], ['CARBONIC ANHYDRASE 2'] or None)
        ransplit_ratio - ratio for patition the set into training, validation and test sets
        para - a dictionary of parameters for computing ifp features (align with generate_dt function)
        rand_seed - random seed for splits
    Returns (training data, test data) from the deepchem library
    """
    if subset not in ['refined', 'casf2016', 'csarhiqS1', 'csarhiqS2', 'csarhiqS3', 'rs-casf2016', 'rs-casf2016-csarhiq']:
        print('Wrong subset!!!')
        return []
    elif subset in ['casf2016', 'csarhiqS1', 'csarhiqS2', 'csarhiqS3']:
        if select_target is not None:
            print('Wrong combination of subset and select_target!!!')
            return []       
    
    dt_dir = data_dir
    os.chdir(dt_dir)
    
    # -----------------------------------------------------------------------------------------------------------------------
    # get folder containing the structural data and the index dataframe containing the pdbs/affinities
    # -----------------------------------------------------------------------------------------------------------------------
    data_folder_dict = {'refined': 'v2020/PDBbind_v2020_refined/refined-set/'}
    data_index_dict = {'refined': 'indexes/rs_index.csv'}
    for s in ['casf2016', 'csarhiqS1', 'csarhiqS2', 'csarhiqS3']:
        data_folder_dict[s] = 'ValidationSets/' + s + '/'
        data_index_dict[s] = 'indexes/' + s + '_index.csv'
    index_rmlst = {'rs-casf2016': ['casf2016'], 
                   'rs-casf2016-csarhiq': ['casf2016', 'csarhiqS1', 'csarhiqS2', 'csarhiqS3']}
    if subset in ['casf2016', 'csarhiqS1', 'csarhiqS2', 'csarhiqS3']:
        cur_fd = data_folder_dict[subset]
        df_index = pd.read_csv(data_index_dict[subset])
    else:
        cur_fd = data_folder_dict['refined']
        df_index_all = pd.read_csv(data_index_dict['refined'])
        if subset != 'refined':
            for rmlst in index_rmlst[subset]:
                rm_index = pd.read_csv(data_index_dict[rmlst])['id'].tolist()
                df_index_all = df_index_all.loc[~df_index_all['id'].isin(rm_index)]
        df_index = df_index_all.loc[df_index_all['name'].isin(select_target)] if select_target is not None else df_index_all
        
    # further filter the dataframe according to select_target
    pdbs_selected = df_index['id'].tolist()
    labels_selected = df_index['affinity'].tolist()
            
    # obtain the pdb and mol2 filename lists for the complexes -----------------------------------------------------------------------
    print('Generate file names................................................................')
    protein_pdbfiles = []
    ligand_pdbfiles = []
    protein_mol2files = []
    ligand_mol2files = []
    for pdb in pdbs_selected:
        protein_pdbfiles += [os.path.join(dt_dir, cur_fd, pdb, "%s_protein.pdb" % pdb)]
        protein_mol2files += [os.path.join(dt_dir, cur_fd, pdb, "%s_protein.mol2" % pdb)]
        ligand_pdbfiles += [os.path.join(dt_dir, cur_fd, pdb, "%s_ligand.pdb" % pdb)]
        ligand_mol2files += [os.path.join(dt_dir, cur_fd, pdb, "%s_ligand.mol2" % pdb)]  
        
    # featurize complexes using SIFts and split them into train, validation and test sets --------------------------------------------
    print('Begin to featurize dataset....................................................................')
    feat_t1 = time.time()
    feat, flrs = featurize_complexes(ligand_pdbfiles = ligand_pdbfiles, 
                                     protein_pdbfiles = protein_pdbfiles,
                                     ligand_mol2files = ligand_mol2files, 
                                     protein_mol2files = protein_mol2files,
                                     pdbids = pdbs_selected,
                                     para = para)
    # Delete labels and ids for failing elements
    labels_lft = np.delete(labels_selected, flrs)
    labels_lft = labels_lft.reshape((len(labels_lft), 1))
    ids_lft = np.delete(pdbs_selected, flrs)
    # splitting 
    print('Begin to split dataset and generate data for subsequent machine-learning prediction...............................')
    random.seed(rand_seed)
    sz = len(ids_lft)
    train_randids = random.sample(range(0, sz), round(sz * randsplit_ratio[0]))
    train_randids.sort()
    ids_valtest = [i for i in range(0, sz) if i not in train_randids]
    valsize = min(round(sz * randsplit_ratio[1]), len(ids_valtest))
    validation_randids = random.sample(ids_valtest, valsize)
    validation_randids.sort()
    test_randids = [i for i in ids_valtest if i not in validation_randids]
    test_randids.sort()    
    np_feat = np.array(feat)
    rtn_lst = []
    if para['ifp_type'] == 'ecif':
        if len(para['lst']) == 0:
            ept, unitlsts = generate_dt(feat = np_feat[train_randids], para = para, lst = [])
            return [], [], [], unitlsts
        else:
            dt_train, t0 = generate_dt(feat = np_feat[train_randids], para = para, lst = [])
            dt_validation, t1 = generate_dt(feat = np_feat[validation_randids], para = para, lst = [])
            dt_test, t2 = generate_dt(feat = np_feat[test_randids], para = para, lst = [])
    else:
        dt_train, unitlsts = generate_dt(feat = np_feat[train_randids], para = para, lst = [])
        dt_validation, t1 = generate_dt(feat = np_feat[validation_randids], para = para, lst = unitlsts)
        dt_test, t2 = generate_dt(feat = np_feat[test_randids], para = para, lst = unitlsts)

    print('%d pdbs are selected, %d pdbs pass the featurization step' % (len(pdbs_selected), len(ids_lft)))

    if para['ifp_type'] in ['sift', 'kbifp', 'apif', 'pharmif', 'rfscore', 'onionnet', 'ecif', 'acfm', 'aifm', 'rfscore_ext', 'rfscore_ext2']:
        if randsplit_ratio[0] > 0:
            train = dc.data.NumpyDataset(X = dt_train, y = labels_lft[train_randids], ids = ids_lft[train_randids])
        else:
            train = []
        if randsplit_ratio[1] > 0:
            validation = dc.data.NumpyDataset(X = dt_validation, y = labels_lft[validation_randids], ids = ids_lft[validation_randids])
        else:
            validation = []
        if randsplit_ratio[2] > 0:
            test = dc.data.NumpyDataset(X = dt_test, y = labels_lft[test_randids], ids = ids_lft[test_randids])
        else:
            test = []
        feat_t2 = time.time()
        print('Featurization and splitting cost %f s...' % (feat_t2 - feat_t1))
        return (train, validation, test, rtn_lst)
    else:
        res_dict = {}
        for pr in para['folding_para']['power']:
            if randsplit_ratio[0] > 0:
                train = dc.data.NumpyDataset(X = dt_train[pr], y = labels_lft[train_randids], ids = ids_lft[train_randids])
            else:
                train = []
            if randsplit_ratio[1] > 0:
                validation = dc.data.NumpyDataset(X = dt_validation[pr], y = labels_lft[validation_randids], ids = ids_lft[validation_randids])
            else:
                validation = []
            if randsplit_ratio[2] > 0:
                test = dc.data.NumpyDataset(X = dt_test[pr], y = labels_lft[test_randids], ids = ids_lft[test_randids])
            else:
                test = []
            res_dict[pr] = (train, validation, test) 
        return res_dict
    

def create_model(tp = 'rf', 
                 rand = 123, 
                 para = {'rf_n_estimators': 50, 'tree_max_depth': 5, 'gb_n_estimators': 50, 'nn_max_iter': 100, 'nn_hidlaysize': (50, )}):
    if tp == 'rf':
        sklearn_model = RandomForestRegressor(random_state = rand, n_estimators = para['rf_n_estimators'])
    elif tp == 'lm':
        sklearn_model = LinearRegression()
    elif tp == 'tree':
        sklearn_model = DecisionTreeRegressor(random_state = rand, max_depth = para['tree_max_depth'])
    elif tp == 'gb':
        sklearn_model = GradientBoostingRegressor(random_state = rand, n_estimators = para['gb_n_estimators'])
    elif tp == 'nn':
        sklearn_model = MLPRegressor(random_state = rand, max_iter = para['nn_max_iter'], hidden_layer_sizes = para['nn_hidlaysize'])
    elif tp == 'voting':
        reg1 = GradientBoostingRegressor(random_state = rand, n_estimators = para['gb_n_estimators'])
        reg2 = RandomForestRegressor(random_state = rand, n_estimators = para['rf_n_estimators'])
        reg3 = DecisionTreeRegressor(random_state = rand, max_depth = para['tree_max_depth'])
        sklearn_model = VotingRegressor(estimators = [('gb', reg1), ('rf', reg2), ('tree', reg3)])
    else:
        print('Wrong model type!!!')
        return []
    
    if tp == 'nn':
        model = dc.models.SklearnModel(sklearn_model, use_weights = False)
    else:
        model = dc.models.SklearnModel(sklearn_model)
    return model


def corr_true_pred(y_true, y_pred):    
    mx = tf.math.reduce_mean(y_true)
    my = tf.math.reduce_mean(y_pred)
    xm, ym = y_true - mx, y_pred - my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num/r_den

def cos_loss_pc(y_true, y_pred):    
    mx = tf.math.reduce_mean(y_true)
    my = tf.math.reduce_mean(y_pred)
    xm, ym = y_true - mx, y_pred - my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return 1 - r_num/r_den

def cus_loss(y_true, y_pred, alpha = 0.8):    
    mx = tf.math.reduce_mean(y_true)
    my = tf.math.reduce_mean(y_pred)
    xm, ym = y_true - mx, y_pred - my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    evl_pc = r_num/r_den
    evl_rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))
    loss = alpha * (1 - evl_pc) + (1 - alpha) * evl_rmse
    return loss

def create_DL_model(train_dt, sz = (60, 64, 1), valid_ratio = 0.1):
    train_dt_X = train_dt.X.reshape((train_dt.X.shape[0], ) + sz)
    train_dt_y = train_dt.y
    X_train, X_valid, y_train, y_valid = train_test_split(train_dt_X, train_dt_y, test_size = valid_ratio)
    model = Sequential([Conv2D(input_shape = sz, 
                               filters = 32, kernel_size = (4, 4), strides=(1, 1), activation = 'relu', 
                               kernel_regularizer = tf.keras.regularizers.L2(0.001)),
                        Conv2D(filters = 64, kernel_size = (4, 4), strides=(1, 1), activation = 'relu',
                               kernel_regularizer = tf.keras.regularizers.L2(0.001)),
                        Conv2D(filters = 128, kernel_size = (4, 4), strides=(1, 1), activation = 'relu',
                               kernel_regularizer = tf.keras.regularizers.L2(0.001)),
                        Flatten(),
                        Dense(units = 400, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2(0.001)),
                        Dense(units = 200, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2(0.001)),
                        Dense(units = 100, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2(0.001)),
                        Dense(units = 1)])
    
    model.compile(loss = cus_loss,
                  optimizer = 'adam',
#                  optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9, decay = 0.000001),
                  metrics = [RMSE(), corr_true_pred])
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.01, patience = 40)
    model.fit(X_train, y_train, 
              epochs = 100, 
              batch_size = 128, 
              validation_data = (X_valid, y_valid), 
              callbacks=[es])
    return model

def create_DL_model2(train_dt, sz = (60, 64, 1), valid_ratio = 0.1):
    train_dt_X = train_dt.X.reshape((train_dt.X.shape[0], ) + sz)
    train_dt_y = train_dt.y
    X_train, X_valid, y_train, y_valid = train_test_split(train_dt_X, train_dt_y, test_size = valid_ratio)
    model = Sequential([Conv2D(input_shape = sz, 
                               filters = 32, kernel_size = (4, 4), strides=(1, 1), activation = 'relu', 
                               kernel_regularizer = tf.keras.regularizers.L2(0.001)),
                        Conv2D(filters = 64, kernel_size = (4, 4), strides=(1, 1), activation = 'relu',
                               kernel_regularizer = tf.keras.regularizers.L2(0.001)),
                        Conv2D(filters = 128, kernel_size = (4, 4), strides=(1, 1), activation = 'relu',
                               kernel_regularizer = tf.keras.regularizers.L2(0.001)),
                        Flatten(),
                        Dropout(0.5),
                        Dense(units = 400, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2(0.001)),
                        Dropout(0.2),
                        Dense(units = 200, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2(0.001)),
                        Dropout(0.2),
                        Dense(units = 100, activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2(0.001)),
                        Dropout(0.2),
                        Dense(units = 1)])
    
    model.compile(loss = cus_loss,
                  optimizer = 'adam',
#                  optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9, decay = 0.000001),
                  metrics = [RMSE(), corr_true_pred])
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.01, patience = 40)
    model.fit(X_train, y_train, 
              epochs = 100, 
              batch_size = 128, 
              validation_data = (X_valid, y_valid), 
              callbacks=[es])
    return model

    
def validate_DL_model(mdl, vset, sz = (60, 64, 1)):
    test_dt_X = vset.X.reshape((vset.X.shape[0], ) + sz)
    test_dt_y = vset.y
    test_scores = mdl.evaluate(test_dt_X, test_dt_y)
    res = {'PC': test_scores[2], 'rmse': test_scores[1]}
    return res

def validate_model(mdl, vset, metrics = [dc.metrics.Metric(dc.metrics.pearson_r2_score), dc.metrics.Metric(mean_squared_error)]):
    test_scores = mdl.evaluate(vset, metrics, transformers = [])
    res = {'PC': sqrt(test_scores['pearson_r2_score']), 'rmse': sqrt(test_scores['mean_squared_error'])}
    return res
            