from scoring_ts import loaddata_fromPDBbind_v2020
from scoring_ts import create_DL_model, create_DL_model2, validate_DL_model
import os
import numpy as np
import time
import json

###############################################################################################################
# run experiments                                                                                             #
###############################################################################################################

#############################################################################################################
# 1. generic scoring
#############################################################################################################
sd = 50
np.random.seed(sd)
data_dir = "Path-to-PDBbind/PDBbind"
spr_train = [0.9, 0.1, 0]
spr_test = [1, 0, 0]
trparadict = {'rf': {'rf_n_estimators': np.arange(300, 800, 100).tolist()},
              'gb': {'gb_n_estimators': np.arange(300, 800, 100).tolist()}}
datasets = ['rs-casf2016-csarhiq', 'casf2016', 'csarhiqS1', 'csarhiqS2', 'csarhiqS3']
# 1.1. featurize datasets -----------------------------------------------------------------------------------
#################################### IMCPs #######################################################################
ifptp = 'rfscore_ext2'
cur_cutoff = 30
bn = 8
cur_bins = [(shell * 30 / bn, (shell + 1) * 30 / bn) for shell in range(bn)]
para = {'addH': True, 'sant': False,
        'cutoff': cur_cutoff, 'ifp_type': ifptp,
        'bins': cur_bins}                                         
feat_dt_train = loaddata_fromPDBbind_v2020(data_dir = data_dir, 
                                           subset = datasets[0], 
                                           select_target = None, 
                                           randsplit_ratio = spr_train,
                                           para = para,
                                           rand_seed = sd)
feat_dt_test1 = loaddata_fromPDBbind_v2020(data_dir = data_dir, 
                                           subset = datasets[1], 
                                           select_target = None, 
                                           randsplit_ratio = spr_test,
                                           para = para,
                                           rand_seed = sd)
feat_dt_test2 = loaddata_fromPDBbind_v2020(data_dir = data_dir, 
                                           subset = datasets[2], 
                                           select_target = None, 
                                           randsplit_ratio = spr_test,
                                           para = para,
                                           rand_seed = sd)
feat_dt_test3 = loaddata_fromPDBbind_v2020(data_dir = data_dir, 
                                           subset = datasets[3], 
                                           select_target = None, 
                                           randsplit_ratio = spr_test,
                                           para = para,
                                           rand_seed = sd)
feat_dt_test4 = loaddata_fromPDBbind_v2020(data_dir = data_dir, 
                                           subset = datasets[4], 
                                           select_target = None, 
                                           randsplit_ratio = spr_test,
                                           para = para,
                                           rand_seed = sd)

# 1.2 Model parameterization and validation -----------------------------------------------------------------
# results saved in 'res.txt' in the data folder -------------------------------------------------------------
resstr = [json.dumps(para), '\nValidation results: \n']

bin_num = len(cur_bins)
cur_sz_lst = [(bin_num, 64, 2), (bin_num, 128, 1)]

for cur_index in range(len(cur_sz_lst)):
    cur_sz = cur_sz_lst[cur_index]
    resstr += [str(cur_index) + ':\n']
    mymdl = create_DL_model(train_dt = feat_dt_train[0], 
                            sz = cur_sz,
                            valid_ratio = 0.1)
    mymdl2 = create_DL_model2(train_dt = feat_dt_train[0], 
                              sz = cur_sz,
                              valid_ratio = 0.1)
    final_res1 = datasets[1] + ' --- ' + json.dumps(validate_DL_model(mdl = mymdl, vset = feat_dt_test1[0], sz = cur_sz)) + '\n' + json.dumps(validate_DL_model(mdl = mymdl2, vset = feat_dt_test1[0], sz = cur_sz))
    final_res2 = datasets[2] + ' --- ' + json.dumps(validate_DL_model(mdl = mymdl, vset = feat_dt_test2[0], sz = cur_sz)) + '\n' + json.dumps(validate_DL_model(mdl = mymdl2, vset = feat_dt_test2[0], sz = cur_sz))
    final_res3 = datasets[3] + ' --- ' + json.dumps(validate_DL_model(mdl = mymdl, vset = feat_dt_test3[0], sz = cur_sz)) + '\n' + json.dumps(validate_DL_model(mdl = mymdl2, vset = feat_dt_test3[0], sz = cur_sz))
    final_res4 = datasets[4] + ' --- ' + json.dumps(validate_DL_model(mdl = mymdl, vset = feat_dt_test4[0], sz = cur_sz)) + '\n' + json.dumps(validate_DL_model(mdl = mymdl2, vset = feat_dt_test4[0], sz = cur_sz))
    resstr += [final_res1 + '\n' + final_res2 + '\n' + final_res3 + '\n' + final_res4 + '\n']
    
resstr += ['\n\n']
fn = open(os.path.join(data_dir, "res.txt"), "a") 
fn.writelines(resstr) 
fn.close()

time.sleep(300)
