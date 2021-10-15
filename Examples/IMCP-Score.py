from scoring_ts import loaddata_fromPDBbind_v2020, create_model, validate_model
import os
import numpy as np
import time
import itertools
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
ifptp = 'rfscore_ext'
cur_cutoff = 18
cur_bins = [(0, cur_cutoff)]
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
mdllst = ['rf', 'gb']

for mdl in mdllst:
    print('Parameterization and validation using ' + mdl + ' models...........')
    feat_t1 = time.time()
    trparas = []
    trparalst = list(trparadict[mdl].keys())
    tmplsts = list(trparadict[mdl].values())
    if len(trparalst) == 1:
        ky = trparalst[0]
        for vl in tmplsts[0]:
            trparas += [{ky: vl}]
    elif len(trparalst) == 2:
        paraprs = list(itertools.product(tmplsts[0], tmplsts[1]))
        for pair in paraprs :
            trparas += [{trparalst[0]: pair[0], trparalst[1]: pair[1]}]
    elif len(trparalst) == 3:
        paraprs = list(itertools.product(tmplsts[0], tmplsts[1], tmplsts[2]))
        for pair in paraprs :
            trparas += [{trparalst[0]: pair[0], trparalst[1]: pair[1], trparalst[2]: pair[2]}]
    valres = []
    cmdls = []
    
    for tmppara in trparas:
        print('Parameters: ' + json.dumps(tmppara) + '........................')
        cmdl = create_model(tp = mdl, rand = sd, para = tmppara)
        cmdl.fit(feat_dt_train[0])
        cmdls += [cmdl]
        valres += [(validate_model(mdl = cmdl, vset = feat_dt_train[1]))]   
    pcs = [i['PC'] for i in valres]
    bstind = pcs.index(max(pcs))
    final_res1 = datasets[1] + ' --- ' + json.dumps(validate_model(mdl = cmdls[bstind], vset = feat_dt_test1[0]))
    final_res2 = datasets[2] + ' --- ' + json.dumps(validate_model(mdl = cmdls[bstind], vset = feat_dt_test2[0]))
    final_res3 = datasets[3] + ' --- ' + json.dumps(validate_model(mdl = cmdls[bstind], vset = feat_dt_test3[0]))
    final_res4 = datasets[4] + ' --- ' + json.dumps(validate_model(mdl = cmdls[bstind], vset = feat_dt_test4[0]))

    resstr += [mdl + ':\n' + final_res1 + '\n' + final_res2 + '\n' + final_res3 + '\n' + final_res4 + '\n']
    feat_t2 = time.time()
    print('Featurization and splitting cost %f s...' % (feat_t2 - feat_t1))
    
resstr += ['\n\n']
fn = open(os.path.join(data_dir, "res.txt"), "a") 
fn.writelines(resstr) 
fn.close()

time.sleep(300)
