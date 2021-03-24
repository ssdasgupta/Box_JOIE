from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), './src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
#sys.path.append('./src')
import os
if not os.path.exists('./tl_results'):
    os.makedirs('./tl_results')

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import time
import multiG  
import model2 as model
from tester2 import Tester
import argparse

# all parameter required
parser = argparse.ArgumentParser(description='JOIE Testing: Type Linking')
parser.add_argument('--modelname', type=str,help='model category')
parser.add_argument('--model', type=str,help='model name including data and model')
parser.add_argument('--testfile', type=str,help='test data')
parser.add_argument('--resultfolder', type=str,help='result output folder')
parser.add_argument('--GPU', type=str, default='0' ,help='GPU Usage')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

path_prefix = args.modelname
hparams_str = args.model
args.method, args.bridge =  hparams_str.split('_')[0], hparams_str.split('_')[1]
args.int_method = hparams_str.split('_')[4]
model_file = path_prefix+"/"+hparams_str+"/"+args.method+'-model-m2.ckpt'
data_file = path_prefix+"/"+hparams_str+"/"+args.method+'-multiG-m2.bin'
test_data = args.testfile
limit_align_file = None
result_folder = './'+args.resultfolder+'/'+args.modelname
result_file = result_folder+"/"+hparams_str+'_result.txt'

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

topK = 10
max_check = 100000

#dup_set = set([])
#for line in open(old_data):
#    dup_set.add(line.rstrip().split('@@@')[0])
tester = Tester()
tester.build(save_path = model_file,
             data_save_path = data_file,
             method=args.method,
             bridge=args.bridge,
             int_method=args.int_method
             )
tester.load_test_type(test_data, splitter = '\t', line_end = '\n')


#tester.load_except_data(except_data, splitter = '@@@', line_end = '\n')

test_id_limits = None
if limit_align_file is not None:
    _, test_id_limits = tester.load_align_ids(limit_align_file, splitter = '\t', line_end = '\n')

import multiprocessing
from multiprocessing import Process, Value, Lock, Manager

cpu_count = multiprocessing.cpu_count()

manager = Manager()
index = Value('i', 0, lock=True) #index
rst_predict = manager.list() #scores for each case
rank_record = manager.list()
prop_record = manager.list()
t0 = time.time()

def test(tester, index, rst_predict, rank_record, prop_record,verbose=True):
    while index.value < len(tester.test_align) and index.value < 99:
        idx = index.value
        index.value += 1
        if idx > 0 and idx % 200 == 0:
            print("Tested %d in %d seconds." % (idx+1, time.time()-t0))
            try:
                print(np.mean(rst_predict, axis=0))
            except:
                pass
        e1, e2 = tester.test_align[idx]
        #vec_e1 = tester.ent_index2vec(e1, source = 1)
        vec_proj_e1_min, vec_proj_e1_max = tester.projection(e1, source = 1)
        vec_pool_e2_min, vec_pool_e2_max = tester.projection_type_matrix(tester.vec_e[2], np.exp(tester.vec_e[3]))
        rst = tester.kNN(vec_proj_e1_min, vec_proj_e1_max, vec_pool_e2_min, vec_pool_e2_max, topK, limit_ids=test_id_limits)#, except_ids=tester.aligned[2])
        #print distance
        #print([x[1] for x in rst])
        this_hit = []
        hit = 0.0
        strl = tester.ent_index2str(rst[0][0], 2)
        strr = tester.ent_index2str(e2, 2)
        this_index = 0
        this_rank = None
        for pr in rst:
            this_index += 1
            if (hit < 1. and (pr[0] == e2 or pr[0] in tester.lr_map[e1])) or (hit < 1. and tester.ent_index2str(pr[0], 2) == strr):
                hit = 1.
                this_rank = this_index
            this_hit.append(hit)
        hit_first = 0
        if rst[0][0] == e2 or rst[0][0] in tester.lr_map[e1] or strl == strr:
            hit_first = 1
        if this_rank is None:
            this_rank = tester.rank_index_from(vec_proj_e1_min, vec_proj_e1_max, vec_pool_e2_min, vec_pool_e2_max, e2, limit_ids=test_id_limits)#, except_ids=tester.aligned[2])
        if this_rank > max_check:
            continue

        if verbose:
            condition = (this_hit[2] == 0 and this_hit[0] == 0 )
            if condition:
                str_l1, dist_l1 = tester.ent_index2str(rst[0][0], 2), rst[0][1]
                str_l2, dist_l2 = tester.ent_index2str(rst[1][0], 2), rst[1][1]
                str_l3, dist_l3 = tester.ent_index2str(rst[2][0], 2), rst[2][1]
                str_a = tester.ent_index2str(e2, 2)
                str_q = tester.ent_index2str(e1, 1)
                print("##################################")
                print(str_q, str_a)
                print((str_l1, dist_l1),(str_l2, dist_l2),(str_l3, dist_l3))

        rst_predict.append(np.array(this_hit))
        rank_record.append(1.0 / (1.0 * this_rank))
        prop_record.append((hit_first, rst[0][1], strl, strr))

# tester.rel_num_cases

processes = [Process(target=test, args=(tester, index, rst_predict, rank_record, prop_record)) for x in range(1)]
for p in processes:
    p.start()
for p in processes:
    p.join()

mean_rank = np.mean(rank_record)
hits = np.mean(rst_predict, axis=0)

# print out result file
fp = open(result_file, 'w')
fp.write("Mean Rank\n")
fp.write(str(mean_rank)+'\n')
#print(' '.join([str(x) for x in hits]) + '\n')
fp.write("Hits@"+str(topK)+'\n')

fp.write(' '.join([str(x) for x in hits]) + '\n')
fp.close()
