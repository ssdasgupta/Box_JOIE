''' Module for held-out test.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import heapq as HP
import sys

import multiG  
import box_model as model
import trainer2 as trainer
from box_utils import BoxMethods
from numpy import linalg as LA

# This class is used to load and combine a TF_Parts and a Data object, and provides some useful methods for training
class Tester(object):
    def __init__(self):
        self.tf_parts = None
        self.multiG = None
        self.vec_e = {}
        self.vec_r = {}
        self.mat = np.array([0])
        # below for test data
        self.test_align = np.array([0])
        self.test_align_rel = []
        self.aligned = {1: set([]), 2: set([])}
        # L1 to L2 map
        self.lr_map = {}
        self.lr_map_rel = {}
        # L2 to L1 map
        self.rl_map = {}
        self.rl_map_rel = {}
        self.sess = None
        self.softplus = lambda x: np.log1p(np.exp(x))
    
    def build(self, save_path = 'this-model.ckpt', data_save_path = 'this-data.bin', method='transe', bridge='CG'):
        self.multiG = multiG.multiG()
        self.multiG.load(data_save_path)
        self.method = method
        self.bridge = bridge

        self.tf_parts = model.TFParts(num_rels1=self.multiG.KG1.num_rels(),
                                 num_ents1=self.multiG.KG1.num_ents(),
                                 num_rels2=self.multiG.KG2.num_rels(),
                                 num_ents2=self.multiG.KG2.num_ents(),
                                 method=self.method,
                                 bridge=self.bridge,
                                 dim1=self.multiG.dim1,
                                 dim2=self.multiG.dim2,
                                 L1=self.multiG.L1)
        print(self.method,self.bridge) #load
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        
        self.tf_parts._saver.restore(sess, save_path)  # load it
        # if self.tf_parts.bridge == "CMP-double":
        #     value_ht1, value_r1, value_ht2, value_r2, value_Mc, value_bc, value_Me, value_be = sess.run(
        #     [self.tf_parts._ht1_norm, self.tf_parts._r1, self.tf_parts._ht2_norm, self.tf_parts._r2, self.tf_parts._Mc, self.tf_parts._bc, self.tf_parts._Me, self.tf_parts._be])  # extract values.
        #     self._Mc = np.array(value_Mc)
        #     self._bc = np.array(value_bc)
        #     self._Me = np.array(value_Me)
        #     self._be = np.array(value_be)
        #     print(self._Mc.shape, self._bc.shape, self._Me.shape, self._be.shape)
        # else:
        #     value_ht1, value_r1, value_ht2, value_r2, value_M, value_b = sess.run(
        #     [self.tf_parts._ht1_norm, self.tf_parts._r1, self.tf_parts._ht2_norm, self.tf_parts._r2, self.tf_parts._M, self.tf_parts._b])  # extract values.
        #     self.mat = np.array(value_M)
        #     self._b = np.array(value_b)
        #     print(self.mat.shape, self._b.shape)
        value_ht1, value_r1, ht2_min, ht2_delta, value_r2 = sess.run(
            [self.tf_parts._ht1_norm,
             self.tf_parts._r1,
             self.tf_parts._ht2.min_embed,
             self.tf_parts._ht2.delta_embed,
             self.tf_parts._r2])

        self.vec_e[1] = np.array(value_ht1)
        self.vec_e[2] = np.array(ht2_min)
        self.vec_e[3] = np.array(ht2_delta)
        self.vec_r[1] = np.array(value_r1)
        self.vec_r[2] = np.array(value_r2)
        self._ht2 = BoxMethods(ht2_min.shape[1], ht2_min.shape[0])
        print(self.vec_e[1].shape, self.vec_e[2].shape, self.vec_r[1].shape, self.vec_r[2].shape)
        sess.close()


    def load_test_type(self, filename, splitter = '\t', line_end = '\n', dedup=True):
        num_lines = 0
        align = []
        dedup_set = set([])
        for line in open(filename):
            if dedup and line in dedup_set:
                continue
            elif dedup:
                dedup_set.add(line)
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 3:
                continue
            num_lines += 1
            e1 = self.multiG.KG1.ent_str2index(line[0])
            e2 = self.multiG.KG2.ent_str2index(line[2])
            if e1 == None or e2 == None:
                continue
            align.append([e1, e2])
            if self.lr_map.get(e1) == None:
                self.lr_map[e1] = set([e2])
            else:
                self.lr_map[e1].add(e2)
            if self.rl_map.get(e2) == None:
                self.rl_map[e2] = set([e1])
            else:
                self.rl_map[e2].add(e1)
        self.test_align = np.array(align, dtype=np.int32)
        print("Loaded test data from %s, %d out of %d." % (filename, len(align), num_lines))


    def load_test_data_rel(self, filename, splitter = '@@@', line_end = '\n'):
        num_lines = 0
        align = []
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 2:
                continue
            num_lines += 1
            e1 = self.multiG.KG1.rel_str2index(line[0])
            e2 = self.multiG.KG2.rel_str2index(line[1])
            if e1 == None or e2 == None:
                continue
            align.append([e1, e2])
            if self.lr_map_rel.get(e1) == None:
                self.lr_map_rel[e1] = set([e2])
            else:
                self.lr_map_rel[e1].add(e2)
            if self.rl_map_rel.get(e2) == None:
                self.rl_map_rel[e2] = set([e1])
            else:
                self.rl_map_rel[e2].add(e1)
        self.test_align_rel = np.array(align, dtype=np.int32)
        print("Loaded test data (rel) from %s, %d out of %d." % (filename, len(align), num_lines))
                
    def load_except_data(self, filename, splitter = '@@@', line_end = '\n'):
        num_lines = 0
        num_read = 0
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 2:
                continue
            num_lines += 1
            e1 = self.multiG.KG1.ent_str2index(line[0])
            e2 = self.multiG.KG2.ent_str2index(line[1])
            if e1 == None or e2 == None:
                continue
            self.aligned[1].add(e1)
            self.aligned[2].add(e2)
            num_read += 1
        print("Loaded excluded ids from %s, %d out of %d." % (filename, num_read, num_lines))

    def load_align_ids(self, filename, splitter = '@@@', line_end = '\n'):
        num_lines = 0
        num_read = 0
        aligned1, aligned2 = set([]), set([])
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 2:
                continue
            num_lines += 1
            e1 = self.multiG.KG1.ent_str2index(line[0])
            e2 = self.multiG.KG2.ent_str2index(line[1])
            if e1 == None or e2 == None:
                continue
            aligned1.add(e1)
            aligned2.add(e2)
            num_read += 1
        print("Loaded excluded ids from %s, %d out of %d." % (filename, num_read, num_lines))
        return aligned1, aligned2
    
    def load_more_truth_data(self, filename, splitter = '@@@', line_end = '\n'):
        num_lines = 0
        count = 0
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 2:
                continue
            num_lines += 1
            e1 = self.multiG.KG1.ent_str2index(line[0])
            e2 = self.multiG.KG2.ent_str2index(line[1])
            if e1 == None or e2 == None:
                continue
            if self.lr_map.get(e1) == None:
                self.lr_map[e1] = set([e2])
            else:
                self.lr_map[e1].add(e2)
            if self.rl_map.get(e2) == None:
                self.rl_map[e2] = set([e1])
            else:
                self.rl_map[e2].add(e1)
            count += 1
        print("Loaded extra truth data into mappings from %s, %d out of %d." % (filename, count, num_lines))
    
    # by default, return head_mat
    def get_mat(self):
        return self.mat
    
    def ent_index2vec(self, e, source):
        assert (source in set([1, 2]))
        return self.vec_e[source][int(e)]

    def rel_index2vec(self, r, source):
        assert (source in set([1, 2]))
        return self.vec_r[source][int(r)]

    def ent_index2box(self, e, source):
        assert (source in set([1, 2]))
        h_min_embed, h_max_embed = self.vec_e[source][int(e)], self.vec_e[source+1][int(e)]


    def ent_str2vec(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        this_index = KG.ent_str2index(str)
        if this_index == None:
            return None
        return self.vec_e[source][this_index]
    
    def rel_str2vec(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        this_index = KG.rel_str2index(str)
        if this_index == None:
            return None
        return self.vec_r[source][this_index]
    
    class index_dist:
        def __init__(self, index, dist):
            self.dist = dist
            self.index = index
            return
        def __lt__(self, other):
            return self.dist > other.dist
                
    def ent_index2str(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        return KG.ent_index2str(str)
    
    def rel_index2str(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        return KG.rel_index2str(str)

    def ent_str2index(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        return KG.ent_str2index(str)
    
    def rel_str2index(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        return KG.rel_str2index(str)
    
    # input must contain a pool of vecs. return a list of indices and dist
    def kNN(self, vec_min, vec_max, vec_pool_min, vec_pool_max, topk=10, self_id=None, except_ids=None, limit_ids=None):
        q = []
        for i in range(len(vec_pool_min)):
            #skip self
            if i == self_id or ((not except_ids is None) and i in except_ids):
                continue
            if (not limit_ids is None) and i not in limit_ids:
                continue

            meet_min = np.maximum(vec_min, vec_pool_min[i])  # batchsize * embed_size
            meet_max = np.minimum(vec_max, vec_pool_max[i])
            dist = -np.prod(self.softplus((meet_max - meet_min)), axis=-1)
            if len(q) < topk:
                HP.heappush(q, self.index_dist(i, dist))
            else:
                #indeed it fetches the biggest
                tmp = HP.nsmallest(1, q)[0]
                if tmp.dist > dist:
                    HP.heapreplace(q, self.index_dist(i, dist) )
        rst = []
        while len(q) > 0:
            item = HP.heappop(q)
            rst.insert(0, (item.index, item.dist))
        return rst

    # input must contain a pool of vecs. return a list of indices and dist
    def NN(self, vec, vec_pool, self_id=None, except_ids=None, limit_ids=None):
        min_dist = sys.maxint
        rst = None
        for i in range(len(vec_pool)):
            #skip self
            if i == self_id or ((not except_ids is None) and i in except_ids):
                continue
            if (not limit_ids is None) and i not in limit_ids:
                continue
            dist = LA.norm(vec - vec_pool[i], ord=(1 if self.multiG.L1 else 2))
            if dist < min_dist:
                min_dist = dist
                rst = i
        return (rst, min_dist)
        
    # input must contain a pool of vecs. return a list of indices and dist. rank an index in a vec_pool from 
    def rank_index_from(self, vec_min, vec_max, vec_pool_min, vec_pool_max, 
             index, self_id = None, except_ids=None, limit_ids=None):

        meet_min = np.maximum(vec_min, vec_pool_min[index])  # batchsize * embed_size
        meet_max = np.minimum(vec_max, vec_pool_max[index])
        dist = - np.prod(self.softplus((meet_max - meet_min)), axis=-1)
        rank = 1
        for i in range(len(vec_pool_min)):
            if i == index or i == self_id or ((not except_ids is None) and i in except_ids):
                continue
            if (not limit_ids is None) and i not in limit_ids:
                continue
            meet_min = np.maximum(vec_min, vec_pool_min[i])  # batchsize * embed_size
            meet_max = np.minimum(vec_max, vec_pool_max[i])
            dist_i = - np.prod(self.softplus((meet_max - meet_min)), axis=-1)
            if dist > dist_i:
                rank += 1
        return rank

    # Change if AM changes
    '''
    def projection(self, e, source):
        assert (source in set([1, 2]))
        vec_e = self.ent_index2vec(e, source)
        #return np.add(np.dot(vec_e, self.mat), self._b)
        return np.dot(vec_e, self.mat)
    '''
    '''
    def projection(self, e, source, activation=True):
        assert (source in set([1, 2]))
        vec_e = self.ent_index2vec(e, source)
        #return np.add(np.dot(vec_e, self.mat), self._b)
        if activation:
            return np.tanh(np.dot(vec_e, self.mat))
        else:
            return np.dot(vec_e, self.mat)

    def projection_rel(self, r, source):
        assert (source in set([1, 2]))
        vec_r = self.rel_index2vec(r, source)
        #return np.add(np.dot(vec_e, self.mat), self._b)
        return np.dot(vec_r, self.mat)

    def projection_vec(self, vec, source):
        assert (source in set([1, 2]))
        #return np.add(np.dot(vec_e, self.mat), self._b)
        return np.dot(vec, self.mat)
    
    # Currently supporting only lan1 to lan2
    def projection_pool(self, ht_vec):
        #return np.add(np.dot(ht_vec, self.mat), self._b)
        return np.dot(ht_vec, self.mat)
    '''
    def projection_type_matrix(self, E_min, E_max):
        return E_min, E_max

    def projection(self, e, source=1): #normalize
        assert (source in set([1, 2]))
        min_e = max_e = self.ent_index2vec(e, source)
        return [min_e, max_e]
