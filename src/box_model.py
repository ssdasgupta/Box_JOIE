'''Model for holding TF parts. etc.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import wandb
import tensorflow as tf

from multiG import multiG
import pickle
from utils import circular_correlation, np_ccorr
from box_utils import BoxMethods
from box_utils import BoxMethodLearntTemp
from box_utils import BoxMethodLearntTempScalar

methods = {
    'BoxMethods': BoxMethods,
    'BoxMethodLearntTemp': BoxMethodLearntTemp,
    'BoxMethodLearntTempScalar': BoxMethodLearntTempScalar
}

# Orthogonal Initializer from
# https://github.com/OlavHN/bnlstm
def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)

def orthogonal_initializer(scale=1.0, dtype=tf.float32):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(orthogonal(shape) * scale, dtype)
  return _initializer

class TFParts(object):
    ''' Box method described here, in this class.
    Let us just say that, instances will be vectors and the ontology will be boxes.
    1. Thus intra model will be one transE or HolE for vectors.
    2. For Boxes it will be a --> b => P(a|b) = 1 and we optimize that using cross entropy.
    3. The cross coupling is defined as (here we can do multiple thing)
        a. Device a distance measure between box and the vectors (Non smooth for sure).
        b. Like query2box make box from the vector and then try to put it in the master vector. 
           (This is same as the CT method described in JOIE.)
    4. The instance level can also be boxes. 
    
    '''

    def __init__(self, num_rels1,
                 num_ents1, num_rels2, num_ents2, 
                 method='distmult', bridge='CG', 
                 dim1=300, dim2=100,
                 batch_sizeK1=512, batch_sizeK2=512, batch_sizeA=256, 
                 vol_temp=1.0, int_temp=0.1, int_method='gumbel', box_method='BoxMethods',
                 transformation='relation_specific', L1=False):
        self._num_relsA = num_rels1
        self._num_entsA = num_ents1
        self._num_relsB = num_rels2
        self._num_entsB = num_ents2
        self.method=method
        self.bridge=bridge
        self._dim1 = dim1
        self._dim2 = dim2
        if bridge =='box':
            self._dim2 = dim1
        self._hidden_dim = hid_dim = 50
        self._batch_sizeK1 = batch_sizeK1
        self._batch_sizeK2 = batch_sizeK2
        self._batch_sizeA = batch_sizeA
        self._epoch_loss = 0
        # margins
        self._m1 = 0.5

        #temperatures
        self.vol_temp = vol_temp
        self.int_temp = int_temp
        self.int_method = int_method
        self.box_method = box_method

        #Relation_transform
        self.transformation = transformation

        self.L1 = L1
        self.mode = 'train'
        self.build()
        print("TFparts build up! Embedding method: ["+self.method+"]. Bridge method:["+self.bridge+"]. intersection method: ["+self.int_method+"]")
        print("Margin Paramter: [m1] "+str(self._m1))

    @property
    def dim(self):
        return self._dim1, self._dim2  

    def build(self):
        tf.reset_default_graph()

        with tf.variable_scope("graph"):
            # Variables (matrix of embeddings/transformations)
            

            #----- KG1 --- What is KG1? Is this ontology or instances?

            ### ------------ These variables must be replaced with box embeddings ------ ### 
            self._ht1 = ht1 = tf.get_variable(
                name='ht1',  # for t AND h
                shape=[self._num_entsA, self._dim1],
                dtype=tf.float32)
            self._r1 = r1 = tf.get_variable(
                name='r1',
                shape=[self._num_relsA, self._dim1],
                dtype=tf.float32)
            
            # KG2 --- Again, What is KG1? Is this ontology or instances?

            # Box method define all the box related classes
            _BoxMethod = methods[self.box_method]

            
            self._ht2 = ht2 = _BoxMethod(
                                   self._dim2,
                                   self._num_entsB,
                                   temperature=self.vol_temp,
                                   int_temp=self.int_temp,
                                   int_method=self.int_method
                                )

            # self._ht2 = ht2 = tf.get_variable(
            #     name='ht2',  # for t AND h
            #     shape=[self._num_entsB, self._dim2],
            #     dtype=tf.float32)

            # Here relations means relation specific transforms for boxes
            # Dimension is doubled because one for min and one for delta
            # Also the transformation is different for head and tail.

            # Here, 

            self._r2_head = r2_head = tf.get_variable(
                name='r2_head',
                shape=[self._num_relsB, self._dim2 * 2], 
                dtype=tf.float32)

            self._r2_tail = r2_tail = tf.get_variable(
                name='r2_tail',
                shape=[self._num_relsB, self._dim2 * 2],
                dtype=tf.float32)

            self._ht1_norm = tf.nn.l2_normalize(ht1, 1)
            # self._ht2_norm = tf.nn.l2_normalize(ht2, 1) ## --- maybe not require for this one, if we are considering boxes.

            ######################## Graph A Loss #######################
            # Language A KM loss : [|| h + r - t ||_2 + m1 - || h + r - t ||_2]+    here [.]+ means max (. , 0)
            self._A_h_index = A_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK1],
                name='A_h_index')
            self._A_r_index = A_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK1],
                name='A_r_index')
            self._A_t_index = A_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK1],
                name='A_t_index')
            self._A_hn_index = A_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK1],
                name='A_hn_index')
            self._A_tn_index = A_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK1],
                name='A_tn_index')
            '''
            A_loss_matrix = tf.subtract(
                tf.add(
                    tf.batch_matmul(A_h_ent_batch, tf.reshape(A_mat_h_batch, [-1, self.dim, self.dim])),
                    A_rel_batch),
                tf.batch_matmul(A_t_ent_batch, tf.reshape(A_mat_h_batch, [-1, self.dim, self.dim]))
            )'''
            
            A_h_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1, A_h_index), 1)
            A_t_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1, A_t_index), 1)
            A_rel_batch = tf.nn.embedding_lookup(r1, A_r_index)
           
            A_hn_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1,A_hn_index), 1)
            A_tn_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1,A_tn_index), 1)

            if self.method == 'transe':
                ##### TransE score
                # This stores h + r - t
                A_loss_matrix = tf.subtract(tf.add(A_h_ent_batch, A_rel_batch), A_t_ent_batch)
                # This stores h' + r - t' for negative samples
                A_neg_matrix = tf.subtract(tf.add(A_hn_ent_batch, A_rel_batch), A_tn_ent_batch)
                if self.L1:
                    self._A_loss = A_loss = tf.reduce_sum(
                        tf.maximum(
                        tf.subtract(tf.add(tf.reduce_sum(tf.abs(A_loss_matrix), 1), self._m1),
                        tf.reduce_sum(tf.abs(A_neg_matrix), 1)), 
                        0.)
                    ) / self._batch_sizeK1
                else:
                    self._A_loss = A_loss = tf.reduce_sum(
                        tf.maximum(
                        tf.subtract(tf.add(tf.sqrt(tf.reduce_sum(tf.square(A_loss_matrix), 1)), self._m1),
                        tf.sqrt(tf.reduce_sum(tf.square(A_neg_matrix), 1))), 
                        0.)
                    ) / self._batch_sizeK1

            elif self.method == 'distmult':
                ##### DistMult score
                A_loss_matrix = tf.reduce_sum(tf.multiply(A_rel_batch, tf.multiply(A_h_ent_batch, A_t_ent_batch)), 1)
                A_neg_matrix = tf.reduce_sum(tf.multiply(A_rel_batch, tf.multiply(A_hn_ent_batch, A_tn_ent_batch)), 1)

                self._A_loss = A_loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(A_neg_matrix, A_loss_matrix), self._m1), 0.)) / self._batch_sizeK1
    
            elif self.method == 'hole':
                ##### HolE score
                A_loss_matrix = tf.reduce_sum(tf.multiply(A_rel_batch, circular_correlation(A_h_ent_batch, A_t_ent_batch)), 1)
                A_neg_matrix = tf.reduce_sum(tf.multiply(A_rel_batch, circular_correlation(A_hn_ent_batch, A_tn_ent_batch)), 1)

                self._A_loss = A_loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(A_neg_matrix, A_loss_matrix), self._m1), 0.)) / self._batch_sizeK1

            else:
                ## Should we do something else? atleast the loss function could be bce to make them in same scale
                raise ValueError('Embedding method not valid!')


            ######################## Graph B Loss #######################
            # Language B KM loss : [|| h + r - t ||_2 + m1 - || h + r - t ||_2]+    here [.]+ means max (. , 0)
            self._B_h_index = B_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_h_index')
            self._B_r_index = B_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_r_index')
            self._B_t_index = B_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_t_index')
            self._B_hn_index = B_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_hn_index')
            self._B_tn_index = B_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_tn_index')
            
            B_rel_batch_head = tf.nn.embedding_lookup(r2_head, B_r_index)
            B_rel_batch_tail = tf.nn.embedding_lookup(r2_tail, B_r_index)
            if self.transformation == 'relation_specific':
                h_min_embed, h_max_embed = self._ht2.get_transformed_embedding(B_h_index, B_rel_batch_head)
                t_min_embed, t_max_embed = self._ht2.get_transformed_embedding(B_t_index, B_rel_batch_tail)
                hn_min_embed, hn_max_embed = self._ht2.get_transformed_embedding(B_hn_index, B_rel_batch_head)
                tn_min_embed, tn_max_embed = self._ht2.get_transformed_embedding(B_tn_index, B_rel_batch_tail)
            else:
                h_min_embed, h_max_embed = self._ht2.get_embedding(B_h_index)
                t_min_embed, t_max_embed = self._ht2.get_embedding(B_t_index)
                hn_min_embed, hn_max_embed = self._ht2.get_embedding(B_hn_index)
                tn_min_embed, tn_max_embed = self._ht2.get_embedding(B_tn_index)

            vol_temp = (self._ht2.get_vol_temperature(B_h_index) + self._ht2.get_vol_temperature(B_t_index)) / 2.
            vol_temp_n = (self._ht2.get_vol_temperature(B_hn_index) + self._ht2.get_vol_temperature(B_tn_index)) / 2.

            int_temp = (self._ht2.get_int_temperature(B_h_index) + self._ht2.get_int_temperature(B_t_index)) / 2.
            int_temp_n = (self._ht2.get_int_temperature(B_hn_index) + self._ht2.get_int_temperature(B_tn_index)) / 2.
            
            self.pos_logit_B = pos_logit = self._ht2.get_conditional_probability(h_min_embed, h_max_embed, 
                t_min_embed, t_max_embed, vol_temp, int_temp)
            neg_logit = self._ht2.get_conditional_probability(hn_min_embed, hn_max_embed, 
                tn_min_embed, tn_max_embed, vol_temp_n, int_temp_n)
            logits = tf.concat([pos_logit, neg_logit], 0)
            label = tf.concat([tf.ones_like(pos_logit),
                tf.zeros_like(neg_logit)], 0)

            self._B_loss = B_loss = self._ht2.get_loss(logits, label)

            ######################## Type Loss #######################
            self._AM_index1 = AM_index1 = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='AM_index1')
            self._AM_index2 = AM_index2 = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='AM_index2')
            
            self._AM_nindex1 = AM_nindex1 = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeA],
                name='AM_nindex1')
            self._AM_nindex2 = AM_nindex2 = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeA],
                name='AM_nindex2')

            AM_ent1_batch = tf.nn.embedding_lookup(ht1, AM_index1)
            AM_ent1_nbatch = tf.nn.embedding_lookup(ht1, AM_nindex1)
            #self.instance_delta = tf.Variable(tf.ones_like(AM_ent1_batch), trainable = False) * 10**(-7)
            # relation_vector =  tf.zeros(dtype=tf.float32, shape=[self._batch_sizeA, self._dim1 * 2])

            AM_ent1_min = AM_ent1_max = AM_ent1_batch
            AM_nent1_min = AM_nent1_max = AM_ent1_nbatch 

            AM_ent2_min, AM_ent2_max = self._ht2.get_embedding(AM_index2)
            AM_nent2_min, AM_nent2_max = self._ht2.get_embedding(AM_nindex2)

            self.vol_temp = vol_temp = self._ht2.get_vol_temperature(AM_index2)
            vol_temp_n = self._ht2.get_vol_temperature(AM_nindex2)

            self.int_temp = int_temp = self._ht2.get_int_temperature(AM_index2)
            int_temp_n = self._ht2.get_int_temperature(AM_nindex2)

            self.pos_logit_AM = pos_logit = self._ht2.get_conditional_probability(AM_ent1_min, AM_ent1_max, 
                AM_ent2_min, AM_ent2_max, vol_temp, int_temp)
            neg_logit = self._ht2.get_conditional_probability(AM_nent1_min, AM_nent1_max, 
                AM_nent2_min, AM_nent2_max, vol_temp_n, int_temp_n)

            logits = tf.concat([pos_logit, neg_logit], 0)
            label = tf.concat([tf.ones_like(pos_logit), tf.zeros_like(neg_logit)], 0)
            self._AM_loss = AM_loss = self._ht2.get_loss(logits, label)

            # Optimizer
            self._lr = lr = tf.placeholder(tf.float32)
            self._opt = opt = tf.train.AdamOptimizer(lr) #Other options #AdagradOptimizer(lr) #GradientDescentOptimizer(lr)
            self._train_op_A = train_op_A = opt.minimize(A_loss)
            self._train_op_B = train_op_B = opt.minimize(B_loss)
            self._train_op_AM = train_op_AM = opt.minimize(AM_loss)

            # Saver
            self._saver = tf.train.Saver()
