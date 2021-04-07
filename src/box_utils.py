from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


my_seed = 20180112
tf.set_random_seed(my_seed)

class BoxMethods(object):
    def __init__(self,
                 embed_dim,
                 vocab_size,
                 temperature = 1.0,
                 int_temp = 0.1,
                 int_method = 'gumbel'):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.vol_temp = temperature
        self.int_temp = int_temp
        self.int_method = int_method
        self.min_embed, self.delta_embed = self.init_word_embedding()

    @property
    def init_embedding_scale(self):
        """For different measures, min and delta have different init value. """
        min_lower_scale, min_higher_scale = 1e-3, 0.1#1e-4, 0.9
        delta_lower_scale, delta_higher_scale = -1.0, -0.01#-1.0, -0.1
        # min_lower_scale, min_higher_scale = 1e-4, 0.5
        # delta_lower_scale, delta_higher_scale = -0.1, -0.0
        return min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale

    def init_word_embedding(self):
        min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale = self.init_embedding_scale
 
        min_embed = tf.Variable(
            tf.random_uniform([self.vocab_size, self.embed_dim], min_lower_scale, min_higher_scale, seed=my_seed),
            trainable=True, name='word_embed')
        delta_embed = tf.Variable(
            tf.random_uniform([self.vocab_size, self.embed_dim], delta_lower_scale, delta_higher_scale,
                              seed=my_seed), trainable=True, name='delta_embed')

        return min_embed, delta_embed

    def get_temperature(self, t1_idx):
        return tf.reshape(tf.constant(self.vol_temp), (-1, 1))

    def get_transformed_embedding(self, t1_idx, rel_vector):
        """Box embedding min + min_rel and log_delta + log_relation_delta is returned"""
        min_relation, delta_relation = tf.split(
            rel_vector, 2, axis=-1, name='split')
        t1_min_embed = tf.nn.embedding_lookup(self.min_embed, t1_idx)
        t1_delta_embed = tf.nn.embedding_lookup(self.delta_embed, t1_idx)
        return t1_min_embed + min_relation, tf.exp(t1_delta_embed + delta_relation)

    def get_embedding(self, t1_idx):
        """Box embedding min and log_delta is returned"""
        t1_min_embed = tf.nn.embedding_lookup(self.min_embed, t1_idx)
        t1_delta_embed = tf.nn.embedding_lookup(self.delta_embed, t1_idx)
        return t1_min_embed , tf.exp(t1_delta_embed)

    def get_conditional_probability(self,
            t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, vol_temp):
        
        ## p(t2/t1) / p(mammal|man) = 1
        meet_min, meet_max = self.calc_intersection(
        t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed)
        """get conditional probabilities"""

        self.overlap_volume = tf.reduce_sum(tf.log(tf.nn.softplus((meet_max - meet_min)
                                                       /vol_temp) * vol_temp + 1e-19), axis=-1)
        self.rhs_volume = tf.reduce_sum(tf.log(tf.nn.softplus((t1_max_embed - t1_min_embed)
                                                   /vol_temp) * vol_temp + 1e-19), axis=-1)
        
        conditional_logits = self.overlap_volume - self.rhs_volume
        return conditional_logits

    def calc_intersection(self,
                          t1_min_embed,
                          t1_max_embed,
                          t2_min_embed,
                          t2_max_embed,
                          method='gumbel', #'gumbel'
                          ):
        """
        # two box embeddings are t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed
        Returns:
            meet_min and meet_max

            *---------\-----| -------\ 
            meet_min = max (*, \-) and
            meet_max = min(|, \)
            smooth max is equivalent to logsum_exp
            smooth min is equivalent to -logsum_exp(-parameters)

        """

        if self.int_method == 'gumbel':
            meet_min = self.int_temp * tf.reduce_logsumexp([t1_min_embed /self.int_temp , t2_min_embed /self.int_temp], 0)
            meet_min = tf.maximum(meet_min, tf.maximum(t1_min_embed, t2_min_embed))
            meet_max = - self.int_temp * tf.reduce_logsumexp([-t1_max_embed /self.int_temp , -t2_max_embed /self.int_temp], 0)
            meet_max = tf.minimum(meet_max, tf.minimum(t1_max_embed, t2_max_embed))
        elif self.int_method == 'hard':
            meet_min = tf.maximum(t1_min_embed, t2_min_embed)  # batchsize * embed_size
            meet_max = tf.minimum(t1_max_embed, t2_max_embed)  # batchsize * embed_size
        else:
            raise ValueError("Intersection Method Not found")
            return

        return meet_min, meet_max

    def get_loss(self, logits, target):
        return - tf.reduce_mean(tf.multiply(logits, target) + tf.multiply(
            tf.log(1 - tf.exp(logits) + 1e-19), 1 - target))

class BoxMethodLearntTemp(BoxMethods):
    def __init__(self,
                 embed_dim,
                 vocab_size,
                 temperature = 1.0,
                 int_temp = 0.1,
                 int_method = 'gumbel',
                 _min=0.01,
                 _max=100):
        super(BoxMethodLearntTemp, self).__init__(embed_dim=embed_dim,
                                      vocab_size=vocab_size,
                                      temperature=temperature,
                                      int_temp=int_temp,
                                      int_method=int_method
                                      )
        self.vol_temp = self.init_temperatures(_min, _max)
    
    def init_temperatures(self, _min=0.01, _max=100):
        temp_basic = tf.Variable(
            tf.random_uniform([self.vocab_size, self.embed_dim], -5, 5, seed=my_seed),
            trainable=True, name='temp_basic')

        temp = _min + tf.sigmoid(temp_basic) * (_max - _min)
        return temp
    
    def get_temperature(self, t1_idx):
        return tf.nn.embedding_lookup(self.vol_temp, t1_idx)

class BoxMethodLearntTempScalar(BoxMethodLearntTemp):
    def __init__(self,
                 embed_dim,
                 vocab_size,
                 temperature = 1.0,
                 int_temp = 0.1,
                 int_method = 'gumbel',
                 _min=0.01,
                 _max=100):
        super(BoxMethodLearntTempScalar, self).__init__(embed_dim=embed_dim,
                                      vocab_size=vocab_size,
                                      temperature=temperature,
                                      int_temp=int_temp,
                                      int_method=int_method
                                      )
        self.vol_temp = self.init_temperatures(_min, _max)
    
    def init_temperatures(self, _min=0.01, _max=100):
        temp_basic = tf.Variable(
            tf.random_uniform([self.vocab_size], -5, 5, seed=my_seed),
            trainable=True, name='temp_basic')
        temp = _min + tf.sigmoid(temp_basic) * (_max - _min)
        return temp
    
    def get_temperature(self, t1_idx):
        return tf.reshape(tf.nn.embedding_lookup(self.vol_temp, t1_idx), (-1, 1))
