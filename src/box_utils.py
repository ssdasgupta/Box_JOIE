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
                 delta_space = 'log'):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.delta_space = delta_space
        self.min_embed, self.delta_embed = self.init_word_embedding()

    @property
    def init_embedding_scale(self):
        """For different measures, min and delta have different init value. """
        if self.delta_space:
            min_lower_scale, min_higher_scale = 1e-4, 0.9
            delta_lower_scale, delta_higher_scale = -1.0, -0.1
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

    def get_transformed_embedding(self, t1_idx, rel_vector):
        """Box embedding min and log_delta is returned"""
        min_relation, delta_relation = tf.split(
            rel_vector, 2, axis=-1, name='split')
        t1_min_embed = tf.nn.embedding_lookup(self.min_embed, t1_idx)
        t1_delta_embed = tf.nn.embedding_lookup(self.delta_embed, t1_idx)
        return t1_min_embed + min_relation, tf.exp(t1_delta_embed + delta_relation)

    def get_conditional_probability(self, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
        meet_min, meet_max = self.calc_intersection(
        t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed)
        """get conditional probabilities"""
        overlap_volume = tf.reduce_prod(tf.nn.softplus((meet_max - meet_min)
                                                       /self.temperature)*self.temperature, axis=-1)
        rhs_volume = tf.reduce_prod(tf.nn.softplus((t1_max_embed - t1_min_embed)
                                                   /self.temperature)*self.temperature, axis=-1)
        conditional_logits = tf.log(overlap_volume+1e-10) - tf.log(rhs_volume+1e-10)
        return conditional_logits

    def calc_intersection(self, t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed):
        """
        # two box embeddings are t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed
        Returns:
            join box, min box, and disjoint condition:
        """
        # join is min value of (a, c), max value of (b, d)
        meet_min = tf.maximum(t1_min_embed, t2_min_embed)  # batchsize * embed_size
        meet_max = tf.minimum(t1_max_embed, t2_max_embed)  # batchsize * embed_size
        return meet_min, meet_max

    def get_loss(self, logits, target):
        return - tf.reduce_mean(tf.multiply(logits, target) + tf.multiply(
            tf.log(1 - tf.exp(logits) + 1e-10), 1 - target))
