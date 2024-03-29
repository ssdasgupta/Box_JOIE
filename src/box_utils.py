from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

euler_gamma = 0.57721566490153286060
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
            tf.random_uniform([self.vocab_size, self.embed_dim],
                min_lower_scale, min_higher_scale, seed=my_seed),
            trainable=True, name='word_embed'
        )
        delta_embed = tf.Variable(
            tf.random_uniform([self.vocab_size, self.embed_dim],
                delta_lower_scale, delta_higher_scale, seed=my_seed), 
            trainable=True, name='delta_embed'
        )
        return min_embed, delta_embed

    def get_temperature(self, t1_idx):
        return tf.reshape(tf.constant(self.vol_temp), (-1, 1))

    def get_temperature(self, t1_idx):
        return tf.reshape(tf.constant(self.int_temp), (-1, 1))

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
            t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, vol_temp, int_temp):
        
        ## p(t2/t1) / p(mammal|man) = 1
        meet_min, meet_max = self.calc_intersection(
        t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed, int_temp)
        """get conditional probabilities"""
        
        if self.int_method == 'hard':

            self.overlap_volume = tf.reduce_sum(tf.log(tf.nn.softplus((meet_max - meet_min)
                                                       /vol_temp) * vol_temp + 1e-19), axis=-1)
            self.rhs_volume = tf.reduce_sum(tf.log(tf.nn.softplus((t1_max_embed - t1_min_embed)
                                                   /vol_temp) * vol_temp + 1e-19), axis=-1)
        elif self.int_method == 'gumbel':
            self.overlap_volume = tf.reduce_sum(tf.log(tf.nn.softplus((meet_max - meet_min - 2 * euler_gamma * int_temp)
                                                       /vol_temp) * vol_temp + 1e-19), axis=-1)

            self.rhs_volume = tf.reduce_sum(tf.log(tf.nn.softplus((t1_max_embed - t1_min_embed - 2 * euler_gamma * int_temp)
                                                   /vol_temp) * vol_temp + 1e-19), axis=-1)

        conditional_logits = self.overlap_volume - self.rhs_volume
        return conditional_logits

    def calc_intersection(self,
                          t1_min_embed,
                          t1_max_embed,
                          t2_min_embed,
                          t2_max_embed,
                          int_temp=0.1
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
            meet_min = int_temp * tf.reduce_logsumexp([t1_min_embed /int_temp , t2_min_embed /int_temp], 0)
            meet_min = tf.maximum(meet_min, tf.maximum(t1_min_embed, t2_min_embed))
            meet_max = - int_temp * tf.reduce_logsumexp([-t1_max_embed /int_temp , -t2_max_embed /int_temp], 0)
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
                 _min_vol=0.01,
                 _max_vol=100.,
                 _min_int=0.0001,
                 _max_int=20.):
        super(BoxMethodLearntTemp, self).__init__(embed_dim=embed_dim,
                                      vocab_size=vocab_size,
                                      temperature=temperature,
                                      int_temp=int_temp,
                                      int_method=int_method
                                      )
        self._min_vol = _min_vol
        self._max_vol = _max_vol
        self._min_int = _min_int
        self._max_int = _max_int
        self.vol_temp = self.init_temperatures(self.vol_temp, _min_vol, _max_vol, name='vol_temp_layer')
        self.int_temp = self.init_temperatures(self.int_temp, _min_int, _max_int, name='int_temp_layer')
    
    def init_temperatures(self, init=1.0, _min=0.01, _max=100, name='temp'):
        assert _min <= init <= _max
        p = (init - _min) / (_max - _min)

        temp = tf.Variable(
            tf.ones([self.vocab_size, self.embed_dim]) * tf.log(p / (1 - p) + 1e-19),
            trainable=True, name=name
        )
        return temp

    def normalized_temp(self, x, _min=0.01, _max=100):
        return _min + tf.sigmoid(x) * (_max - _min)
    
    def get_vol_temperature(self, t1_idx):
        return self.normalized_temp(
                tf.nn.embedding_lookup(self.vol_temp, t1_idx),
                self._min_vol,
                self._max_vol
            )

    def get_int_temperature(self, t1_idx):
        return self.normalized_temp(
                tf.nn.embedding_lookup(self.int_temp, t1_idx),
                self._min_int,
                self._max_int
            )

class BoxMethodLearntTempScalar(BoxMethodLearntTemp):
    def init_temperatures(self, init=1.0, _min=0.01, _max=100, name='temp'):
        assert _min <= init <= _max
        p = (init - _min) / (_max - _min)

        temp = tf.Variable(
            tf.ones([self.vocab_size]) * tf.log(p / (1 - p) + 1e-19),
            trainable=True, name=name
        )
        return temp

    def get_vol_temperature(self, t1_idx):
        return self.normalized_temp(
                tf.reshape(tf.nn.embedding_lookup(self.vol_temp, t1_idx), (-1, 1)),
                self._min_vol,
                self._max_vol
            )

    def get_int_temperature(self, t1_idx):
        return self.normalized_temp(
            tf.reshape(tf.nn.embedding_lookup(self.int_temp, t1_idx), (-1, 1)),
            self._min_int,
            self._max_int
        )
