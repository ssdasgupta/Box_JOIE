''' Module for training TF parts.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import wandb
import tqdm

from multiG import multiG 
import box_model as model

class Trainer(object):
    def __init__(self):
        self.batch_sizeK1=512
        self.batch_sizeK2=128
        self.batch_sizeA=32
        self.dim1=300
        self.dim2=50
        self._m1 = 0.5
        self._a1 = 5.
        self._a2 = 0.5
        self.multiG = None
        self.tf_parts = None
        self.save_path = 'this-model.ckpt'
        self.multiG_save_path = 'this-multiG.bin'
        self.L1=False
        self.sess = None

    def build(self, multiG, method='transe', bridge='CG-one',  dim1=300, dim2=50, batch_sizeK1=1024, batch_sizeK2=1024,
        batch_sizeA=32, a1=5., a2=0.5, m1=0.5, vol_temp=1.0, int_temp=0.1, int_method='gumbel', sampling_type='uniform',
        box_method='BoxMethods', transformation= 'relation_specific', save_path = 'this-model.ckpt',
        multiG_save_path = 'this-multiG.bin', log_save_path = 'tf_log', L1=False, eval_file = None, eval_freq=2, debug=False):
        
        self.multiG = multiG
        self.method = method
        self.bridge = bridge
        self.dim1 = self.multiG.dim1 = self.multiG.KG1.dim = dim1 # update dim
        self.dim2 = self.multiG.dim2 = self.multiG.KG2.dim = dim2 # update dim
        #self.multiG.KG1.wv_dim = self.multiG.KG2.wv_dim = wv_dim
        self.batch_sizeK1 = self.multiG.batch_sizeK1 = batch_sizeK1
        self.batch_sizeK2 = self.multiG.batch_sizeK2 = batch_sizeK2
        self.batch_sizeA = self.multiG.batch_sizeA = batch_sizeA
        self.sampling_type = sampling_type
        self.multiG_save_path = multiG_save_path
        self.log_save_path = log_save_path
        self.save_path = save_path
        self.L1 = self.multiG.L1 = L1
        self.eval_freq = eval_freq
        self.debug = debug
        self.tf_parts = model.TFParts(num_rels1=self.multiG.KG1.num_rels(),
                                 num_ents1=self.multiG.KG1.num_ents(),
                                 num_rels2=self.multiG.KG2.num_rels(),
                                 num_ents2=self.multiG.KG2.num_ents(),
                                 method=self.method,
                                 bridge=self.bridge,
                                 dim1=dim1,
                                 dim2=dim2,
                                 batch_sizeK1=self.batch_sizeK1,
                                 batch_sizeK2=self.batch_sizeK2,
                                 batch_sizeA=self.batch_sizeA,
                                 vol_temp=vol_temp,
                                 int_temp=int_temp,
                                 box_method=box_method,
                                 transformation=transformation,
                                 int_method=int_method,
                                 L1=self.L1)
        self.tf_parts._m1 = m1
        if eval_file:
            self.load_test_type(eval_file)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(log_save_path, graph=tf.get_default_graph()) 

    def gen_KM_batch(self, KG_index, batchsize, forever=False, shuffle=True): #batchsize is required
        KG = self.multiG.KG1
        if KG_index == 2:
            KG = self.multiG.KG2
        l = KG.triples.shape[0]
        while True:
            triples = KG.triples
            if shuffle:
                np.random.shuffle(triples)
            for i in range(0, l, batchsize):
                batch = triples[i: i+batchsize, :]
                if batch.shape[0] < batchsize:
                    batch = np.concatenate((batch, self.multiG.triples[:batchsize - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == batchsize
                if KG_index == 2:
                    neg_batch = KG.corrupt_batch(batch, sampling_type=self.sampling_type) # Sampling type: 'frequency', 'Uniform'
                else:
                    neg_batch = KG.corrupt_batch(batch, sampling_type='uniform')

                h_batch, r_batch, t_batch = batch[:, 0], batch[:, 1], batch[:, 2]
                neg_h_batch, neg_t_batch = neg_batch[:, 0], neg_batch[:, 2]
                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(np.int64), neg_h_batch.astype(np.int64), neg_t_batch.astype(np.int64)
            if not forever:
                break

    def gen_AM_batch(self, forever=False, shuffle=True): # not changed with its batchsize
        multiG = self.multiG
        l = len(multiG.align)
        while True:
            align = multiG.align
            if shuffle:
                np.random.shuffle(align)
            for i in range(0, l, self.batch_sizeA):
                batch = align[i: i+self.batch_sizeA, :]
                if batch.shape[0] < self.batch_sizeA:
                    batch = np.concatenate((batch, align[:self.batch_sizeA - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_sizeA
                n_batch = multiG.corrupt_align_batch(batch,tar=1, sampling_type=self.sampling_type) # only neg on class # Type: 'frequency', 'Uniform'

                e1_batch, e2_batch, e1_nbatch, e2_nbatch = batch[:, 0], batch[:, 1], n_batch[:, 0], n_batch[:, 1]
                yield e1_batch.astype(np.int64), e2_batch.astype(np.int64), e1_nbatch.astype(np.int64), e2_nbatch.astype(np.int64)
            if not forever:
                break
    
    def gen_AM_batch_non_neg(self, forever=False, shuffle=True):
        multiG = self.multiG
        l = len(multiG.align)
        while True:
            align = multiG.align
            if shuffle:
                np.random.shuffle(align)
            for i in range(0, l, self.batch_sizeA):
                batch = align[i: i+self.batch_sizeA, :]
                if batch.shape[0] < self.batch_sizeA:
                    batch = np.concatenate((batch, align[:self.batch_sizeA - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_sizeA
                e1_batch, e2_batch = batch[:, 0], batch[:, 1]
                yield e1_batch.astype(np.int64), e2_batch.astype(np.int64)
            if not forever:
                break

    def train1epoch_KM(self, sess, num_A_batch, num_B_batch, a2, lr, epoch):

        this_gen_A_batch = self.gen_KM_batch(KG_index=1, batchsize=self.batch_sizeK1, forever=True)
        this_gen_B_batch = self.gen_KM_batch(KG_index=2, batchsize=self.batch_sizeK2,forever=True)
        
        this_loss = []
        
        loss_A = loss_B = 0

        for batch_id in range(num_A_batch):
            # Optimize loss A
            A_h_index, A_r_index, A_t_index, A_hn_index, A_tn_index  = next(this_gen_A_batch)
            _, loss_A = sess.run([self.tf_parts._train_op_A, self.tf_parts._A_loss],
                    feed_dict={self.tf_parts._A_h_index: A_h_index, 
                               self.tf_parts._A_r_index: A_r_index,
                               self.tf_parts._A_t_index: A_t_index,
                               self.tf_parts._A_hn_index: A_hn_index, 
                               self.tf_parts._A_tn_index: A_tn_index,
                               self.tf_parts._lr: lr})
            batch_loss = [loss_A]
            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)
            if ((batch_id + 1) % 500 == 0 or batch_id == num_A_batch - 1):
                print('\rprocess KG1: %d / %d. Epoch %d' % (batch_id+1, num_A_batch+1, epoch))
            '''
            if batch_id == num_B_batch - 1:
                self.writer.add_summary(summary_op, epoch)
            '''
        for batch_id in range(num_B_batch):
            # Optimize loss B

            B_h_index, B_r_index, B_t_index, B_hn_index, B_tn_index  = next(this_gen_B_batch)
        
            _, loss_B = sess.run([self.tf_parts._train_op_B, self.tf_parts._B_loss],
                    feed_dict={self.tf_parts._B_h_index: B_h_index, 
                               self.tf_parts._B_r_index: B_r_index,
                               self.tf_parts._B_t_index: B_t_index,
                               self.tf_parts._B_hn_index: B_hn_index, 
                               self.tf_parts._B_tn_index: B_tn_index,
                                   self.tf_parts._lr: lr})
            
            # Observe total loss
            batch_loss = [loss_B]
            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)
            if ((batch_id + 1) % 500 == 0 or batch_id == num_B_batch - 1):
                print('\rprocess KG2: %d / %d. Epoch %d' % (batch_id+1, num_B_batch+1, epoch))
            '''
            if batch_id == num_B_batch - 1:
                self.writer.add_summary(summary_op, epoch)
            '''
        this_total_loss = np.sum(this_loss)
        print("KM Loss of epoch",epoch,":", this_total_loss)
        
        metric = {"A_B_loss": this_total_loss}
        wandb.log(metric, commit=False)
        
        return this_total_loss

    def train1epoch_AM(self, sess, num_AM_batch, a1, a2, lr, epoch):

        this_gen_AM_batch = self.gen_AM_batch(forever=True)
        #this_gen_AM_batch = self.gen_AM_batch_non_neg(forever=True)
        
        this_loss = []
        
        loss_AM = 0

        for batch_id in range(num_AM_batch):
            # Optimize loss A

            e1_index, e2_index, e1_nindex, e2_nindex  = next(this_gen_AM_batch)
            _, loss_AM, logits_AM = sess.run([self.tf_parts._train_op_AM, self.tf_parts._AM_loss, 
                    self.tf_parts.pos_logit_AM],
                    feed_dict={self.tf_parts._AM_index1: e1_index, 
                               self.tf_parts._AM_index2: e2_index,
                               self.tf_parts._AM_nindex1: e1_nindex,
                               self.tf_parts._AM_nindex2: e2_nindex,
                               self.tf_parts._lr: lr * a1})
            # Observe total loss
            batch_loss = [loss_AM]
            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)
            if ((batch_id + 1) % 100 == 0) or batch_id == num_AM_batch - 1:
                print('\rprocess: %d / %d. Epoch %d' % (batch_id+1, num_AM_batch+1, epoch))
            '''
            if batch_id == num_AM_batch - 1:
                self.writer.add_summary(summary_op, epoch)
            '''

        this_total_loss = np.sum(this_loss)
        print("AM Loss of epoch", epoch, ":", this_total_loss)
        metric = {"AM_loss": this_total_loss}
        wandb.log(metric)
        return this_total_loss

    def train1epoch_associative(self, sess, lr, a1, a2, epoch, AM_fold = 1):
       
        num_A_batch = int(self.multiG.KG1.num_triples() / self.batch_sizeK1)
        num_B_batch = int(self.multiG.KG2.num_triples() / self.batch_sizeK2)
        num_AM_batch = int(self.multiG.num_align() / self.batch_sizeA)
        
        
        if epoch <= 1:
            print('num_KG1_batch =', num_A_batch)
            print('num_KG2_batch =', num_B_batch)
            print('num_AM_batch =', num_AM_batch)
        loss_KM = self.train1epoch_KM(sess, num_A_batch, num_B_batch, a2, lr, epoch)
        #keep only the last loss
        for i in range(AM_fold):
            loss_AM = self.train1epoch_AM(sess, num_AM_batch, a1, a2, lr, epoch)
        return (loss_KM, loss_AM)

    def train(self, epochs=20, save_every_epoch=10, lr=0.001, a1=0.1, a2=0.05, m1=0.5, m2=1.0, AM_fold=1, half_loss_per_epoch=-1,
                 eval_freq=10):
        #sess = tf.Session()
        #sess.run(tf.initialize_all_variables())
        self.tf_parts._m1 = m1  
        t0 = time.time()
        self.best_mrr = 0.0
        best_hits_at_1 = 0.
        best_hits_at_3 = 0.
        for epoch in range(epochs):
            if epoch % self.eval_freq == 0:
                ranks, mrr, hits_at_1, hits_at_3 = self.eval()
                if mrr > self.best_mrr:
                    self.best_mrr = max(mrr, self.best_mrr)
                    hits_at_1 = max(hits_at_1, best_hits_at_1)
                    hits_at_3 = max(hits_at_3, best_hits_at_3)
                metric = {"eval_rank": ranks, "eval_mrr": mrr, "best_mrr": self.best_mrr,
                           "best_hits_at_3": best_hits_at_3, "best_hits_at_1": best_hits_at_1}
                wandb.log(metric, commit=False)
                print(f"Eval after {epoch} epochs: Rank {ranks}, mrr {mrr}")
                print(f"hits_at_3: {hits_at_3}, hits_at_1: {hits_at_1}")
            
            if half_loss_per_epoch > 0 and (epoch + 1) % half_loss_per_epoch == 0:
                lr /= 2.
            epoch_lossKM, epoch_lossAM = self.train1epoch_associative(self.sess, lr, a1, a2, epoch, AM_fold)
            print("Time use: %d" % (time.time() - t0))
            
            if np.isnan(epoch_lossKM) or np.isnan(epoch_lossAM):
                print("Training collapsed.")
                return
            
            # if (epoch + 1) % save_every_epoch == 0:
            #     # this_save_path = self.tf_parts._saver.save(self.sess, self.save_path)
            #     # self.multiG.save(self.multiG_save_path)
            #     print("MTransE saved in file: %s. Multi-graph saved in file: %s" % (this_save_path, self.multiG_save_path))
        this_save_path = self.tf_parts._saver.save(self.sess, self.save_path)
        self.multiG.save(self.multiG_save_path)
        print("MTransE saved in file: %s. Multi-graph saved in file: %s" % (this_save_path, self.multiG_save_path))
        print("Done")

    def eval(self):
        ranks = []
        mrr = []
        hits_at_1 = 0.
        hits_at_3 = 0.
        for ele in self.test_align:
            self.tf_parts.mode = 'eval'
            logits_AM = self.sess.run([self.tf_parts.pos_logit_AM],
                    feed_dict={self.tf_parts._AM_index1: [ele[0]] * self.multiG.KG2.num_ents() , 
                               self.tf_parts._AM_index2: np.arange(self.multiG.KG2.num_ents())})
            # logits_AM, temperature = self.sess.run([self.tf_parts.pos_logit_AM, self.tf_parts.temperature],
            #         feed_dict={self.tf_parts._AM_index1: [ele[0]] * self.multiG.KG2.num_ents() , 
            #                    self.tf_parts._AM_index2: np.arange(self.multiG.KG2.num_ents())})

            self.tf_parts.mode = 'train'
            if len(logits_AM) == 1:
                logits_AM = logits_AM[0]
            if np.isnan(logits_AM).any():
                raise RuntimeError

            rank = self.get_rank(logits_AM, ele[1])
            ranks.append(rank) # Extra dim of len 1
            mrr.append(1 / rank)
            if rank == 1:
                hits_at_1+=1
            if rank < 4:
                 hits_at_3+=1
        return np.mean(ranks), np.mean(mrr), hits_at_1/len(ranks), hits_at_3/len(ranks)
    
    def get_rank(self, score, target_idx):
        target_value = score[target_idx]
        before_me = np.where(score > target_value)
        if self.debug:
            if self.best_mrr > 0.75 and len(before_me[0]) <= 4 and len(before_me[0]) >= 2:
                print("---------------------")
                string = ''
                for ele in before_me[0]:
                    string+=self.multiG.KG2.ent_index2str(ele)
                    string+= '|'
                print(string)
                print("**********************")
                print(f"target: {self.multiG.KG2.ent_index2str(target_idx)}")
        equal_me = np.where(np.where(score  == target_value)[0] != target_idx)
        return len(before_me[0]) + len(equal_me[0]) /2 + 1



    def load_test_type(self, filename, splitter = '\t', line_end = '\n', dedup=True):
            num_lines = 0
            align = []
            lr_map = {}
            rl_map = {}
            e1_set = set()
            e2_set = set()
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
                e1_set.add(e1)
                e2_set.add(e2)
                if e1 == None or e2 == None:
                    continue
                align.append([e1, e2])
                if lr_map.get(e1) == None:
                    lr_map[e1] = set([e2])
                else:
                    lr_map[e1].add(e2)
                if rl_map.get(e2) == None:
                    rl_map[e2] = set([e1])
                else:
                    rl_map[e2].add(e1)

            self.test_align = np.array(align, dtype=np.int32)
            self.lr_map =lr_map
            self.rl_map = rl_map
            print("Loaded test data from %s, %d out of %d." % (filename, len(align), num_lines))

# A safer loading is available in Tester, with parameters like batch_size and dim recorded in the corresponding Data component
def load_tfparts(multiG, method='transe', bridge='CG-one', dim1=300, dim2=100, batch_sizeK1=1024, batch_sizeK=1024, batch_sizeA=64,
                save_path = 'this-model.ckpt', L1=False):
    tf_parts = model.TFParts(num_rels1=multiG.KG1.num_rels(), 
                            num_ents1=multiG.KG1.num_ents(), 
                            num_rels2=multiG.KG2.num_rels(), 
                            num_ents2=multiG.KG2.num_ents(),
                            method=self.method,
                            bridge=self.bridge, 
                            dim1=dim1, 
                            dim2=dim2, 
                            batch_sizeK=batch_sizeK, 
                            batch_sizeA=batch_sizeA, 
                            L1=L1)
    #with tf.Session() as sess:
    sess = tf.Session()
    tf_parts._saver.restore(sess, save_path)
