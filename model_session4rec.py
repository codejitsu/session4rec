import numpy as np
import tensorflow as tf
import pandas as pd
import os
from tensorflow.python.ops import rnn, rnn_cell

from defaults import Defaults
import evaluation

PATH_TO_TRAIN = './data/rsc15_train_full.txt'
PATH_TO_TEST = './data/rsc15_test.txt'

class Session4RecPredictor:
    def __init__(self, defaults, session):
        self.sess = session
        self.is_training = defaults.is_training

        self.layers = defaults.layers
        self.rnn_size = defaults.rnn_size
        self.n_epochs = defaults.n_epochs
        self.batch_size = defaults.batch_size
        self.dropout_p_hidden = defaults.dropout_p_hidden
        self.learning_rate = defaults.learning_rate
        self.decay = defaults.decay
        self.decay_steps = defaults.decay_steps
        self.sigma = defaults.sigma
        self.init_as_normal = defaults.init_as_normal
        self.reset_after_session = defaults.reset_after_session
        self.session_key = defaults.session_key
        self.item_key = defaults.item_key
        self.time_key = defaults.time_key
        self.grad_cap = defaults.grad_cap
        self.n_items = defaults.n_items
        if defaults.hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif defaults.hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        if defaults.loss == 'cross-entropy':
            if defaults.final_act == 'tanh':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy
        elif defaults.loss == 'bpr':
            if defaults.final_act == 'linear':
                self.final_activation = self.linear
            elif defaults.final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr
        elif defaults.loss == 'top1':
            if defaults.final_act == 'linear':
                self.final_activation = self.linear
            elif defaults.final_act == 'relu':
                self.final_activatin = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1
        else:
            raise NotImplementedError

        self.checkpoint_dir = defaults.checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            raise Exception("[!] Checkpoint Dir not found")

        self.model()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep = 10)

        if self.is_training:
            return

        # use self.predict_state to hold hidden states during prediction.
        self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype = np.float32) for _ in range(self.layers)]
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, '{}/gru-model-{}'.format(self.checkpoint_dir, defaults.test_model))

    # activation

    def linear(self, X):
        return X

    def tanh(self, X):
        return tf.nn.tanh(X)

    def softmax(self, X):
        return tf.nn.softmax(X)

    def softmaxth(self, X):
        return tf.nn.softmax(tf.tanh(X))

    def relu(self, X):
        return tf.nn.relu(X)

    def sigmoid(self, X):
        return tf.nn.sigmoid(X)

    # loss
    def cross_entropy(self, yhat):
        return tf.reduce_mean(-tf.log(tf.diag_part(yhat)+1e-24))

    def bpr(self, yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat)-yhatT)))

    def top1(self, yhat):
        yhatT = tf.transpose(yhat)
        term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat)+yhatT)+tf.nn.sigmoid(yhatT**2), axis=0)
        term2 = tf.nn.sigmoid(tf.diag_part(yhat)**2) / self.batch_size
        return tf.reduce_mean(term1 - term2)

    def model(self):
        self.X = tf.placeholder(tf.int32, [self.batch_size], name='input')
        self.Y = tf.placeholder(tf.int32, [self.batch_size], name='output')
        self.state = [tf.placeholder(tf.float32, [self.batch_size, self.rnn_size], name='rnn_state') for _ in range(self.layers)]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('gru_layer'):
            sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (self.n_items + self.rnn_size))
            if self.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
            embedding = tf.get_variable('embedding', [self.n_items, self.rnn_size], initializer=initializer)
            softmax_W = tf.get_variable('softmax_w', [self.n_items, self.rnn_size], initializer=initializer)
            softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.constant_initializer(0.0))

            cell = rnn_cell.GRUCell(self.rnn_size, activation=self.hidden_act)
            drop_cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden)
            stacked_cell = rnn_cell.MultiRNNCell([drop_cell] * self.layers)

            inputs = tf.nn.embedding_lookup(embedding, self.X)
            output, state = stacked_cell(inputs, tuple(self.state))
            self.final_state = state

        if self.is_training:
            '''
            Use other examples of the minibatch as negative samples.
            '''
            sampled_W = tf.nn.embedding_lookup(softmax_W, self.Y)
            sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y)
            logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
            self.yhat = self.final_activation(logits)
            self.cost = self.loss_function(self.yhat)
        else:
            logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
            self.yhat = self.final_activation(logits)

        if not self.is_training:
            return

        self.lr = tf.maximum(1e-5,tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay, staircase=True))

        '''
        Try different optimizers.
        '''
        #optimizer = tf.train.AdagradOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        #optimizer = tf.train.AdadeltaOptimizer(self.lr)
        #optimizer = tf.train.RMSPropOptimizer(self.lr)

        tvars = tf.trainable_variables()
        gvs = optimizer.compute_gradients(self.cost, tvars)
        if self.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in gvs]
        else:
            capped_gvs = gvs
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step = self.global_step)

    def init(self, data):
        data.sort([self.session_key, self.time_key], inplace = True)
        offset_sessions = np.zeros(data[self.session_key].nunique() + 1, dtype = np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        return offset_sessions

    def train(self, data):
        self.error_during_train = False
        itemids = data[self.item_key].unique()

        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)

        data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':self.itemidmap[itemids].values}), on=self.item_key, how='inner')

        offset_sessions = self.init(data)

        print('training model...')

        for epoch in range(self.n_epochs):
            print('training epoch: {}'.format(epoch))

            epoch_cost = []
            state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]
            session_idx_arr = np.arange(len(offset_sessions)-1)
            iters = np.arange(self.batch_size)
            maxiter = iters.max()
            start = offset_sessions[session_idx_arr[iters]]
            end = offset_sessions[session_idx_arr[iters]+1]
            finished = False

            while not finished:
                minlen = (end-start).min()
                out_idx = data.ItemIdx.values[start]
                for i in range(minlen-1):
                    in_idx = out_idx
                    out_idx = data.ItemIdx.values[start+i+1]
                    # prepare inputs, targeted outputs and hidden states
                    fetches = [self.cost, self.final_state, self.global_step, self.lr, self.train_op]
                    feed_dict = {self.X: in_idx, self.Y: out_idx}
                    for j in range(self.layers):
                        feed_dict[self.state[j]] = state[j]

                    cost, state, step, lr, _ = self.sess.run(fetches, feed_dict)
                    epoch_cost.append(cost)
                    if np.isnan(cost):
                        print(str(epoch) + ':Nan error!')
                        self.error_during_train = True
                        return
                    if step == 1 or step % self.decay_steps == 0:
                        avgc = np.mean(epoch_cost)
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tloss: {:.6f}'.format(epoch, step, lr, avgc))
                start = start+minlen-1
                mask = np.arange(len(iters))[(end-start)<=1]
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(offset_sessions)-1:
                        finished = True
                        break
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[session_idx_arr[maxiter]]
                    end[idx] = offset_sessions[session_idx_arr[maxiter]+1]
                if len(mask) and self.reset_after_session:
                    for i in range(self.layers):
                        state[i][mask] = 0

            avgc = np.mean(epoch_cost)
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                self.error_during_train = True
                return

            save_path = self.saver.save(self.sess, '{}/model.ckpt'.format(self.checkpoint_dir), global_step = epoch)
            print('Model saved to {}'.format(save_path))

if __name__ == '__main__':
    print('Start session4rec training...')

    defaults = Defaults()

    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId': np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId': np.int64})

    defaults.n_items = len(data['ItemId'].unique())
    defaults.dropout_p_hidden = 1.0 if defaults.is_training == 0 else 0.5

    if not os.path.exists(defaults.checkpoint_dir):
        os.mkdir(defaults.checkpoint_dir)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(config = gpu_config) as session:
        predictor = Session4RecPredictor(defaults, session)

        if defaults.is_training:
            predictor.train(data)
        else:
            res = evaluation.evaluate_sessions_batch(gru, data, valid)
            print('Recall@20: {}\tMRR@20: {}'.format(res[0], res[1]))
