import numpy as np
import pandas as pd
from CNN import CNN_fc
import tensorflow as tf
import pickle


class seq2expr (object):
    def __init__(self, flags):

        if flags.nChans != '0':
            nChans = [int(i) for i in flags.nChans.split(',')]
        else:
            nChans = []

        psb = (int(flags.psb.split(',')[0]), int(flags.psb.split(',')[1]))
        csc = (int(flags.csc.split(',')[0]), int(flags.csc.split(',')[1]))
        sc = (int(flags.sc.split(',')[0]), int(flags.sc.split(',')[1]))

        self.model = CNN_fc(dropout_rate = flags.dr, poolSize_bind = psb,
        convSize_coop = csc, stride_coop = sc, coopAct = flags.cAct,
        fcConvChan_coop = nChans, outAct = flags.oAct)

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer = self.optimizer, net = self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, flags.o + '/checkpoints', max_to_keep = None)

        #These metrics accumulate the values over epochs
        self.running_loss_train = tf.keras.metrics.Mean(name = 'running_loss_train')
        self.loss_train = None
        self.loss_valid = None
        self.loss_test = None
        self.flags_ = flags


    @tf.function
    def train_step(self, seq, conc, gt_expr):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs = (seq, conc), training=True)
            loss = self.loss(gt_expr, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        #Note: Train loss is being updated by a loss which obtained by training = True prediction
        self.running_loss_train(loss)


    def valid_step(self, seq, conc, gt_expr):
        predictions = self.model(inputs = (seq, conc), training=False)
        loss = self.loss(gt_expr, predictions)
        self.loss_valid = loss.numpy() # Assunes validation data comes all in single batch


    def test_step(self, seq, conc, gt_expr):
        predictions = self.model(inputs = (seq, conc), training=False)
        loss = self.loss(gt_expr, predictions)
        self.loss_test = loss.numpy() # Assumes test data comes all in single batch

    def _predict_KO(self, seq, conc):
        dl_ko = np.copy(conc)
        dl_ko[:,0] = 0
        tw_ko = np.copy(conc)
        tw_ko[:,1] = 0
        sn_ko = np.copy(conc)
        sn_ko[:,2] = 0

        dl_ko_pred = self.model(inputs = (seq, dl_ko), training=False)
        tw_ko_pred = self.model(inputs = (seq, tw_ko), training=False)
        sn_ko_pred = self.model(inputs = (seq, sn_ko), training=False)

        return dl_ko_pred, tw_ko_pred, sn_ko_pred

    def collect_loss_train(self):
        """
        resets runnings trail loss and appends it to the train loss list
        It should be run at the end of each epoch
        """
        self.loss_train = self.running_loss_train.result().numpy()
        self.running_loss_train.reset_states()


    def compute_TF_roles(self, seq, conc):
        """
        Evaluate the learned TF roles for the given seq and TF conc.
        evaluates delta-expression (ref-ko; activator +, repressor -)
        for dl, tw and sn on 3 regions along vd axis: bin IDs [1:6], [7:12], [13:17]

        returns tensor(#enhancer * #regions * #TFs)
        where regions:v,p,d  and  TFs: dl, tw, sn

        Mainly useful for input data with offset = 0
        """

        ref_pred = self.model(inputs = (seq, conc), training=False)
        ref_pred = tf.reshape(ref_pred, [-1,17])

        dl_ko_pred, tw_ko_pred, sn_ko_pred = self._predict_KO(seq, conc)

        dl_ko_pred = tf.reshape(dl_ko_pred, [-1,17])
        tw_ko_pred = tf.reshape(tw_ko_pred, [-1,17])
        sn_ko_pred = tf.reshape(sn_ko_pred, [-1,17])

        #vental: v; peak: p; dorsal: d
        delta_dl_v = ref_pred[:,:6] - dl_ko_pred[:,:6]
        delta_dl_p = ref_pred[:,6:12] - dl_ko_pred[:,6:12]
        delta_dl_d = ref_pred[:,12:17] - dl_ko_pred[:,12:17]

        delta_tw_v = ref_pred[:,:6] - tw_ko_pred[:,:6]
        delta_tw_p = ref_pred[:,6:12] - tw_ko_pred[:,6:12]
        delta_tw_d = ref_pred[:,12:17] - tw_ko_pred[:,12:17]

        delta_sn_v = ref_pred[:,:6] - sn_ko_pred[:,:6]
        delta_sn_p = ref_pred[:,6:12] - sn_ko_pred[:,6:12]
        delta_sn_d = ref_pred[:,12:17] - sn_ko_pred[:,12:17]

        delta_dl_v = tf.reduce_mean(delta_dl_v, axis = 1, keepdims = True)
        delta_dl_p = tf.reduce_mean(delta_dl_p, axis = 1, keepdims = True)
        delta_dl_d = tf.reduce_mean(delta_dl_d, axis = 1, keepdims = True)

        delta_tw_v = tf.reduce_mean(delta_tw_v, axis = 1, keepdims = True)
        delta_tw_p = tf.reduce_mean(delta_tw_p, axis = 1, keepdims = True)
        delta_tw_d = tf.reduce_mean(delta_tw_d, axis = 1, keepdims = True)

        delta_sn_v = tf.reduce_mean(delta_sn_v, axis = 1, keepdims = True)
        delta_sn_p = tf.reduce_mean(delta_sn_p, axis = 1, keepdims = True)
        delta_sn_d = tf.reduce_mean(delta_sn_d, axis = 1, keepdims = True)

        delta_dl = tf.concat([delta_dl_v, delta_dl_p, delta_dl_d], axis = 1)
        delta_tw = tf.concat([delta_tw_v, delta_tw_p, delta_tw_d], axis = 1)
        delta_sn = tf.concat([delta_sn_v, delta_sn_p, delta_sn_d], axis = 1)

        ret = tf.stack([delta_dl, delta_tw, delta_sn], axis = 2)
        return ret.numpy()

    def save(self, epoch):
        """
        Saves checkpont (model and optimizer states) and losses.
        If predicted TF roles are provided, it saves them as well!
        """

        # save model
        self.manager.save(epoch)

        # save accuracies
        df = pd.DataFrame([[epoch, self.loss_train, self.loss_valid, self.loss_test]])
        with open(self.flags_.o + '/loss.tab', 'a') as fw:
            hdr = False if fw.tell() != 0 else ['Epoch', 'Train loss', 'Valid loss', 'Test loss']
            df.to_csv(fw, index = False, header = hdr, sep = '\t')

    def restore(self, epoch = None):
        if epoch:
            self.ckpt.restore(self.flags_.o + '/checkpoints/ckpt-' + str(epoch))
        else:
            self.ckpt.restore(self.manager.latest_checkpoint)

        return self.ckpt.step.numpy()

    def scale_LR(self, factor):
        """
        scales the learning rate by the provided factor
        """
        old_LR = self.optimizer.lr.read_value()
        self.optimizer.lr.assign(factor * old_LR)


    def predict(self, seq, conc):
        return self.model(inputs = (seq, conc), training=False)
