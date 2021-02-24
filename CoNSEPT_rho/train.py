from Seq_Scan import Seq
from S2E import seq2expr
from utils import early_stop, save_pickle, save_expr
from absl import flags
from absl import logging
from absl import app
from utils import make_dir
import numpy as np


FLAGS = flags.FLAGS

flags.DEFINE_string('sf', None,'path to sequence file containing all training, testing and validation sequences')
flags.DEFINE_string('pwm', None,'path to pwm file')
flags.DEFINE_string('ef', None,'path to expression file')
flags.DEFINE_string('tf', None,'path to TF concentration file')
flags.DEFINE_integer('ne', None,'number of enhancers')
flags.DEFINE_integer('nb', None,'number of bins')
flags.DEFINE_integer('nTrain', None,'the first nTrain enhancers are used for training')
flags.DEFINE_integer('nValid', None,'the next nValid enahncers are used for validating')
flags.DEFINE_integer('nTest', None,'the last nTest enahncers are used for testing')
flags.DEFINE_boolean('restore', False, 'whether restore the last save model from checkpoints | Default: False')
flags.DEFINE_integer('nEpoch', None,'number of epochs')
flags.DEFINE_integer('bs', 20,'batch size | Default: 20')
flags.DEFINE_integer('save_freq', 4,'saving frequency | Default: 4')
flags.DEFINE_string('o', None,'path to output directory')
flags.DEFINE_string('nChans', None,'number of channels for cooperativity layers, specify one number per layer, seperate values with comma')
flags.DEFINE_float('dr', None,'dropout rate')
flags.DEFINE_string('psb', None,'pool size after binding, specify with two comma seperated integers')
flags.DEFINE_string('csc', None,'convolution size cooperativity layers, specify with two comma seperated integers')
flags.DEFINE_string('sc', None,'stride of cooperativity layer, specify with two comma seperated integers')
flags.DEFINE_string('cAct', None,'activation function for cooperativity layer')
flags.DEFINE_string('oAct', None,'activation function for output layer')

flags.DEFINE_boolean('predict', False,'Whether use model to predict only (no training), if True need to provide checkpoint to read')
flags.DEFINE_integer('ckpt', None,'the checkpoint to be read for prediction')
flags.DEFINE_string('pred_dir', None,'the folder name inside the specified output directory to write predictions')




def main(argv):

    """
    Prepare data
    """
    # This data is not augmented (with offsets)
    print("start building the data")
    data = Seq(seq_file = FLAGS.sf, PWM = FLAGS.pwm,
        expression_file = FLAGS.ef, TF_file = FLAGS.tf, nEnhancers = FLAGS.ne,
        nBins = FLAGS.nb, nTrain = FLAGS.nTrain, nValid = FLAGS.nValid,
        nTest = FLAGS.nTest, training = False)
    print("done building the data")
    if not FLAGS.predict:
        train_seq, train_TF, train_rho = data.next_batch(all_data = 'train')
        valid_seq, valid_TF, valid_rho = data.next_batch(all_data = 'valid')
        test_seq, test_TF, test_rho = data.next_batch(all_data = 'test')

    all_seq, all_TF, all_rho = data.next_batch(all_data = 'all')

    # This data is augmented (with offsets) for training
    if not FLAGS.predict:
        data = Seq(seq_file = FLAGS.sf, PWM = FLAGS.pwm,
            expression_file = FLAGS.ef, TF_file = FLAGS.tf, nEnhancers = FLAGS.ne,
            nBins = FLAGS.nb, nTrain = FLAGS.nTrain, nValid = FLAGS.nValid,
            nTest = FLAGS.nTest, training = True)


    """
    Make directories
    """
    #make_dir(FLAGS.o)
    make_dir(FLAGS.o)
    make_dir(FLAGS.o + '/checkpoints')
    make_dir(FLAGS.o + '/TF_roles')
    if FLAGS.predict:
        make_dir(FLAGS.o + '/' + FLAGS.pred_dir)


    """
    Define model
    """
    model = seq2expr(FLAGS)

    if FLAGS.predict:
        model.restore(epoch = FLAGS.ckpt)
        pred_expr = model.predict(seq = all_seq, conc = all_TF)
        pred_expr = np.reshape(pred_expr, (-1,17))
        save_expr(FLAGS.o + '/' + FLAGS.pred_dir +'/predictions_epoch_' + str(FLAGS.ckpt) + '.csv', pred_expr)
        return


    if FLAGS.restore:
        epoch = model.restore() + 1
    else:
        epoch = 0

    """
    Training
    """
    #ToDo: If a better functionality for resuming training is required, add step
    # to checkpoint to keep tack of trained steps and better handle the remaining
    # required steps

    terminate = False
    reduced_LR_count = 0
    while (not terminate and epoch < FLAGS.nEpoch):
        seq_batches, TF_batches, rho_batches = data.next_batch(size = FLAGS.bs)
        for currSeq, currTF, currRho in zip (seq_batches, TF_batches, rho_batches):
            model.train_step(seq = currSeq, conc = currTF, gt_expr = currRho)

        model.collect_loss_train()
        model.valid_step(seq = valid_seq, conc = valid_TF, gt_expr = valid_rho)

        if (epoch + 1) % FLAGS.save_freq == 0:
            model.test_step(seq = test_seq, conc = test_TF, gt_expr = test_rho)
            model.save(epoch)

            TF_roles_train = model.compute_TF_roles(seq = train_seq, conc = train_TF)
            TF_roles_valid = model.compute_TF_roles(seq = valid_seq, conc = valid_TF)
            TF_roles_test = model.compute_TF_roles(seq = test_seq, conc = test_TF)

            #TODO: make TF_roles directory
            path_write = FLAGS.o + '/TF_roles/'

            save_pickle(path_write + 'train_epoch_' + str(epoch) + '.pkl', TF_roles_train)
            save_pickle(path_write + 'valid_epoch_' + str(epoch) + '.pkl', TF_roles_valid)
            save_pickle(path_write + 'test_epoch_' + str(epoch) + '.pkl', TF_roles_test)

        if model.loss_train < 0.007 and reduced_LR_count < 1:
            model.scale_LR(0.1)
            reduced_LR_count += 1
        elif model.loss_train < 0.006 and reduced_LR_count < 2:
            model.scale_LR(0.1)
            reduced_LR_count += 1

        epoch += 1
        model.ckpt.step.assign_add(1)
        #For now I disbaled early stopping
        #terminate = early_stop(model.loss_train, model.loss_valid, 0.0061)


    # predict expression after training
    pred_expr = model.predict(seq = all_seq, conc = all_TF)
    pred_expr = np.reshape(pred_expr, (-1,17))
    save_expr(FLAGS.o + '/predicted_exprs.csv', pred_expr)


if __name__ == "__main__":
    app.run(main)
