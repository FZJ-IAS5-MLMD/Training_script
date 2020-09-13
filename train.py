########Settings to shutup tensorflow###########################################################################
# WARNING:tensorflow:Entity <..>> could not be transformed and will be executed as-is
# to stop this warning, downgrade gast
# pip install gast==0.2.2

import sys; sys.path.insert(0, '/home/h2/hpclab12/bin/mimicpy')
import sys; sys.path.insert(0, '/home/h2/hpclab12/bin/pinn_tf2')

import os, warnings
warnings.filterwarnings('ignore') # stop future warnings

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# even after all this, deprication warning will appear
# to prevent this, set loggin level to ERROR below
tf.logging.set_verbosity(tf.logging.DEBUG)
# however, this will stop all input to tensorboard

#For optimization
#os.environ['CUDA_VISIBLE_DEVICES'] = '0' # required for tf to use gpu
#os.environ['TF_XLA_FLAGS']='--tf_xla_auto_jit=1 /home/raghavan/pinn_test_mimic' # accelerate tf models, stops warning
###############################################################################################################

########Global settings for training###########################################################################
num_epochs = 1
split={'train': 8, 'vali': 1, 'test': 1}
batch_size = {'train': 10, 'vali': 10, 'test': 10}
mpt = ['data/mimic.mpt', 'data/mimic.mpt', 'data/mimic.mpt', 'data/mimic.mpt', 'data/mimic.mpt']
trr = ['data/mimic.trr', 'data/mimic.trr', 'data/mimic.trr', 'data/mimic.trr', 'data/mimic.trr']
ener = ['data/ENERGIES', 'data/ENERGIES', 'data/ENERGIES', 'data/ENERGIES', 'data/ENERGIES']
model_dir = 'model'
###############################################################################################################

########Helper Funcstions/Classes##############################################################################
from pinn.io import sparse_batch
from pinn.models import potential_model
from pinn.io import load_mimic, distribute_data
from pinn.models import potential_model
from pinn.networks import pinet
from pinn.utils import hvd
from pinn.io.trr import get_trr_frames
num_samples = sum([get_trr_frames(i) for i in trr])

import math
import time

chkpts = []

class SessHook(tf.train.SessionRunHook):
    
    def __init__(self, mode, rate=1):
        mode_len = math.ceil( num_samples*split[mode]/sum(split.values()) )
        self._n_batchs = math.ceil( mode_len/batch_size[mode] )
        self._steps = num_epochs*self._n_batchs

        self._epoch = 0
        self._rate = rate

    def begin(self, rate=1):
        self._global_step_t = tf.train.get_or_create_global_step()

    def _new_epoch(self):
        self._print(mode='epoch')
        self._step = 0
        self._line()
        self._print(mode='heading')

    def _print(self, mode=None):
        if mode=='epoch':
            print("\n========>EPOCH NUMBER {:d}<========\n".format(self._epoch))
        elif mode=='heading':
            print("| {:^10} | {:^15} |".format("Step", "Time(s)"))
        elif mode=='title':
            print(" *** Evaluating Checkpoint {:d} ***\n".format(self._global_step))
        else:
            print('| {:^10d} | {:^15f} |'.format(self._step, self._curr_time-self.time))
            
    def _line(self):
        print("--------------------------------")

    def after_create_session(self, sess, coord):
        self._global_step = sess.run(self._global_step_t)
        self._print(mode='title')
        self._step = 0
        self.time = time.time()
        self._epoch += 1
        self._new_epoch()

    def before_run(self, run_context):
        if self._step == self._n_batchs:
            self._line()
            self._epoch += 1
            if self._epoch > num_epochs:
                run_context.request_stop()
            self._new_epoch()

        self._step += 1
        self._line()

    def after_run(self, run_context, run_values):
        self._curr_time = time.time()
        self._global_step = run_context.session.run(self._global_step_t)
        if self._step%self._rate == 0 or self._step == 1:
            self._print()
        self.time = self._curr_time

    def end(self, sess):
        self._line()
        global_step = sess.run(self._global_step_t)
        print('\n           *** Finished ***'.format(global_step))
    
class TrainHook(SessHook):
    def __init__(self):
        super().__init__('train')
        print("Initiating training with following parameters:")
        print("---------------------------")
        print("| {:15s} | {:5d} |".format("No. of Batches", self._n_batchs))
        print("---------------------------")
        print("| {:15s} | {:5d} |".format("No. of Epochs", num_epochs))
        print("---------------------------")
        print("| {:15s} | {:5d} |".format("No. of Steps", self._steps))
        print("---------------------------\n")

    def _print(self, mode=None):
        if mode=='epoch':
            print("\n===============>EPOCH NUMBER {:d}<==============\n".format(self._epoch))
        elif mode=='heading':
            print("| {:^10} | {:^10} | {:^15} |".format("Local", "Global", "Time(s)"))
        elif mode=='title':
            print("\n   *** Training started from Step {:d} ***\n".format(self._global_step))
        else:
            print('| {:^10d} | {:^10d} | {:^15f} |'.format(self._step, self._global_step, self._curr_time-self.time))

    def _line(self):
        print("---------------------------------------------")

class CheckPtLogger(tf.estimator.CheckpointSaverListener):
    def begin(self):
        chkpts = []

    def before_save(self, session, global_step_value):
        chkpts.append(str(global_step_value))

    def end(self, session, global_step_value):
        print("---------------------------------------------")
        if len(chkpts) == 1:
            print('     Wrote checkpoint at step '+( ', '.join(chkpts) ) )
        else:
            print('     Wrote checkpoints at steps '+( ', '.join(chkpts) ) )

def evalResult(ev):
    print("\nEvaluation result:")
    print("---------------------------------------")
    for k, v in ev.items():
        if 'METRICS' not in k: continue
        else: print("| {:15s} | {:17f} |".format(k.split('/')[1], v))
        print("---------------------------------------")
    print('\n')

###############################################################################################################


########Training and Evaluation################################################################################

print("Loaded tensorflow..\nStarting..")

pre_fn = lambda tensors: pinet(tensors, preprocess=True)

mpt_chunks, trr_chunks, ener_chunks = distribute_data(mpt, trr, ener)
print(f"Distributed data at rank {hvd.rank()}: f{(mpt_chunks, trr_chunks, ener_chunks)}")
dataset = lambda: load_mimic(mpt_chunks, trr_chunks, ener_chunks, split=split)

train = lambda: dataset()['train'].cache().repeat().shuffle(1000).\
        apply(sparse_batch(batch_size['train'])).map(pre_fn, 8)
valid = lambda: dataset()['vali'].cache().repeat().apply(sparse_batch(batch_size['vali'])).map(pre_fn, 8)
test = lambda: dataset()['test'].cache().repeat().apply(sparse_batch(batch_size['test'])).map(pre_fn, 8)

params = {'model_dir': model_dir,
          'network': 'pinet',
          'parallel': True,
          'network_params': {},
          'model_params': {'learning_rate': 1e-4, 'decay_step':10, 'decay_rate': 0.70}}

config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True

config = tf.estimator.RunConfig(log_step_count_steps=1, session_config=config_proto, save_summary_steps=1, keep_checkpoint_max=None, save_checkpoints_steps=2)
model = potential_model(params, config=config)


model.train(input_fn=train, steps=5, hooks=[TrainHook()], saving_listeners=[CheckPtLogger()])

###############################################################################################################
