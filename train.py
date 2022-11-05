# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import time
import numpy as np

from mindspore import context, Tensor, Parameter
from mindspore.train import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.nn.optim import Adam
from mindspore.common import set_seed
from src.omnipose import get_omnipose
from dataset.dataset import keypoint_dataset
from src.loss import *
from src.eval import EvaluateCallBack
from config import cfg
from config import update_config
import warnings
warnings.filterwarnings("ignore") 
set_seed(1)


def get_lr(begin_epoch,
           total_epochs,
           steps_per_epoch,
           lr_init=0.1,
           factor=0.1,
           epoch_number_to_drop=(90, 120)
           ):
    """
    Generate learning rate array.

    Args:
        begin_epoch (int): Initial epoch of training.
        total_epochs (int): Total epoch of training.
        steps_per_epoch (float): Steps of one epoch.
        lr_init (float): Initial learning rate. Default: 0.316.
        factor:Factor of lr to drop.
        epoch_number_to_drop:Learing rate will drop after these epochs.
    Returns:
        np.array, learning rate array.
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    step_number_to_drop = [steps_per_epoch * x for x in epoch_number_to_drop]
    for i in range(int(total_steps)):
        if i in step_number_to_drop:
            lr_init = lr_init * factor
        lr_each_step.append(lr_init)
    current_step = steps_per_epoch * begin_epoch
    lr_each_step = np.array(lr_each_step, dtype=np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate
    
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',          help='experiment configure file name',
                        default='experiments/coco/omnipose_w48_384x288.yaml', type=str)

    parser.add_argument('--modelDir',     help='model directory', type=str, default='')
    parser.add_argument('--logDir',       help='log directory', type=str, default='')
    parser.add_argument('--dataDir',      help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')

    args = parser.parse_args()
    return args


def run_train(args):
    update_config(cfg, args)
    rank = 0
    device_num = 1

    # only rank = 0 can write
    rank_save_flag = False
    if rank == 0 or device_num == 1:
        rank_save_flag = True

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, device_id=0)
    # context.set_context(mode=context.PYNATIVE_MODE, save_graphs=False, device_target="GPU", device_id=2)
    NUM_WORKERS = 4 
    train_dataset, _ = keypoint_dataset(cfg,
                                  rank=rank,
                                  group_size=device_num,
                                  train_mode=True,
                                  num_parallel_workers=NUM_WORKERS)

    val_dataset, _ = keypoint_dataset(cfg,
                                  train_mode=False,
                                  num_parallel_workers=NUM_WORKERS)
    # network
    net = get_omnipose(cfg, is_train=True)
    
    loss = JointsMSELoss(use_target_weight=True)
    net_with_loss = WithLossCell(net, loss)

    # lr schedule and optim
    dataset_size = train_dataset.get_dataset_size()
    # LR_STEP= [90, 120]
    lr = Tensor(get_lr(0,
                       cfg.TRAIN.END_EPOCH,
                       dataset_size,
                       lr_init=cfg.TRAIN.LR,
                       factor=cfg.TRAIN.LR_FACTOR,
                       epoch_number_to_drop=cfg.TRAIN.LR_STEP))

    opt = Adam(net.trainable_params(), learning_rate=lr)

    # callback
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossMonitor(per_print_times=100)
    cb = [time_cb, loss_cb]

    ckpt_save_dir = "./ckpt"
    config_ck = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix="simplepose", directory=ckpt_save_dir, config=config_ck)
    cb.append(ckpoint_cb)

    eval_cb = EvaluateCallBack(model=net, eval_dataset=val_dataset, loss_fn=loss, cfg=cfg)
    cb.append(eval_cb)

    # train model
    model = Model(net_with_loss, loss_fn=None, optimizer=opt, amp_level="O2")
    epoch_size = cfg.TRAIN.END_EPOCH - cfg.TRAIN.BEGIN_EPOCH
    print('start training, epoch size = %d' % epoch_size)
    model.train(epoch_size, train_dataset, callbacks=cb, dataset_sink_mode=False)

if __name__ == '__main__':
    arg = parse_args()
    run_train(arg)
