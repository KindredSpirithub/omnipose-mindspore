import os
from pickle import FALSE
import time
import argparse
import numpy as np
import mindspore
from mindspore import ops
from mindspore import context   
import mindspore.nn as nn
from mindspore import Tensor, float32, context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from dataset.dataset import *
from src.omnipose  import get_omnipose
# from src.simplepose import get_pose_net
from src.loss import *
from src.predict import get_final_preds
from src.utils.transform import flip_back
from config import cfg
from config import update_config
from src.evaluate.coco_eval import evaluate
import warnings
warnings.filterwarnings("ignore") 

def validate(cfg, val_dataset, model, output_dir):
    # switch to evaluate mode
    model.set_train(False)

    # init record
    num_samples = val_dataset.get_dataset_size() * cfg.TEST.BATCH_SIZE
    all_preds = np.zeros((num_samples, cfg.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 2))
    image_id = []
    idx = 0

    # start eval
    start = time.time()
    for item in val_dataset.create_dict_iterator():
        # input data
        inputs = item['image'].asnumpy()
        # compute output
        output = model(Tensor(inputs, float32)).asnumpy()
        if cfg.TEST.FLIP_TEST:
            inputs_flipped = Tensor(inputs[:, :, :, ::-1], float32)
            output_flipped = model(inputs_flipped)
            output_flipped = flip_back(output_flipped.asnumpy(), flip_pairs)

            # feature is not aligned, shift flipped heatmap for higher accuracy
            SHIFT_HEATMAP=True
            if SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.copy()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        # meta data
        c = item['center'].asnumpy()
        s = item['scale'].asnumpy()
        score = item['score'].asnumpy()
        file_id = list(item['id'].asnumpy())

        # pred by heatmaps
        preds, maxvals = get_final_preds(cfg, output.copy(), c, s)

        num_images, _ = preds.shape[:2]
        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 1] = score

        image_id.extend(file_id)
        idx += num_images
        if idx % 1024 == 0:
            print('{} samples validated in {} seconds'.format(idx, time.time() - start))
            start = time.time()

    print(all_preds[:idx].shape, all_boxes[:idx].shape, len(image_id))
    _, perf_indicator = evaluate(
        cfg, all_preds[:idx], output_dir, all_boxes[:idx], image_id)

    print("AP:", perf_indicator)
    return perf_indicator

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',          help='experiment configure file name',
                        default='experiments/coco/omnipose_w48_384x288.yaml', type=str)
    parser.add_argument('--opts',         help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--modelDir',     help='model directory', type=str, default='')
    parser.add_argument('--logDir',       help='log directory', type=str, default='')
    parser.add_argument('--dataDir',      help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')

    args = parser.parse_args()
    return args


def main(args):
    device = int(os.getenv('DEVICE_ID', '2'))

    update_config(cfg, args)

    # context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="GPU", device_id=0)
    context.set_context(mode=context.PYNATIVE_MODE, save_graphs=False, device_target="GPU", device_id=device)

    rank=0
    device_num=1
    NUM_WORKERS = 4
    eval_set, _ = keypoint_dataset(cfg,
                                  rank=rank,
                                  group_size=device_num,
                                  train_mode=False,
                                  num_parallel_workers=NUM_WORKERS)
    steps_per_epoch = eval_set.get_dataset_size()

    network = get_omnipose(cfg, is_train=False)
    # network = get_pose_net(False, ckpt_path='./models/omnipose-1_7014.ckpt')

    ckpt_dir = "./ckpt/simplepose_7-23_9014.ckpt"
    print(ckpt_dir)
    param_dict = load_checkpoint(ckpt_file_name=ckpt_dir)
    load_param_into_net(network, param_dict)
    print("start eval")

    # evaluate on validation set
    output_dir = ckpt_dir.split('.')[0]
    validate(cfg, eval_set, network, output_dir)
            

if __name__ == "__main__":
    arg = parse_args()
    main(arg)