import os
import time
import numpy as np

from mindspore import Tensor, float32, context
from mindspore.common import set_seed
from mindspore.train.callback import Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from dataset.dataset import flip_pairs
from src.evaluate.coco_eval import evaluate
from src.utils.transform import flip_back
from src.predict import get_final_preds
# from src.config import config as cfg

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
    for item in val_dataset.create_dict_iterator(num_epochs=1):
        # input data
        inputs = item['image'].asnumpy()
        # compute output
        output = model(Tensor(inputs, float32)).asnumpy()
        if cfg.TEST.FLIP_TEST:
            inputs_flipped = Tensor(inputs[:, :, :, ::-1], float32)
            output_flipped = model(inputs_flipped)
            output_flipped = flip_back(output_flipped.asnumpy(), flip_pairs)

            # feature is not aligned, shift flipped heatmap for higher accuracy
            SHIFT_HEATMAP = False
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
        if idx % 3000 == 0:
            print('{} samples validated in {} seconds'.format(idx, time.time() - start))
            start = time.time()

    print(all_preds[:idx].shape, all_boxes[:idx].shape, len(image_id))
    _, perf_indicator = evaluate(
        cfg, all_preds[:idx], output_dir, all_boxes[:idx], image_id)
    print("AP:", perf_indicator)
    return perf_indicator



class EvaluateCallBack(Callback):
    """EvaluateCallBack"""

    def __init__(self, model, eval_dataset, loss_fn, cfg, log_dir=None):
        super(EvaluateCallBack, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.total_epochs = cfg.TRAIN.END_EPOCH
        self.loss = loss_fn
        self.save_freq = 1
        self.cfg=cfg
        self.log_dir=log_dir
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.cur_epoch_num
        if cur_epoch_num > self.total_epochs * 0.8 or cur_epoch_num % self.save_freq == 0:
            AP = validate(cfg=self.cfg, model=self.model, val_dataset=self.eval_dataset,output_dir=self.cfg.OUTPUT_DIR)
            # if AP >self.cfg.BEST_AP:
            #     self.cfg.BEST_EPOCH = cur_epoch_num
            #     self.cfg.BEST_AP = AP
            line = "epoch: %d, AP: %.4f, \n" % (cur_epoch_num, AP)
            print(line)
            with open(self.log_dir+'/logs.txt', 'a') as f:
                f.write(line)
            # print("best epoch : ", self.cfg.BEST_EPOCH,"best AP : ", self.cfg.BEST_AP)
    '''
    def step_end(self, run_context):
        AP = validate(cfg=cfg, model=self.model, val_dataset=self.eval_dataset,output_dir=cfg.OUTPUT_DIR)

        print("AP : ",AP)
    '''



