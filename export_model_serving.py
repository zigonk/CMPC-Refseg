from __future__ import division

import sys
import os
import argparse
import tensorflow as tf
import skimage
from skimage import io as sio
import time
# import matplotlib.pyplot as plt
from get_model import get_segmentation_model
from pydensecrf import densecrf

from util import data_reader_refvos, data_reader
from util.processing_tools import *
from util import im_processing, eval_tools, MovingAverage


def export_model(dataset, tfmodel_folder, model_name, pre_emb=False):
    global args
    weights = tfmodel_folder
    print("Loading trained weights from {}".format(weights))

    H, W = 320, 320
    vocab_size = 8803 if dataset == 'referit' else 1917498
    emb_name = dataset


    if pre_emb:
        # use pretrained embbeding
        print("Use pretrained Embeddings.")
        model = get_segmentation_model(model_name, H=H, W=W,
                                       mode='eval', 
                                       vocab_size=vocab_size, 
                                       emb_name=emb_name, 
                                       emb_dir=args.embdir)
    else:
        model = get_segmentation_model(model_name, H=H, W=W,
                                       mode='eval', vocab_size=vocab_size)

    snapshot_restorer = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    snapshot_restorer.restore(sess, weights)
    # Log tensorboard
    export_path_base = sys.argv[-1]
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(args.version)))
    print('Exporting trained model to', export_path)
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.

    tensor_info_imgs = tf.compat.v1.saved_model.utils.build_tensor_info(model.im)
    tensor_info_texts = tf.compat.v1.saved_model.utils.build_tensor_info(model.words)
    tensor_info_seq_len = tf.compat.v1.saved_model.utils.build_tensor_info(model.seq_len)
    tensor_info_preds = tf.compat.v1.saved_model.utils.build_tensor_info(model.sigm)

    prediction_signature = (
        tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
            inputs={
                'images': tensor_info_imgs,
                'sentences': tensor_info_texts,
                'sequence_lenghts': tensor_info_seq_len
            },
            outputs={'masks': tensor_info_preds},
            method_name=tf.compat.v1.saved_model.signature_constants
            .PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature
        },
        main_op=tf.compat.v1.tables_initializer(),
        strip_default_attrs=True)

    builder.save()

    print('Done exporting!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type=str, default='0')
    parser.add_argument('-i', type=int, default=800000)
    parser.add_argument('-s', type=int, default=100000)
    parser.add_argument('-lastiter', type=int, default=0) #last iter for continue training
    parser.add_argument('-st', type=int, default=700000)  # stop training when get st iters
    parser.add_argument('-m', type=str)  # 'train' 'test'
    parser.add_argument('-d', type=str, default='referit')  # 'Gref' 'unc' 'unc+' 'referit'
    parser.add_argument('-t', type=str)  # 'train' 'trainval' 'val' 'test' 'testA' 'testB'
    parser.add_argument('-f', type=str)  # directory to save models
    parser.add_argument('-lr', type=float, default=0.00025)  # start learning rate
    parser.add_argument('-bs', type=int, default=1)  # batch size
    parser.add_argument('-datadir', type=str, default='./')
    parser.add_argument('-pretrain', type=str, default='')
    parser.add_argument('-finetune', default=False, action='store_true') 
    parser.add_argument('-v', default=False, action='store_true')  # visualization
    parser.add_argument('-c', default=False, action='store_true')  # whether or not apply DenseCRF
    parser.add_argument('-emb', default=False, action='store_true')  # whether or not use Pretrained Embeddings
    parser.add_argument('-embdir', type=str, default='')  # whether or not use Pretrained Embeddings
    parser.add_argument('-n', type=str, default='')  # select model
    parser.add_argument('-conv5', default=False, action='store_true')  # finetune conv layers
    parser.add_argument('-freeze_bn', default=False, action='store_true')  # finetune conv layers
    parser.add_argument('-is_aug', default=False, action='store_true')  # finetune conv layers
    parser.add_argument('-log_dir', type=str, default='./logdir')
    parser.add_argument('-im_dir', type=str, default='')
    parser.add_argument('-mask_dir', type=str, default='')
    parser.add_argument('-meta', type=str, default='./train_meta.json')
    parser.add_argument('-version', type=str, default='1')

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    mu = np.array((104.00698793, 116.66876762, 122.67891434))

    export_model(dataset=args.d,
                tfmodel_folder=args.f,
                model_name=args.n,
                stop_iter=args.st,
                pre_emb=args.emb)