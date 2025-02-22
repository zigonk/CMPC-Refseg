from __future__ import division

import sys
import os
import argparse
import tensorflow as tf
import skimage
from skimage import io as sio
import time
import cv2
import json
from PIL import Image
# import matplotlib.pyplot as plt
from get_model import get_segmentation_model
from pydensecrf import densecrf

from util import data_reader
from util.processing_tools import *
from util import im_processing, eval_tools, MovingAverage, text_processing


def train(max_iter, snapshot, dataset, setname, mu, lr, bs, tfmodel_folder,
          conv5, model_name, stop_iter, pre_emb=False):
    iters_per_log = 100
    data_folder = './' + dataset + '/' + setname + '_batch/'
    data_prefix = dataset + '_' + setname
    snapshot_file = os.path.join(tfmodel_folder, dataset + '_iter_%d.tfmodel')
    if not os.path.isdir(tfmodel_folder):
        os.makedirs(tfmodel_folder)

    cls_loss_avg = 0
    avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0, 0, 0
    decay = 0.99
    vocab_size = 8803 if dataset == 'referit' else 12112
    emb_name = 'referit' if dataset == 'referit' else 'Gref'

    if pre_emb:
        print("Use pretrained Embeddings.")
        model = get_segmentation_model(model_name, mode='train',
                                       vocab_size=vocab_size, start_lr=lr,
                                       batch_size=bs, conv5=conv5, emb_name=emb_name)
    else:
        model = get_segmentation_model(model_name, mode='train',
                                       vocab_size=vocab_size, start_lr=lr,
                                       batch_size=bs, conv5=conv5)

    weights = './data/weights/deeplab_resnet_init.ckpt'
    print("Loading pretrained weights from {}".format(weights))
    load_var = {var.op.name: var for var in tf.global_variables()
                if var.name.startswith('res') or var.name.startswith('bn') or var.name.startswith('conv1')}

    snapshot_loader = tf.train.Saver(load_var)
    snapshot_saver = tf.train.Saver(max_to_keep=4)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    snapshot_loader.restore(sess, weights)

    im_h, im_w, num_steps = model.H, model.W, model.num_steps
    text_batch = np.zeros((bs, num_steps), dtype=np.float32)
    image_batch = np.zeros((bs, im_h, im_w, 3), dtype=np.float32)
    mask_batch = np.zeros((bs, im_h, im_w, 1), dtype=np.float32)
    valid_idx_batch = np.zeros((bs, 1), dtype=np.int32)

    reader = data_reader.DataReader(data_folder, data_prefix)

    # for time calculate
    last_time = time.time()
    time_avg = MovingAverage()
    for n_iter in range(max_iter):

        for n_batch in range(bs):
            batch = reader.read_batch(is_log=(n_batch == 0 and n_iter % iters_per_log == 0))
            text = batch['text_batch']
            im = batch['im_batch'].astype(np.float32)
            mask = np.expand_dims(batch['mask_batch'].astype(np.float32), axis=2)

            im = im[:, :, ::-1]
            im -= mu

            text_batch[n_batch, ...] = text
            image_batch[n_batch, ...] = im
            mask_batch[n_batch, ...] = mask

            for idx in range(text.shape[0]):
                if text[idx] != 0:
                    valid_idx_batch[n_batch, :] = idx
                    break

        _, cls_loss_val, lr_val, scores_val, label_val = sess.run([model.train_step,
                                                                   model.cls_loss,
                                                                   model.learning_rate,
                                                                   model.pred,
                                                                   model.target],
                                                                  feed_dict={
                                                                      model.words: text_batch,
                                                                      # np.expand_dims(text, axis=0),
                                                                      model.im: image_batch,
                                                                      # np.expand_dims(im, axis=0),
                                                                      model.target_fine: mask_batch,
                                                                      # np.expand_dims(mask, axis=0)
                                                                      model.valid_idx: valid_idx_batch
                                                                  })
        cls_loss_avg = decay * cls_loss_avg + (1 - decay) * cls_loss_val

        # Accuracy
        accuracy_all, accuracy_pos, accuracy_neg = compute_accuracy(scores_val, label_val)
        avg_accuracy_all = decay * avg_accuracy_all + (1 - decay) * accuracy_all
        avg_accuracy_pos = decay * avg_accuracy_pos + (1 - decay) * accuracy_pos
        avg_accuracy_neg = decay * avg_accuracy_neg + (1 - decay) * accuracy_neg

        # timing
        cur_time = time.time()
        elapsed = cur_time - last_time
        last_time = cur_time

        if n_iter % iters_per_log == 0:
            print('iter = %d, loss (cur) = %f, loss (avg) = %f, lr = %f'
                  % (n_iter, cls_loss_val, cls_loss_avg, lr_val))
            print('iter = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
                  % (n_iter, accuracy_all, accuracy_pos, accuracy_neg))
            print('iter = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
                  % (n_iter, avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg))
            time_avg.add(elapsed)
            print('iter = %d, cur time = %.5f, avg time = %.5f, model_name: %s' % (n_iter, elapsed, time_avg.get_avg(), model_name))

        # Save snapshot
        if (n_iter + 1) % snapshot == 0 or (n_iter + 1) >= max_iter:
            snapshot_saver.save(sess, snapshot_file % (n_iter + 1))
            print('snapshot saved to ' + snapshot_file % (n_iter + 1))
        if (n_iter + 1) >= stop_iter:
            print('stop training at iter ' + str(stop_iter))
            break

    print('Optimization done.')

def load_image(img_path):
    if (not os.path.exists(img_path)):
        return None
    return np.asarray(Image.open(img_path))

def load_frame_from_id(vid, frame_id):
    frame_path = os.path.join(args.imdir, str('{}/{}.jpg'.format(vid, frame_id)))
    return load_image(frame_path)

def test(iter, dataset, visualize, setname, dcrf, mu, tfmodel_folder, model_name, pre_emb=False):
    data_folder = './' + dataset + '/' + setname + '_batch/'
    data_prefix = dataset + '_' + setname
    if visualize:
        save_dir = './' + dataset + '/visualization/' + str(iter) + '/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    weights = os.path.join(tfmodel_folder, dataset + '_iter_' + str(iter) + '.tfmodel')
    print("Loading trained weights from {}".format(weights))

    score_thresh = 1e-9
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    cum_I, cum_U = 0, 0
    mean_IoU, mean_dcrf_IoU = 0, 0
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    if dcrf:
        cum_I_dcrf, cum_U_dcrf = 0, 0
        seg_correct_dcrf = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0.
    T = 20 # truncated long sentence
    H, W = 320, 320
    vocab_size = 8803 if dataset == 'referit' else 12112
    emb_name = 'referit' if dataset == 'referit' else 'Gref'
    vocab_file = './data/vocabulary_Gref.txt'
    vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)
    IU_result = list()

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

    # Load pretrained model
    snapshot_restorer = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    snapshot_restorer.restore(sess, weights)
     
    meta_expression = {}
    with open(args.meta) as meta_file:
        meta_expression = json.load(meta_file)
    videos = meta_expression['videos']
    for vid_ind, vid in reversed(list(enumerate(videos.keys()))):  
        print("Running on video {}/{}".format(vid_ind + 1, len(videos.keys())))
        expressions = videos[vid]['expressions']
        # instance_ids = [expression['obj_id'] for expression_id in videos[vid]['expressions']]
        frame_ids = videos[vid]['frames']
        for eid in expressions:
            exp = expressions[eid]['exp']
            index = int(eid)
            vis_dir = os.path.join(args.visdir, str('{}/{}/'.format(vid, index)))
            mask_dir = os.path.join(args.maskdir, str('{}/{}/'.format(vid, index)))
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            avg_time = 0
            total_frame = 0
#             Process text
            text = np.array(text_processing.preprocess_sentence(exp, vocab_dict, T))
            valid_idx = np.zeros([1], dtype=np.int32)
            for idx in range(text.shape[0]):
                if text[idx] != 0:
                    valid_idx[0] = idx
                    break
            for fid in frame_ids:
                vis_path = os.path.join(vis_dir, str('{}.png'.format(fid)))
                mask_path = os.path.join(mask_dir, str('{}.npy'.format(fid)))
                if os.path.exists(vis_path):
                    continue
                frame = load_frame_from_id(vid, fid)
                if frame is None:
                    continue
                last_time = time.time()
#                 im = frame.copy()
                im = frame
#                 mask = np.array(frame, dtype=np.float32)

                proc_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, H, W))
                proc_im_ = proc_im.astype(np.float32)
                # proc_im_ = proc_im_[:, :, ::-1]
                proc_im_ -= mu
                scores_val, up_val, sigm_val = sess.run([model.pred, model.up, model.sigm],
                                                        feed_dict={
                                                            model.words: np.expand_dims(text, axis=0),
                                                            model.im: np.expand_dims(proc_im_, axis=0),
                                                            model.valid_idx: np.expand_dims(valid_idx, axis=0)
                                                        })
                # scores_val = np.squeeze(scores_val)
                # pred_raw = (scores_val >= score_thresh).astype(np.float32)
                up_val = np.squeeze(up_val)
                pred_raw = (up_val >= score_thresh).astype('uint8') * 255
#                 pred_raw = (up_val >= score_thresh).astype(np.float32)
#                 predicts = im_processing.resize_and_crop(pred_raw, mask.shape[0], mask.shape[1])
                if dcrf:
                    # Dense CRF post-processing
                    sigm_val = np.squeeze(sigm_val) + 1e-7
                    d = densecrf.DenseCRF2D(W, H, 2)
                    U = np.expand_dims(-np.log(sigm_val), axis=0)
                    U_ = np.expand_dims(-np.log(1 - sigm_val), axis=0)
                    unary = np.concatenate((U_, U), axis=0)
                    unary = unary.reshape((2, -1))
                    d.setUnaryEnergy(unary)
                    d.addPairwiseGaussian(sxy=3, compat=3)
                    d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=proc_im, compat=10)
                    Q = d.inference(5)
                    pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype('uint8') * 255
#                     pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)
#                     predicts_dcrf = im_processing.resize_and_crop(pred_raw_dcrf, mask.shape[0], mask.shape[1])
                if visualize:
                    if dcrf:
                        cv2.imwrite(vis_path, pred_raw_dcrf)
#                         np.save(mask_path, np.array(pred_raw_dcrf))
#                         visualize_seg(vis_path, im, exp, predicts_dcrf)
                    else:
                        np.save(mask_path, np.array(sigm_val))
#                         cv2.imwrite(vis_path, pred_raw)
#                         visualize_seg(vis_path, im, exp, predicts)
#                         np.save(mask_path, np.array(pred_raw))
    # I, U = eval_tools.compute_mask_IU(predicts, mask)
    # IU_result.append({'batch_no': n_iter, 'I': I, 'U': U})
    # mean_IoU += float(I) / U
    # cum_I += I
    # cum_U += U
    # msg = 'cumulative IoU = %f' % (cum_I / cum_U)
    # for n_eval_iou in range(len(eval_seg_iou_list)):
    #     eval_seg_iou = eval_seg_iou_list[n_eval_iou]
    #     seg_correct[n_eval_iou] += (I / U >= eval_seg_iou)
    # if dcrf:
    #     I_dcrf, U_dcrf = eval_tools.compute_mask_IU(predicts_dcrf, mask)
    #     mean_dcrf_IoU += float(I_dcrf) / U_dcrf
    #     cum_I_dcrf += I_dcrf
    #     cum_U_dcrf += U_dcrf
    #     msg += '\tcumulative IoU (dcrf) = %f' % (cum_I_dcrf / cum_U_dcrf)
    #     for n_eval_iou in range(len(eval_seg_iou_list)):
    #         eval_seg_iou = eval_seg_iou_list[n_eval_iou]
    #         seg_correct_dcrf[n_eval_iou] += (I_dcrf / U_dcrf >= eval_seg_iou)
    # print(msg)
    seg_total += 1

    # Print results
    # print('Segmentation evaluation (without DenseCRF):')
    # result_str = ''
    # for n_eval_iou in range(len(eval_seg_iou_list)):
    #     result_str += 'precision@%s = %f\n' % \
    #                   (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] / seg_total)
    # result_str += 'overall IoU = %f; mean IoU = %f\n' % (cum_I / cum_U, mean_IoU / seg_total)
    # print(result_str)
    # if dcrf:
    #     print('Segmentation evaluation (with DenseCRF):')
    #     result_str = ''
    #     for n_eval_iou in range(len(eval_seg_iou_list)):
    #         result_str += 'precision@%s = %f\n' % \
    #                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct_dcrf[n_eval_iou] / seg_total)
    #     result_str += 'overall IoU = %f; mean IoU = %f\n' % (cum_I_dcrf / cum_U_dcrf, mean_dcrf_IoU / seg_total)
    #     print(result_str)


def visualize_seg(vis_path, im, sent, predicts, mask=None):
    # print("visualizing")
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (30, 30)
    fontScale              = 1
    fontColor              = (255,0,0)
    lineType               = 2

    # Ignore sio warnings of low-contrast image.
    import warnings
    warnings.filterwarnings('ignore')

    sio.imsave(vis_path, im)

    # im_gt = np.zeros_like(im)
    # im_gt[:, :, 2] = 170
    # im_gt[:, :, 0] += mask.astype('uint8') * 170
    # im_gt = im_gt.astype('int16')
    # im_gt[:, :, 2] += mask.astype('int16') * (-170)
    # im_gt = im_gt.astype('uint8')
    # sio.imsave(os.path.join(sent_dir, "gt.png"), im_gt)
    im_seg = im / 2
    im_seg[:, :, 0] += predicts.astype('uint8') * 100
    im_seg = im_seg.astype('uint8')
    vis_img = cv2.putText(im_seg, sent, 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)
    sio.imsave(vis_path, vis_img)

    # plt.imshow(im_seg.astype('uint8'))
    # plt.title(sent)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type=str, default='0')
    parser.add_argument('-i', type=int, default=800000)
    parser.add_argument('-s', type=int, default=100000)
    parser.add_argument('-st', type=int, default=700000)  # stop training when get st iters
    parser.add_argument('-m', type=str)  # 'train' 'test'
    parser.add_argument('-imdir', type=str)
    parser.add_argument('-visdir', type=str)
    parser.add_argument('-maskdir', type=str)
    parser.add_argument('-meta', type=str)
    parser.add_argument('-embdir', type=str)
    parser.add_argument('-d', type=str, default='referit')  # 'Gref' 'unc' 'unc+' 'referit'
    parser.add_argument('-t', type=str)  # 'train' 'trainval' 'val' 'test' 'testA' 'testB'
    parser.add_argument('-f', type=str)  # directory to save models
    parser.add_argument('-lr', type=float, default=0.00025)  # start learning rate
    parser.add_argument('-bs', type=int, default=1)  # batch size
    parser.add_argument('-v', default=False, action='store_true')  # visualization
    parser.add_argument('-c', default=False, action='store_true')  # whether or not apply DenseCRF
    parser.add_argument('-emb', default=False, action='store_true')  # whether or not use Pretrained Embeddings
    parser.add_argument('-n', type=str, default='')  # select model
    parser.add_argument('-conv5', default=False, action='store_true')  # finetune conv layers
    global args
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    mu = np.array((104.00698793, 116.66876762, 122.67891434))

    if args.m == 'train':
        train(max_iter=args.i,
              snapshot=args.s,
              dataset=args.d,
              setname=args.t,
              mu=mu,
              lr=args.lr,
              bs=args.bs,
              tfmodel_folder=args.f,
              conv5=args.conv5,
              model_name=args.n,
              stop_iter=args.st,
              pre_emb=args.emb)
    elif args.m == 'test':
        test(iter=args.i,
             dataset=args.d,
             visualize=args.v,
             setname=args.t,
             dcrf=args.c,
             mu=mu,
             tfmodel_folder=args.f,
             model_name=args.n,
             pre_emb=args.emb)
