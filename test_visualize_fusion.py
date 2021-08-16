from __future__ import division

import sys
import os
import argparse
from typing import DefaultDict
import tensorflow as tf
import skimage
from skimage import io as sio
import time
import cv2
import json
from PIL import Image
import matplotlib.pyplot as plt
from get_model import get_segmentation_model
from pydensecrf import densecrf
import math

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
    return np.asarray(skimage.io.imread(img_path))

def load_frame_from_id(vid, frame_id):
    frame_path = os.path.join(args.imdir, str('{}/{}.jpg'.format(vid, frame_id)))
    return load_image(frame_path)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def test(iter, dataset, visualize, setname, dcrf, mu, tfmodel_path, model_name, pre_emb=False):
    data_folder = './' + dataset + '/' + setname + '_batch/'
    data_prefix = dataset + '_' + setname
    if visualize:
        save_dir = './' + dataset + '/visualization/' + str(iter) + '/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    weights = os.path.join(tfmodel_path)
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
    emb_name = 'referit' if dataset == 'referit' else 'refvos'
    vocab_file = './data/vocabulary_refvos.txt'
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
    plt.figure(figsize=[15, 6])
    sorted_video_key = ['b5514f75d8', 'a9f23c9150', '4fe6619a47', '3f4bacb16a', '65e0640a2a', 'e11254d3b9', '1335b16cf9', '226f1e10f7', '0e8a6b63bb', '65350fd60a', '62bf7630b3', '1e20ceafae', 'c74fc37224', '45dc90f558', 'd975e5f4a9', 'cb06f84b6e', 'eea1a45e49', '369919ef49', '822c31928a', '7daa6343e6', '246e38963b', '547416bda1', '466734bc5c', '30fe0ed0ce', 'c280d21988', '94fa9bd3b5', 'bf2d38aefe', '54526e3c66', 'e027ebc228', 'f7255a57d0', 'c9ef04fe59', 'b205d868e6', '621487be65', '6a75316e99', '0062f687f1', 'd69812339e', 'd59c093632', '3dd327ab4e', 'f2a45acf1c', '03fe6115d4', 'a4bce691c6', 'c16d9a4ade', 'b83923fd72', '8273b59141', 'a2948d4116', '8e2e5af6a8', 'e633eec195', '8939473ea7', '13ca7bbcfd', '8dea7458de', 'deed0ab4fc', '853ca85618', '975be70866', 'e10236eb37', '9a38b8e463', 'c42fdedcdd', '0f3f8b2b2f', '1ab5f4bbc5', 'f143fede6f', '33c8dcbe09', '3674b2c70a', '541ccb0844', '623d24ce2b', '5d2020eff8', '4ee0105885', 'abae1ce57d', '17cba76927', 'a1251195e7', '44e5d1a969', '35948a7fca', '9787f452bf', '4f5b3310e3', '696e01387c', '5460cc540a', '9da2156a73', '43115c42b2', '0daaddc9da', '64c6f2ed76', '6031809500', '182dbfd6ba', '1a1dbe153e', 'dce363032d', 'f3678388a7', '7a19a80b19', 'a0fc95d8fc', '01c88b5b60', '0788b4033d', 'dea0160a12', '33e8066265', 'eb263ef128', 'b2256e265c', '749f1abdf9', '335fc10235', '0b0c90e21a', '06a5dfb511', '9f429af409', 'b05faf54f7', 'b772ac822a', '29c06df0f2', '218ac81c2d', '48d2909d9e', '6cb5b08d93', '77df215672', '332dabe378', 'b00ff71889', '60362df585', 'ab9a7583f1', '352ad66724', '47d01d34c8', '13c3cea202', '188cb4e03d', '35d5e5149d', 'f39c805b54', 'd1ac0d8b81', '3be852ed44', '0a598e18a8', 'fb104c286f', 'eeb18f9d47', 'b7b7e52e02', '37b4ec2e1a', 'a00c3fa88e', 'cbea8f6bea', '9f16d17e42', 'd7a38bf258', '7a72130f21', 'f054e28786', '85968ae408', 'd7ff44ea97', 'bf4cc89b18', 'fef7e84268', 'fd8cf868b2', '92fde455eb', '3e03f623bb', '0782a6df7e', 'cd896a9bee', '6cc8bce61a', 'f7d7fb16d0', '8c60938d92', '9fd2d2782b', '1a609fa7ee', '69c0f7494e', '7775043b5e', '61fca8cbf1', 'cc7c3138ff', '72d613f21a', 'cdcfd9f93a', '1a894a8f98', '9c0b55cae5', '0390fabe58', '31d3a7d2ee', '450bd2e238', 'dab44991de', '4f6662e4e0', 'd1dd586cfd', '4037d8305d', '257f7fd5b8', '63883da4f5', 'b58a97176b', '411774e9ff', 'a46012c642', '20a93b4c54', '3b72dc1941', '559a611d86', 'c2bbd6d121', '9ce299a510', '4b783f1fc5', 'b7928ea5c0', 'eb49ce8027', 'bc9ba8917e', '9f21474aca', '68dab8f80c', '0c04834d61', 'b90f8c11db', '06cd94d38d', 'aceb34fcbe', 'dc197289ef', '0723d7d4fe', '39bce09d8d', '8d803e87f7', '39b7491321', 'b3b92781d9', '4307020e0f', '1f390d22ea', '31e0beaf99', '19cde15c4b', 'ba8823f2d2', 'ee9415c553', 'a7462d6aaf', '1e0257109e', '2b904b76c9', '97b38cabcc', '34564d26d8', '8b7b57b94d', 'cc1a82ac2a', '7741a0fbce', 'cd69993923', 'a806e58451', '7f26b553ae', 'ebe7138e58', '0620b43a31', '7836afc0c2', '152fe4902a']
    sorted_video_key = ['853ca85618'] #Example 1
    # sorted_video_key = ['b205d868e6'] #Example 2
    # sorted_video_key = ['d975e5f4a9'] #Example 3
    # sorted_video_key = ['b5514f75d8'] #Example 4
    # visualize_list = ['63883da4f5_1_00120', 'b3b92781d9_0_00120', 'bc9ba8917e_1_00160', 'eb49ce8027_1_00000', '6a75316e99_1_00220']
    # sorted_video_key = ['63883da4f5', 'b3b92781d9', 'bc9ba8917e', 'eb49ce8027', '6a75316e99']
    words_parse_dict = DefaultDict(lambda: np.asarray([0, 0, 0, 0], dtype=np.float32))
    words_count = DefaultDict(lambda: 0)
    for vid_ind, vid in enumerate(sorted_video_key):
        print("Running on video {}/{}".format(vid_ind + 1, len(videos.keys())))
        print("Video id:{}".format(vid))
        expressions = videos[vid]['expressions']
        # instance_ids = [expression['obj_id'] for expression_id in videos[vid]['expressions']]
        frame_ids = videos[vid]['frames']
        for eid in expressions:
            exp = expressions[eid]['exp']
            if (eid != '2'):
                continue
            exp = 'the hand is stretched out toward the tree and lizard behind the white'
            index = int(eid)
            vis_dir = args.visdir
#             mask_dir = os.path.join(args.maskdir, str('{}/{}/'.format(vid, index)))
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
#             if not os.path.exists(mask_dir):
#                 os.makedirs(mask_dir)
            avg_time = 0
            total_frame = 0
#             Process text
            exp = args.exp_test
            text, seq_len = text_processing.preprocess_sentence_lstm(exp, vocab_dict, T)
            for fid in frame_ids:
                frame_id = int(fid)
                if (frame_id % 20 != 0):
                    continue
                # visualize_name = '{}_{}_{}'.format(vid, eid, fid)
                # if (visualize_name not in visualize_list):
                #     continue
                vis_path = os.path.join(vis_dir, str('{}_{}_{}.png'.format(vid,eid,fid)))
                # if (os.path.exists(vis_path)): 
                #     continue
                frame = load_frame_from_id(vid, fid)
                frame = np.asarray(skimage.io.imread(args.im_test))
                if frame is None:
                    continue
                last_time = time.time()
#                 im = frame.copy()
                im = frame
#                 mask = np.array(frame, dtype=np.float32)

                proc_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, H, W))
                proc_im_ = proc_im.astype(np.float32)
                proc_im_ = proc_im_[:, :, ::-1]
                proc_im_ -= mu
                scores_val, up_val, sigm_val, up_c3, up_c4, up_c5, words_parse = sess.run([model.pred, 
                                                                                                model.up, 
                                                                                                model.sigm, 
                                                                                                model.up_c3, 
                                                                                                model.up_c4, 
                                                                                                model.up_c5,
                                                                                                model.words_parse
                                                                                                # model.consitency_score
                                                                                                ],
                                                                                                feed_dict={
                                                                                                    model.words: np.expand_dims(text, axis=0),
                                                                                                    model.im: np.expand_dims(proc_im_, axis=0),
                                                                                                    model.seq_len: np.expand_dims(seq_len, axis=0)
                                                                                                })
                
                # print(exp)
                # print(words_parse)
                # break
                # scores_val = np.squeeze(scores_val)
                # pred_raw = (scores_val >= score_thresh).astype(np.float32)
                exp_split = exp.split(' ')[:20]
                words_parse = np.round(words_parse, 2)
                # print(words_parse)
                up_c3 = im_processing.resize_and_crop(sigmoid(np.squeeze(up_c3)), frame.shape[0], frame.shape[1])
                up_c4 = im_processing.resize_and_crop(sigmoid(np.squeeze(up_c4)), frame.shape[0], frame.shape[1])
                up_c5 = im_processing.resize_and_crop(sigmoid(np.squeeze(up_c5)), frame.shape[0], frame.shape[1])
                sigm_val = im_processing.resize_and_crop(sigmoid(np.squeeze(sigm_val)), frame.shape[0], frame.shape[1])
                up_val = np.squeeze(up_val)
                # if (not math.isnan(consitency_score) and consitency_score < 0.3):
                plt.clf()
                plt.subplot(1, 5, 1)
                plt.text(0, -100, 'Expression: ' + exp, fontsize=16)
                plt.imshow(frame)
                # plt.axis('off')
                plt.subplot(1, 5, 2)
                plt.imshow(up_c3).set_cmap('jet')
                # plt.axis('off')
                plt.subplot(1, 5, 3)
                plt.imshow(up_c4).set_cmap('jet')
                # plt.axis('off')
                plt.subplot(1, 5, 4)
                plt.imshow(up_c5).set_cmap('jet')
                # plt.axis('off')
                plt.subplot(1, 5, 5)
                plt.imshow(sigm_val).set_cmap('jet')
                # plt.axis('off')
                # plt.savefig(vis_path, bbox_inches='tight',pad_inches = 0)
                # plt.savefig(vis_path)
                plt.savefig(args.export_dir + '/' + exp.replace(' ', '_') + '.png')
                break
            for i, word in enumerate(exp_split):
                print(word)
                print(words_parse[0][0][i])
            print('---------------------')
            return
                # words_parse_dict[word] += words_parse[0][0][i]
                # words_count[word] += 1
#                 pred_raw = (up_val >= score_thresh).astype('uint8') * 255
#                 pred_raw = (up_val >= score_thresh).astype(np.float32)
#                 predicts = im_processing.resize_and_crop(pred_raw, mask.shape[0], mask.shape[1])
#                 if dcrf:
#                     # Dense CRF post-processing
#                     sigm_val = np.squeeze(sigm_val) + 1e-7
#                     d = densecrf.DenseCRF2D(W, H, 2)
#                     U = np.expand_dims(-np.log(sigm_val), axis=0)
#                     U_ = np.expand_dims(-np.log(1 - sigm_val), axis=0)
#                     unary = np.concatenate((U_, U), axis=0)
#                     unary = unary.reshape((2, -1))
#                     d.setUnaryEnergy(unary)
#                     d.addPairwiseGaussian(sxy=3, compat=3)
#                     d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=proc_im, compat=10)
#                     Q = d.inference(5)
#                     pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype('uint8') * 255
# #                     pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)
# #                     predicts_dcrf = im_processing.resize_and_crop(pred_raw_dcrf, mask.shape[0], mask.shape[1])
#                 if visualize:
#                     if dcrf:
#                         cv2.imwrite(vis_path, pred_raw_dcrf)
# #                         np.save(mask_path, np.array(pred_raw_dcrf))
# #                         visualize_seg(vis_path, im, exp, predicts_dcrf)
#                     else:
#                         np.save(mask_path, np.array(sigm_val))
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
    # for word in words_parse_dict.keys():
    #     words_parse_dict[word] /= words_count[word]
    #     words_parse_dict[word] = np.round(words_parse_dict[word], 2)
    #     print(word)
    #     print(words_parse_dict[word])
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
    parser.add_argument('-im_test', type=str)
    parser.add_argument('-exp_test', type=str)
    parser.add_argument('-export_dir', type=str)
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
             tfmodel_path=args.f,
             model_name=args.n,
             pre_emb=args.emb)
