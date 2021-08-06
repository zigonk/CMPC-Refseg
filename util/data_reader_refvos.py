from __future__ import print_function
from util import processing_tools

import numpy as np
import os
import threading
import skimage
import skimage.io
import queue as queue
import cv2
from util import im_processing, text_processing, io
import json

object_color = {
    '1': [236, 95, 103],
    '2': [249, 145, 87],
    '3': [250, 200, 99],
    '4': [153, 199, 148],
    '5': [98, 179, 178],
    '6': [102, 153, 204]
}

vocab_file = './data/vocabulary_refvos.txt'
anchor_file = './data/anchors.txt'
vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

def preprocess_data(im, mask, sent, obj_id, T, input_H, input_W):
    anchors = io.read_anchors(anchor_file)
    mask_color = object_color[obj_id]
    mask_obj = np.asarray(((mask == mask_color)[:,:,0]))
    # Augmentation brightness 
    im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, input_H, input_W))
    mask = im_processing.resize_and_pad(mask_obj, input_H, input_W)
    bbox = im_processing.bboxes_from_masks(np.asarray(mask))
    bbox[:,2:4] += bbox[:,:2]
    label_bbox, true_bbox = processing_tools.preprocess_true_boxes(bbox, input_H, anchors)
    text, sequence_length = text_processing.preprocess_sentence_lstm(sent, vocab_dict, T)
    return {
        'text_batch': np.asarray(text),
        'im_batch': np.asarray(im),
        'seq_length': sequence_length,
        'mask_batch': (mask > 0),
        'sent_batch': [sent],
        'label_bbox': label_bbox,
        'true_bbox': true_bbox
    }

def run_prefetch(prefetch_queue, im_dir, mask_dir, metadata, num_batch, shuffle, T, input_H, input_W):
    n_batch_prefetch = 0
    fetch_order = np.arange(num_batch)
    while True:
        # Shuffle the batch order for every epoch
        if n_batch_prefetch == 0 and shuffle:
            fetch_order = np.random.permutation(num_batch)

        # Load batch from file
        batch_id = fetch_order[n_batch_prefetch]
        im_name, mask_name, sent, obj_id = metadata[batch_id]
        # Load image
        im_name = os.path.join(im_dir, im_name)
        im = skimage.io.imread(im_name)
        # Load mask
        mask_name = os.path.join(mask_dir, mask_name)
        mask = skimage.io.imread(mask_name)[:,:,:3]
        # Preprocess data
        batch = preprocess_data(im, mask, sent, obj_id, T, input_H, input_W)
        
        # add loaded batch to fetchqing queue
        prefetch_queue.put(batch, block=True)

        # Move to next batch
        n_batch_prefetch = (n_batch_prefetch + 1) % num_batch

class DataReader:
    def __init__(self, im_dir, mask_dir, train_metadata, shuffle=True, prefetch_num=8, T=20, input_H=320, input_W=320):
        self.im_dir = im_dir
        self.mask_dir = mask_dir
        self.metadata = json.load(open(train_metadata))
        self.shuffle = shuffle
        self.prefetch_num = prefetch_num

        self.n_batch = 0
        self.n_epoch = 0
        self.T = T
        self.input_H = input_H
        self.input_W = input_W

        # Search the folder to see the number of num_batch
        self.num_batch = len(self.metadata)

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=prefetch_num)
        self.prefetch_thread = threading.Thread(target=run_prefetch,
            args=(self.prefetch_queue, self.im_dir, self.mask_dir, self.metadata,
                  self.num_batch, self.shuffle,
                  self.T, self.input_H, self.input_W))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def read_batch(self, is_log = True):
        if is_log:
            print('data reader: epoch = %d, batch = %d / %d' % (self.n_epoch, self.n_batch, self.num_batch))

        # Get a batch from the prefetching queue
#         if self.prefetch_queue.empty():
#             print('data reader: waiting for file input (IO is slow)...')
        batch = self.prefetch_queue.get(block=True)
        self.n_batch = (self.n_batch + 1) % self.num_batch
        self.n_epoch += (self.n_batch == 0)
        return batch
