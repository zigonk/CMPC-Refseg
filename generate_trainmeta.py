import json
import os
import skimage
import skimage.io
import numpy as np
from tqdm import tqdm 

meta_path = '/mnt/MyPassport/Youtube-VOS/meta_expressions/meta_expressions/train/meta_expressions.json'
mask_dir = '/mnt/MyPassport/Youtube-VOS/train/train/Annotations'

object_color = {
    '1': [236, 95, 103],
    '2': [249, 145, 87],
    '3': [250, 200, 99],
    '4': [153, 199, 148],
    '5': [98, 179, 178],
    '6': [102, 153, 204]
}

def check_valid_mask(mask, obj_id):
    if (len(mask.shape) == 2):
        return False
    mask = mask[:,:,:3]
    mask_color = object_color[obj_id]
    return np.any(mask == mask_color)


query_dict = json.load(open(meta_path))
videos = query_dict['videos']
samples = []
train_meta = []
for vid in tqdm(videos):
    video = videos[vid]
    expressions = video['expressions']
    frames = video['frames']
    for eid in expressions:
        exp = expressions[eid]['exp']
        obj_id = expressions[eid]['obj_id']
        for fid in frames:
            mask_path = os.path.join(mask_dir, vid, fid + '.png')
            mask = skimage.io.imread(mask_path)
            if (not check_valid_mask(mask, obj_id)):
                continue
            im_name = os.path.join(vid, fid + '.jpg')
            mask_name = os.path.join(vid, fid + '.png')
            train_meta.append([im_name, mask_name, exp, obj_id])

with open('train_meta.json', 'w') as f:
    json.dump(train_meta, f)