import numpy as np
import os
import json
import cv2

cfg = {
    "meta": "/mnt/MyPassport/Youtube-VOS/meta_expressions/meta_expressions/valid/meta_expressions.json",
    "visdir": "./Annotations_1channel",
}



black_img = np.zeros((720, 1280)).astype(np.uint8)

cnt = 0
meta_expression = {}
with open(cfg['meta']) as meta_file:
    meta_expression = json.load(meta_file)
videos = meta_expression['videos']
for vid_ind, vid in enumerate(videos.keys()):  
    print("Running on video {}/{}".format(vid_ind + 1, len(videos.keys())))
    expressions = [videos[vid]['expressions'][expression_id]['exp'] for expression_id in videos[vid]['expressions'].keys()]
    # instance_ids = [expression['obj_id'] for expression_id in videos[vid]['expressions']]
    frame_ids = videos[vid]['frames']
    for index, exp in enumerate(expressions):
        vis_dir = os.path.join(cfg['visdir'], str('{}/{}/'.format(vid, index)))
        # mask_dir = os.path.join(cfg.maskdir, str('{}/{}/'.format(vid, index)))
        avg_time = 0
        total_frame = 0
        for fid in frame_ids:
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            vis_path = os.path.join(vis_dir, str('{}.png'.format(fid)))
            cnt += 1
            cv2.imwrite(vis_path, black_img)

print(cnt)
