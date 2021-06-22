from util import data_reader_refvos

im_dir = '/mnt/MyPassport/Youtube-VOS/train/train/JPEGImages'
mask_dir = '/mnt/MyPassport/Youtube-VOS/train/train/Annotations'

reader = data_reader_refvos.DataReader(im_dir, mask_dir, metadata)