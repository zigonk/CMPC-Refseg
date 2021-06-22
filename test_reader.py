from util import data_reader_refvos

im_dir = '/mnt/MyPassport/Youtube-VOS/train/train/JPEGImages'
mask_dir = '/mnt/MyPassport/Youtube-VOS/train/train/Annotations'
metadata = './train_meta.json'

reader = data_reader_refvos.DataReader(im_dir, mask_dir, metadata)
print(reader.read_batch())