from PIL import Image
import os
import numpy as np
import params as p
import re
import random

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO


class InputProducer():
    filenames = []
    epoch = 0
    current_pic = 0
    n_pictures = 0
    shuffle = False

    def __init__(self, folder, ending):
        self.filenames = filter_filenames(os.listdir(folder).sort(), ending)
        self.n_pictures = len(filenames)
        if shuffle:
            random.shuffle(filenames)

        assert self.n_pictures > 0,\
                "No pictures with ending %s found in %s!"%(ending,folder)

    def get_batch(batch_size):
        batch = np.array([batch_size, p.IMAGE_SIZE, p.IMAGE_SIZE, p.CHANNELS])
        for i_pic in range(batch_size):
            pic_filename = filenames[current_pic]
            batch[iPic, :,:,:] = Image.open(pic_filename)\
                        .resize((p.IMAGE_SIZE, p.IMAGE_SIZE),Image.ANTIALIAS)

            self.current_pic = self.current_pic + 1
            if self.current_pic >= self.n_pictures:
                self.current_pic = 0
                self.epochs = self.epochs + 1
                if shuffle:
                    random.shuffle(filenames)

        return batch


    def filter_filenames(filename_list, ending):
        filtered_list = []
        for name in filename_list:
            if re.search('\.'+ending+'\Z', name) != None:
                filtered_list.append(name)
        return filtered_list

    def compress_batch(input_batch, level):
        batch_size = input_batch.shape()[0]
        out_batch = np.array(input_batch.shape())

        for i_pic in range(len(batch_size)):
            out = BytesIO()
            im = Image.fromarray(input_batch[i_pic,:,:,:], 'RGB')
            im.save(out, format=p.FORMAT, quality=level)
            out_batch[i_pic,:,:,:] = Image.open(out)

        return out_batch
