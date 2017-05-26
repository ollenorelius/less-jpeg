from PIL import Image
import os
import numpy as np
import params as p
import re


try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO


class InputProducer():
    filenames = []
    filenames_ordered = []
    epoch = 0
    current_pic = 0
    pic_slice_index = 0
    pic_slices = []
    n_pictures = 0
    shuffle = p.SHUFFLE
    folder = ''
    def __init__(self, folder, ending):

        self.folder = folder
        self.filenames = self.filter_filenames(os.listdir(folder), ending)
        self.filenames_ordered = list(self.filenames)
        self.n_pictures = len(self.filenames)
        if self.shuffle:
            np.random.shuffle(self.filenames)
        self.pic_slices, _, _ = self.new_slice_list(self.current_pic, trim = False)

        assert self.n_pictures > 0,\
                "No pictures with ending %s found in %s!"%(ending,folder)

    def get_batch(self, batch_size):
        batch = np.ones([batch_size, p.IMAGE_SIZE, p.IMAGE_SIZE, p.CHANNELS], dtype='uint8')*255
        for i_pic in range(batch_size):

            batch[i_pic,:,:,:] = self.pic_slices.pop()
            if len(self.pic_slices) == 0:
                self.current_pic = self.current_pic + 1
                if self.current_pic >= self.n_pictures:
                    self.current_pic = 0
                    self.epoch = self.epoch + 1
                    if self.shuffle:
                        np.random.shuffle(self.filenames)
                self.pic_slices, _, _ = self.new_slice_list(self.current_pic,trim=True)

        return batch


    def get_picture(self, number):
        pic_slices, w, h = self.new_slice_list(number, trim=False, shuffle=False)
        slice_count = len(pic_slices)
        batch = np.ones([slice_count, p.IMAGE_SIZE, p.IMAGE_SIZE, p.CHANNELS], dtype='uint8')*255
        for i in range(slice_count):
            batch[i] = pic_slices[i]

        return batch, w, h


    def filter_filenames(self, filename_list, ending):
        filtered_list = []
        for name in filename_list:
            if re.search('\.'+ending+'\Z', name) != None:
                filtered_list.append(name)
        return filtered_list

    def compress_batch(self, input_batch, level):
        batch_size = input_batch.shape[0]
        out_batch = np.ones(input_batch.shape, dtype='uint8')*255
        for i_pic in range(batch_size):
            out = BytesIO()
            im = Image.fromarray(input_batch[i_pic,:,:,:], 'RGB')
            im.save(out, format=p.FORMAT, quality=level)
            out_batch[i_pic,:,:,0:3] = Image.open(out)

        return out_batch

    def save_batch(self, batch, folder):
        batch_size = batch.shape[0]
        for i_pic in range(batch_size):
            out = open(folder + '/%s.png'%i_pic, 'wb')
            im = Image.fromarray(batch[i_pic,:,:,:], 'RGB')
            im.save(out, format='png', quality=100)
        return 1

    def new_slice_list(self, pic_number, trim, shuffle=p.SHUFFLE):
        pic_filename = self.filenames_ordered[pic_number]
        img = Image.open(self.folder + "/" + pic_filename)

        h, w = img.size

        np2_w =  2** np.ceil(np.log2(w))
        np2_h =  2** np.ceil(np.log2(h))
        img = np.pad(np.asarray(img)[:,:,0:3],
            ((0, int(np2_w - w)), (0, int(np2_h - h)), (0,0)),
            mode='constant', constant_values=0)
        ims = p.IMAGE_SIZE
        w_slices = int(np.ceil(np2_w / ims))
        h_slices = int(np.ceil(np2_h / ims))

        pic_slices = []
        for x in range(w_slices):
            for y in range(h_slices):
                x1 = int(x*ims)
                x2 = int((x+1)*ims)

                y1 = int(y*ims)
                y2 = int((y+1)*ims)
                #print(np.var(img[x1:x2, y1:y2, :]))
                if np.var(img[x1:x2, y1:y2, :]) == 0 and trim:
                    continue

                pic_slices.append(img[x1:x2, y1:y2, :])
        if shuffle:
            np.random.shuffle(pic_slices)
        return pic_slices, w, h
