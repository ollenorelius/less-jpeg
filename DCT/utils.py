import numpy as np
import params as p
from PIL import Image
from scipy.fftpack import dct, idct

def stitch_image(batch, w, h):
    np2_w =  int(2** np.ceil(np.log2(h)))
    np2_h =  int(2** np.ceil(np.log2(w)))

    ims = p.IMAGE_SIZE
    w_slices = int(np.ceil(np2_w / ims))
    h_slices = int(np.ceil(np2_h / ims))
    pred_pic = np.zeros([np2_w,np2_h,p.CHANNELS],dtype='uint8')

    print('w_s = %s, h_s = %s, batch shape = %s'%(w_slices, h_slices, batch.shape))

    for y in range(h_slices):
        for x in range(w_slices):
            x1 = int(x*ims)
            x2 = int((x+1)*ims)
            y1 = int(y*ims)
            y2 = int((y+1)*ims)

            pred_pic[y1:y2, x1:x2, :] = batch[y*h_slices + x]
    return Image.fromarray(pred_pic[0:w, 0:h, :])

def batch_DCT(batch):
    #batch = batch.reshape([-1, 8, 8, 8, 8, 3])
    #trans = dct(dct(batch, axis=4), axis=2)
    #trans = trans.reshape([-1, p.IMAGE_SIZE, p.IMAGE_SIZE, 3])
    #batch = batch.reshape([-1, p.IMAGE_SIZE, p.IMAGE_SIZE, 3])
    batch2 = np.zeros(batch.shape)
    for x in range(8):
        for y in range(8):
            xs = x*8
            xe = (x+1)*8
            ys = y*8
            ye = (y+1)*8
            batch2[:,xs:xe,ys:ye,:] = dct(dct(batch[:,xs:xe,ys:ye,:], axis=1, n=8), axis=2, n=8)
    return batch2

def batch_iDCT(batch):
    #batch = batch.reshape([-1, 8, 8, 8, 8, 3])
    #invtrans = idct(idct(batch, axis=4), axis=2)/(2*8)**2
    #invtrans = invtrans.reshape([-1, p.IMAGE_SIZE, p.IMAGE_SIZE, 3])
    #return invtrans
    batch2 = np.zeros(batch.shape)
    for x in range(8):
        for y in range(8):
            xs = x*8
            xe = (x+1)*8
            ys = y*8
            ye = (y+1)*8
            batch2[:,xs:xe,ys:ye,:] = idct(idct(batch[:,xs:xe,ys:ye,:], axis=1), axis=2)
    return batch2/(2*8)**2
