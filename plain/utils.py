import numpy as np
import params as p
from PIL import Image
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
