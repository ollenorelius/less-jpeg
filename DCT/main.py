import tensorflow as tf
import network as net
from input_producer import InputProducer
import loss
import params as p
import numpy as np
import utils as u
from PIL import Image

data_folder = 'data'
batch_size = p.BATCH_SIZE
input_prod = InputProducer(data_folder, 'png')

inp = tf.placeholder(tf.float32, shape=[None, p.IMAGE_SIZE, p.IMAGE_SIZE, p.CHANNELS])
inp_compressed = tf.placeholder(tf.float32, shape=[None, p.IMAGE_SIZE, p.IMAGE_SIZE, p.CHANNELS])

#feature_representation = net.create_forward_net(inp_compressed)
#return_image = net.create_backward_net(feature_representation, inp_compressed)

#feature_representation = net.create_deep_forward_net(inp_compressed)
return_image = net.create_flat_net(inp_compressed)

loss_value = loss.create_loss(inp, return_image)
tf.summary.scalar('Loss', loss_value)
global_step = tf.Variable(0, dtype=tf.int32)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss_value, global_step=global_step)


merged = tf.summary.merge_all()

import os
import sys
import time

if not os.path.exists('./networks/'):
    os.makedirs('./networks/')

with tf.Session() as sess:
    net_name = 'less_jpeg-full-deep-res'
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("output/"+net_name, sess.graph)


    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)

    if '-new' not in sys.argv:
        print('loading network.. ', end='')
        try:
            saver.restore(sess, './networks/%s.cpt'%net_name)
            print('Done.')
        except:
            print('Couldn\'t load net, creating new! E:(%s)'%sys.exc_info()[0])
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
    else:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    for i in range(1000000):
        comp_qual = 30
        batch = input_prod.get_batch(batch_size)
        batch_DCT = u.batch_DCT(batch)
        batch_comp = u.batch_DCT(input_prod.compress_batch(batch, comp_qual))
        #input_prod.save_batch(batch_comp, 'data/compressed')

        train_step.run(feed_dict={inp:batch_DCT, inp_compressed:batch_comp})
        if i%30 == 0:
            print('step %s, loss: %e'%(i, loss_value.eval(feed_dict={inp:batch_DCT, inp_compressed:batch_comp})))
            writer.add_summary(merged.eval(feed_dict={inp:batch_DCT, inp_compressed:batch_comp}), global_step=sess.run(global_step))
        if i%10000 == 0:
            batch_pic, w, h = input_prod.get_picture(0)
            batch_comp_pic = input_prod.compress_batch(batch_pic, comp_qual)
            batch_comp_pic = u.batch_DCT(batch_comp_pic)
            sub_batch_list = []

            for j in range(batch_comp_pic.shape[0]//batch_size):
                sub_batch_list.append(batch_comp_pic[j*batch_size:(j+1)*batch_size])
            if (batch_pic.shape[0]//batch_size)*batch_size != batch_pic.shape:
                sub_batch_list.append(batch_comp_pic[(batch_pic.shape[0]//batch_size)*batch_size:])

            pred = np.ones(batch_comp_pic.shape)

            for j, sub_batch in enumerate(sub_batch_list):
                pred[j*batch_size:(j+1)*batch_size] = np.squeeze(return_image.eval(feed_dict={inp_compressed:sub_batch}))
            pred = np.clip(u.batch_iDCT(pred), 0,255).astype('uint8')

            pred_pic = u.stitch_image(pred, w, h)
            pred_pic.save('predictions/%s.png'%i)
            saver.save(sess, './networks/%s.cpt'%net_name)
