"""
This is basically a rewrite of jiahui/test.py which allows to do the inpainting
without all the command line stuff.
"""
import os

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from .inpaint_model import InpaintCAModel


def inpaint(
    image, mask, path_to_config='../jiahui/inpaint.yml', 
    checkpoint_dir='../model_logs/release_places2_256'
):
    """
    Takes the image+mask and returns the inpainted image.
    """
    assert image.shape == mask.shape
    
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    path_to_config = os.path.abspath(path_to_config)
    try:
        FLAGS = ng.Config(path_to_config)
    except AssertionError:
        print('Config file not found at: %s' % path_to_config)
        raise

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    model = InpaintCAModel()
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            
            try:
                var_value = tf.contrib.framework.load_variable(
                    checkpoint_dir, from_name
                )
            except Exception as e:
                print('Pretrained model not found at path %s' % checkpoint_dir)
                raise
                
            assign_ops.append(tf.assign(var, var_value))
            
        sess.run(assign_ops)
        result = sess.run(output)
        
    return result[0][:, :, ::-1]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
if __name__ == '__main__':
    pass









