from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
import time
import uuid

import align.detect_face
import numpy as np
import src.dao.facedao as facedao
import tensorflow as tf
from lib import facenet
from lib import utils
from scipy import misc
from src.domain.face import *


def main(args):
    # init logging
    mtcnn_logtime = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
    utils.logging_config('../logs/align_mtcnn' + mtcnn_logtime + '.log')

    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(
            config=tf.ConfigProto(device_count={'GPU': 1}, gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # init mtcnn
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            # init facenet
            facenet.load_model(args.model_dir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    minsize = 30  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    total_aligned, successfully_aligned, fail_aligned, face_num_greaterone = 0, 0, 0, 0

    logging.info('开始提取人脸......')
    for root, child_directory, filename_list in os.walk(args.input_dir):
        for filename in filename_list:
            localtime = time.strftime('%Y/%m/%d', time.localtime(time.time()))
            parentfilepath = args.output_dir + '/' + localtime
            utils.mkdir(parentfilepath)

            total_aligned += 1
            input_filepath = os.path.join(root, filename)

            split = input_filepath.split('/')[-4:]
            db_inputname = os.path.join(split[0], split[1], split[2], split[3])

            o_filename = str(uuid.uuid1()) + '.jpg'
            output_filepath = os.path.join(parentfilepath, o_filename)
            try:
                img = misc.imread(input_filepath, mode='RGB')
            except(IOError, ValueError, IndexError) as e:
                errorMessage = '{},{}'.format(input_filepath, e)
                logging.info(errorMessage)
                continue
            else:
                if img.ndim < 2:
                    logging.error('图片维度<2,对齐失败："%s"' % input_filepath)
                    fail_aligned += 1
                    continue
                if img.ndim == 2:
                    facenet.to_rgb(img)
                img = img[:, :, 0:3]
                bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces == 1:
                    try:
                        det = bounding_boxes[:, 0:4]
                        det_arr = []
                        img_size = np.asarray(img.shape)[0:2]
                        det_arr.append(np.squeeze(det))
                        for i, det in enumerate(det_arr):
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                            bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                            bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                            bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])

                            # bb[0] = np.maximum(det[0], 0)
                            # bb[1] = np.maximum(det[1], 0)
                            # bb[2] = np.minimum(det[2], img_size[1])
                            # bb[3] = np.minimum(det[3], img_size[0])

                            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                            scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')

                            crop = scaled.reshape(-1, args.image_size, args.image_size, 3)
                            prewhitened = facenet.prewhiten(crop)
                            feed_dict = {images_placeholder: prewhitened, phase_train_placeholder: False}
                            emb = sess.run(embeddings, feed_dict=feed_dict)

                            misc.imsave(output_filepath, scaled)
                            successfully_aligned += 1

                            face = Face()
                            face.ori_pic = db_inputname
                            face.detect_pic = os.path.join(localtime, o_filename)
                            face.feature_vector = emb.tolist()
                            facedao.insert_face(face)

                            logging.info('提取成功："%s"' % input_filepath)
                    except Exception as e:
                        logging.error(e)
                        continue
                elif nrof_faces > 1:
                    logging.error('检测出人脸数量大于1："%s"' % input_filepath)
                    face_num_greaterone += 1
                else:
                    logging.error('未提取出人脸："%s"' % input_filepath)
                    fail_aligned += 1
    logging.info('图片总数：%d:,提取成功：%d,人脸数量过多不提取：%d,提取失败：%d' % (
        total_aligned, successfully_aligned, face_num_greaterone, fail_aligned))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--model_dir', type=str, help='Directory with model.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=32)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
