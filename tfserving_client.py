#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: qujun
# datetime:5/12/20 10:43 PM

from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

import grpc
import numpy
import tensorflow as tf


from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np
import time

import time
import requests
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)
    print(type(img.eval()))


def grp_client(_argv):
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')
    channel = grpc.insecure_channel("127.0.0.1:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'yolov3'
    request.model_spec.signature_name = 'serving_default'

    img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)
    request.inputs['input'].CopyFrom(tf.make_tensor_proto(img))
    result_future = stub.Predict.future(request, 10.25)
    print("finished!!!")
    output = result_future.result().outputs
    boxes_o = output['yolo_nms']
    boxes_n = tf.make_ndarray(boxes_o)
    boxes = tf.convert_to_tensor(boxes_n, dtype=float)
    scores_n = tf.make_ndarray(output['yolo_nms_1'])
    scores = tf.convert_to_tensor(scores_n)
    classes_n = tf.make_ndarray(output['yolo_nms_2'])
    classes = tf.convert_to_tensor(classes_n)
    nums_n = tf.make_ndarray(output['yolo_nms_3'])
    nums = tf.convert_to_tensor(nums_n)

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(FLAGS.output, img)
    logging.info('output saved to: {}'.format(FLAGS.output))

if __name__ == "__main__":
    try:
        app.run(grp_client)
    except SystemExit:
        pass