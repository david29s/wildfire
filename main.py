import os
import cv2

import numpy as np
import tensorflow as tf


def classify_image():
    classifications = []

    for currentLine in tf.gfile.GFile(labels):
        classification = currentLine.rstrip()
        classifications.append(classification)

    with tf.gfile.FastGFile(graph, 'rb') as graph_file:
        defined_graph = tf.GraphDef()
        defined_graph.ParseFromString(graph_file.read())

        tf.import_graph_def(defined_graph, name='')

    with tf.Session() as session:
        for file in os.listdir(images_directory):
            image = cv2.imread(images_directory + file)
            tf_image = np.array(image)[:, :, 0:3]

            final_layer = session.graph.get_tensor_by_name('final_result:0')

            predictions = session.run(final_layer, {'DecodeJpeg:0': tf_image})
            prediction = predictions[0].argsort()[-len(predictions[0]):][::-1][0]

            confidence = round(predictions[0][prediction] * 100.0, 2)
            label = classifications[prediction] + " ({}%)".format(confidence)

            cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(file, image)
            cv2.waitKey(0)


graph = "./model/graph.pb"
labels = "./model/labels.txt"

images_directory = "./images/"

classify_image()
