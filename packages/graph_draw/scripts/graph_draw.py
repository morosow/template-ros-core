#!/usr/bin/env python3

import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from duckietown.dtros import DTROS


class Instruments:

    @staticmethod
    def _assert_image_shapes_equal(org_img: np.ndarray, pred_img: np.ndarray, metric: str):
        msg = (f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
               f"{str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}")

        assert org_img.shape == pred_img.shape, msg

    @staticmethod
    def rmse(org_img: np.ndarray, pred_img: np.ndarray, max_p=255) -> float:
        """
        Root Mean Squared Error
        Calculated individually for all bands, then averaged
        """

        Instruments._assert_image_shapes_equal(org_img, pred_img, "RMSE")

        org_img = org_img.astype(np.float32)

        rmse_bands = []

        dif = np.subtract(org_img, pred_img)
        m = np.mean(np.square(dif / max_p))
        s = np.sqrt(m)

        rmse_bands.append(s)
        return np.mean(rmse_bands)

    @staticmethod
    def rmse_sim(a, b):
        a = cv2.resize(a, (430, 960))
        b = cv2.resize(b, (430, 960))
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)

        return Instruments.rmse(a, b)


class Graph:
    nodes = dict()
    last = None
    first = None
    SIMILARITY_THRESHOLD = 0.2

    def __init__(self, name='default'):
        self.name = name

    @classmethod
    def neighbour_setter(cls, node, value):
        if value not in [node.neighbour_1, node.neighbour_2, node.neighbour_3, node.neighbour_4]:
            if node.neighbour_1 is None:
                node.neighbour_1 = value

            elif node.neighbour_2 is None:
                node.neighbour_2 = value

            elif node.neighbour_3 is None:
                node.neighbour_3 = value

            else:
                node.neighbour_4 = value

    @classmethod
    def add_node(cls, node):
        cls.nodes[node.name] = node

        if cls.last is None:
            cls.first = node
            cls.last = node
        else:
            if cls.first is None:
                cls.first = node
                cls.neighbour_setter(cls.last, node.name)
                cls.neighbour_setter(node, cls.last.name)

            elif cls.get_similarity(node.image, cls.first.image) < cls.SIMILARITY_THRESHOLD and\
            cls.first.name != cls.last.name:
                cls.neighbour_setter(cls.last, cls.first.name)
                cls.neighbour_setter(cls.first, cls.last.name)
                cls.last = cls.first
                cls.first = None
                del node

            else:
                cls.neighbour_setter(cls.last, node.name)
                cls.neighbour_setter(node, cls.last.name)
                cls.last = node

    @staticmethod
    def get_similarity(img_1, img_2):
        if str(img_1) == str(img_2):
            return 1
        return Instruments.rmse_sim(img_1, img_2)


default_graph = Graph()


class Node:
    def __init__(self, image, graph=default_graph):
        self.name = str(uuid.uuid4())
        self.neighbour_1 = None
        self.neighbour_2 = None
        self.neighbour_3 = None
        self.neighbour_4 = None

        if len(image.shape) == 3:
            self.image = image
        else:
            print('Incorrect image type!')
            raise AttributeError

        graph.add_node(self)


class GraphDraw(DTROS):

    def __init__(self, node_name):
        self.sub_mode = rospy.Subscriber('/image/compressed/', CompressedImage, image_cb)
        self.bridge = CvBridge()

    def image_cb(self, image_msg):
        # Decode from compressed image with OpenCV
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr(f"Could not decode image: {e}")

        Node(image, graph=default_graph)


if __name__ == "__main__":
    rospy.init_node('GraphDrawNode')
    graphDraw = GraphDraw('GraphDrawNode')
    rospy.spin()
