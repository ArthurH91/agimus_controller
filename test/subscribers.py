#!/usr/bin/env python

import unittest
import rostest
import roslib
import rospy


PKG = "agimus_controller"
roslib.load_manifest(PKG)  # This line is not needed with Catkin.


class TestAgimusControllerSubscribers(unittest.TestCase):
    def test_com_position_subscriber(self):
        rospy.Publisher


if __name__ == "__main__":
    rostest.rosrun(
        PKG,
        "test_subscribers",
        TestAgimusControllerSubscribers,
        sysargs=None,
    )
