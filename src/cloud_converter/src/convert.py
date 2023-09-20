#!/usr/bin/env python

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import JointState

class Converter(object):
    def __init__(self):
        self.pcl_pub = rospy.Publisher('velodyne_points', PointCloud2, queue_size=10)
        self.js_pub = rospy.Publisher('joint_states', JointState, queue_size=1)

        self.pcl_sub = rospy.Subscriber("ouster/points", PointCloud2, self.pcl_callback)
        self.js_sub = rospy.Subscriber("joint_states/old", JointState, self.js_callback)

    def pcl_callback(self, msg):
        time_stamp = rospy.Time.now()
        header_msg = Header(stamp=time_stamp, frame_id='velodyne')
        msg.header = header_msg
        self.pcl_pub.publish(msg)

    def js_callback(self, msg):
        time_stamp = rospy.Time.now()
        header_msg = Header(stamp=time_stamp, frame_id='')
        msg.header = header_msg
        self.js_pub.publish(msg)

def main():
    rospy.init_node('cloud_converter', anonymous=True)

    converter = Converter()
    rospy.spin()

if __name__ == "__main__":
    main()