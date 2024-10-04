import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped
import numpy as np
import copy
from std_msgs.msg import Header  
import math
import yaml


class PointCloudUpdater:



    def __init__(self):
        

        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)

        print("frame grabber subscriber:", self.image_sub)

        self.binary_image_pub = rospy.Publisher('/binary_image', Image, queue_size=10)

        self.image_width = rospy.get_param('/usb_cam/image_width', default=1280)
        self.image_height = rospy.get_param('/usb_cam/image_height', default=1024)

        self.write_folder = rospy.get_param('~image_path')
        self.i=1


    def image_callback(self, msg):
        

        # make sure this importing is the same as that of the mapping function
        # Assuming RGB format
        rgb_image_data = np.frombuffer(msg.data, dtype=np.uint8)

        # Reshape the RGB data
        rgb_image = rgb_image_data.reshape((self.image_height, self.image_width, 3))
        
        full_str=self.write_folder + '/rgb_image_'+ str(self.i) + '.npy'
        
        # save rgb_image as is
        with open(full_str, 'wb') as f:
            np.save(f,rgb_image)

        print("Saved", self.i, "images")
        self.i=self.i+1
        

            

if __name__ == '__main__':
    try:
        rospy.init_node('image_acquisition')
        pc_updater = PointCloudUpdater()
        rospy.spin()
        
    except rospy.ROSInterruptException:
        self.vis.destroy_window()
        pass
