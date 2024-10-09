import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2

class GatingVisualizer:
    def __init__(self):
        rospy.init_node('gating_signal_visualizer')

        # Subscribe to the binary image topic
        self.ecg_image_sub = rospy.Subscriber('/ecg_signal', Image, self.gating_callback)

    def gating_callback(self, msg):
        # # Extract image data
        # width = msg.width
        # height = msg.height

        # try:
        #     # if binary
        #     image_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((-1, width))   

        # except:
        #     # if rgb
        #     image_data  = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
        #     # image_data  = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, -1)) 

        # # Display the binary image
        # cv2.imshow('Image', image_data)
        cv2.waitKey(10)
        pass


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    visualizer = GatingVisualizer()
    try:
        visualizer.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()