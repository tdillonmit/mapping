import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2

class RGBImageVisualizer:
    def __init__(self):
        rospy.init_node('rgb_image_visualizer')

        # Subscribe to the binary image topic
        self.rgb_image_sub = rospy.Subscriber('/rgb_image', Image, self.rgb_image_callback)

    def rgb_image_callback(self, msg):
        # Extract image data
        width = msg.width
        height = msg.height

        try:
            
            # if rgb
            image_data  = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3)) 
        except:
            print("failed to visualize incoming ros message!")
        # except:
        #     # if rgb
          
        #     image_data  = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
        #     # image_data  = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, -1)) 

        # Display the binary image
        cv2.imshow('Image', image_data)
        cv2.waitKey(10)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    visualizer = RGBImageVisualizer()
    try:
        visualizer.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()