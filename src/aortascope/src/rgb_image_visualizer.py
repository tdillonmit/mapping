import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
import subprocess

# aortascope gui functions
def get_window_id(window_name):
    # Run wmctrl command to get the list of windows
    result = subprocess.run(['wmctrl', '-l'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        # Decode the result to get the list of windows
        windows = result.stdout.decode().splitlines()
        
        # Loop through each window and check if the window name matches
        for window in windows:
            if window_name in window:
                # The window ID is the first part of the line
                window_id = window.split()[0]
                return window_id
    return None

def bring_window_to_front(window_id):
    if window_id:
        subprocess.run(['wmctrl', '-i', '-r', window_id, '-b', 'add,above'])

class RGBImageVisualizer:
    def __init__(self):
        rospy.init_node('rgb_image_visualizer')

        # Subscribe to the binary image topic
        self.rgb_image_sub = rospy.Subscriber('/rgb_image', Image, self.rgb_image_callback)

        # change to 1 for single display
        display_scale_factor = 1


        self.cv2_window_height = int(224 * 2.6 * 0.95) 
        self.cv2_window_width = int(self.cv2_window_height * 0.95) 
        self.cv2_window_x = int(483)
        self.cv2_window_y = int(0 )


        # placeholder_image = np.zeros((self.cv2_window_height, self.cv2_window_width, 3), dtype=np.uint8)
        # cv2.imshow("IVUS Image", placeholder_image)
        # cv2.moveWindow("IVUS Image", self.cv2_window_x, self.cv2_window_y)
        # cv2.waitKey(2)

        # window_name = "IVUS Image"  # Replace with your window's title
        # window_id = get_window_id(window_name)
        # bring_window_to_front(window_id)



    def rgb_image_callback(self, msg):
        # Extract image data
        width = msg.width
        height = msg.height

        try:
            
        
            image_data  = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3)) 
           
            
           
        except:
            print("failed to visualize incoming ros message!")
        # except:
        #     # if rgb
          
        #     image_data  = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
        #     # image_data  = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, -1)) 

        # Display the binary image
        # cv2.imshow('Image', image_data)

       
        # cv2.imshow("ivus", image_data)
        # cv2.namedWindow("IVUS Image", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("IVUS Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("IVUS Image", cv2.resize(image_data, (self.cv2_window_height, self.cv2_window_width)))
        cv2.waitKey(10)
        cv2.moveWindow("IVUS Image", self.cv2_window_x, self.cv2_window_y)
        
        print("ivusimage complete")


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