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


        # load the logo with alpha channel (PNG)

        logo_rgba = cv2.imread('/home/tdillon/mapping/src/aortascope/aortascope_logo.png', cv2.IMREAD_UNCHANGED)
        h0, w0 = logo_rgba.shape[:2]

        # choose max logo width as 10% of your window width
        max_logo_w = int(self.cv2_window_width * 0.1)
        scale = max_logo_w / w0

        # compute new dimensions, preserving aspect ratio
        new_w = int(w0 * scale)
        new_h = int(h0 * scale)

        # resize
        logo_rgba = cv2.resize(logo_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # split channels as before
        self.logo_bgr   = logo_rgba[:, :, :3]
        alpha_channel   = logo_rgba[:, :, 3] / 255.0
        self.alpha_mask = cv2.merge([alpha_channel]*3)
        self.logo_h, self.logo_w = new_h, new_w



    def rgb_image_callback(self, msg):
        # # Extract image data
        # width = msg.width
        # height = msg.height

        # try:
            
        
        #     image_data  = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3)) 
           
            
           
        # except:
        #     print("failed to visualize incoming ros message!")
        # # except:
        # #     # if rgb
          
        # #     image_data  = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
        # #     # image_data  = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, -1)) 

        # # Display the binary image
        # # cv2.imshow('Image', image_data)

       
        # cv2.imshow("IVUS Image", cv2.resize(image_data, (self.cv2_window_height, self.cv2_window_width)))
        # cv2.waitKey(10)
        # cv2.moveWindow("IVUS Image", self.cv2_window_x, self.cv2_window_y)
        

        # unpack ROS image
        h, w = msg.height, msg.width
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))

        # resize to your display size
        disp = cv2.resize(frame, (self.cv2_window_width, self.cv2_window_height))

        # 5) Simply overwrite the top‐left ROI with your logo
        disp[0:self.logo_h, 0:self.logo_w] = self.logo_bgr

        # show
        cv2.imshow("IVUS Image", disp)
        cv2.moveWindow("IVUS Image", self.cv2_window_x, self.cv2_window_y)
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