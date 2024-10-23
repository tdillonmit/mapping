import rospy
from std_msgs.msg import Int32
import numpy as np
import cv2

class GatingVisualizer:
    def __init__(self):
        rospy.init_node('gating_signal_visualizer')

        # Parameters for visualization
        self.signal_length = 100  # Width of the window in pixels
        self.signal_height = 100  # Height of the window in pixels
        self.signal_image = np.zeros((self.signal_height, self.signal_length, 3), dtype=np.uint8)

        # Variable to hold the latest signal value
        self.latest_signal_value = 0

        # Subscribe to the binary signal topic
        self.ecg_image_sub = rospy.Subscriber('/ecg', Int32, self.gating_callback)

        # Set a timer to update the visualization at a fixed rate (e.g., 30 Hz)
        update_rate = 30
        self.timer = rospy.Timer(rospy.Duration(1.0 / 30), self.update_visualization)

        print("Initialized")

    def gating_callback(self, msg):
        # Store the latest value from the message
        self.latest_signal_value = msg.data

    def update_visualization(self, event):
        # Draw a white line for 1 and a black line for 0 based on the latest signal value
        color = (255, 255, 255) if self.latest_signal_value == 1 else (0, 0, 0)

        # Shift the image left to make space for the new signal
        self.signal_image[:, :-1] = self.signal_image[:, 1:]

        # Update the last column with the new signal value
        self.signal_image[:, -1] = color

        resized_image = cv2.resize(self.signal_image, (self.signal_length * 10, self.signal_height * 3))

        # Display the image
        cv2.imshow("ECG Signal", resized_image)
        cv2.waitKey(1)

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

    
