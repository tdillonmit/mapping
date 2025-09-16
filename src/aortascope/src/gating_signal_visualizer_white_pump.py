import rospy
from std_msgs.msg import Int32
import numpy as np
import cv2

class GatingVisualizer:
    def __init__(self):
        rospy.init_node('gating_signal_visualizer_white_pump')

        # Parameters for visualization
        self.signal_length = 100  # Width of the window in pixels
        self.signal_height = 100  # Height of the window in pixels
        self.signal_image = np.zeros((self.signal_height, self.signal_length, 3), dtype=np.uint8)

        # Separate image for graph
        y_min = 0
        y_max = 1500
        self.graph_image = np.zeros((self.signal_height, self.signal_length, 3), dtype=np.uint8)
        self.y_min = y_min
        self.y_max = y_max

        # Variable to hold the latest signal value
        self.latest_signal_value = 0

        # Subscribe to the binary signal topic
        self.ecg_image_sub = rospy.Subscriber('/ecg', Int32, self.gating_callback)

        # Set a timer to update the visualization at a fixed rate (e.g., 30 Hz)
        update_rate = 70
        self.timer = rospy.Timer(rospy.Duration(1.0 / update_rate), self.update_visualization)

        print("Initialized")

        # self.previous_value = self.latest_signal_value

    def gating_callback(self, msg):
        # Store the latest value from the message
        self.latest_signal_value = msg.data

    # def update_visualization(self, event):
    #     # Draw a white line for 1 and a black line for 0 based on the latest signal value
    #     # delta = self.latest_signal_value - self.previous_value
    #     # self.previous_value = self.latest_signal_value
    #     threshold = 620
    #     # if(self.latest_signal_value < threshold):
    #     #     print(self.latest_signal_value)
    #     color = (255, 255, 255) if self.latest_signal_value < threshold else (0, 0, 0)

    #     # Shift the image left to make space for the new signal
    #     self.signal_image[:, :-1] = self.signal_image[:, 1:]

    #     # Update the last column with the new signal value
    #     self.signal_image[:, -1] = color
        

    #     resized_image = cv2.resize(self.signal_image, (self.signal_length * 10, self.signal_height * 4))


    #     # Update the graph
    #     self.graph_image[:, :-1] = self.graph_image[:, 1:]  # Shift left to create scrolling effect
    #     normalized_value = int(
    #         (self.latest_signal_value - self.y_min) / (self.y_max - self.y_min) * (self.signal_height - 1)
    #     )
    #     normalized_value = np.clip(normalized_value, 0, self.signal_height - 1)
    #     self.graph_image[:, -1] = (0, 0, 0)  # Set background for the new column
    #     self.graph_image[self.signal_height - 1 - normalized_value, -1] = (0, 255, 0)  # Plot signal as green point

    #     # Display the image
    #     cv2.imshow("ECG Signal", resized_image)
    #     # cv2.imshow("Signal Graph", cv2.resize(self.graph_image, (self.signal_length * 10, self.signal_height * 3)))
    #     cv2.waitKey(1)

    def update_visualization(self, event):
        threshold = 620

        # Shift the image left to make space for the new signal
        self.signal_image[:, :-1] = self.signal_image[:, 1:]

        # Normalize signal to fit the signal height
        normalized_value = int(
            (self.latest_signal_value - self.y_min) / (self.y_max - self.y_min) * (self.signal_height - 1)
        )
        normalized_value = np.clip(normalized_value, 0, self.signal_height - 1)

        # Clear the last column
        self.signal_image[:, -1] = (0, 0, 0)

        # Fill up to normalized height with white
        self.signal_image[self.signal_height - 1 - normalized_value :, -1] = (255, 255, 255)

        # Resize for display
        resized_image = cv2.resize(self.signal_image, (self.signal_length * 10, self.signal_height * 4))

        # Update scrolling graph as before
        self.graph_image[:, :-1] = self.graph_image[:, 1:]
        self.graph_image[:, -1] = (0, 0, 0)
        self.graph_image[self.signal_height - 1 - normalized_value, -1] = (0, 255, 0)

        # Display
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

    
