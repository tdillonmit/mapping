import rospy
from tkinter import Tk, Label, Scale, Button
import numpy as np

def on_closing():
    root.destroy()

def update_parameters(event):
    # Update the parameters (you may want to add additional logic or error handling)
    rospy.set_param("no_points", no_points.get())
    rospy.set_param("threshold", threshold.get())
    rospy.set_param("crop_index", crop_index.get())

def on_closing():
    root.destroy()

try:
    # Initialize rospy without creating a node
    rospy.init_node('mapping_parameters_gui', anonymous=True)

    # Create the GUI window
    root = Tk()
    root.title("Mapping Parameter Tuning")

    # Create a Scale widget for threshold
    label_threshold = Label(root, text="Threshold:")
    label_threshold.pack()
    threshold = Scale(root, from_=50, to=255, orient="horizontal", resolution=1, length=300, command=update_parameters)
    threshold.set(50)  # Default value is 50
    threshold.pack()

    # Create a Scale widget for scaling_factor
    label_scaling = Label(root, text="Number of points:")
    label_scaling.pack()
    no_points= Scale(root, from_=100, to=8000, orient="horizontal", resolution=10, length=300, command=update_parameters)
    no_points.set(1000)  # Default value is 1.0
    no_points.pack()

    # Create a Scale widget for scaling_factor
    label_scaling = Label(root, text="Crop index:")
    label_scaling.pack()
    crop_index= Scale(root, from_=1, to=500, orient="horizontal", resolution=1, length=300, command=update_parameters)
    crop_index.set(60)  # Default value is 1.0
    crop_index.pack()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    # Run the GUI event loop
    root.mainloop()

except (rospy.ROSInterruptException, KeyboardInterrupt):
    # Handle both ROSInterruptException and KeyboardInterrupt (Ctrl+C)
    pass
