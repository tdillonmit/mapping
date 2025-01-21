import rospy
from tkinter import Tk, Label, Scale, Button, IntVar
import numpy as np

def on_closing():
    root.destroy()

def update_parameters(event=None):
    # Update the parameters (you may want to add additional logic or error handling)
    rospy.set_param("no_points", no_points.get())
    rospy.set_param("threshold", threshold.get())
    rospy.set_param("crop_index", crop_index.get())
    rospy.set_param("pullback", pullback.get())

def on_closing():
    root.destroy()

def start_pullback():
    """
    Set the pullback state to 1 (start).
    """
    pullback.set(1)
    update_parameters()

def stop_pullback():
    """
    Set the pullback state to 0 (stop).
    """
    pullback.set(0)
    update_parameters()

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

    # pullback start / stop
    pullback = IntVar(value=0)  # 0 = stopped, 1 = started

    # Add buttons to start and stop pullback
    start_button = Button(root, text="Start Pullback", command=start_pullback)
    start_button.pack()

    stop_button = Button(root, text="Stop Pullback", command=stop_pullback)
    stop_button.pack()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    # Run the GUI event loop
    root.mainloop()

except (rospy.ROSInterruptException, KeyboardInterrupt):
    # Handle both ROSInterruptException and KeyboardInterrupt (Ctrl+C)
    pass
