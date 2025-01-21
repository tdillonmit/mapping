import rospy
from tkinter import Tk, Label, Scale, Button
import numpy as np

def update_parameters(event):
    # Update the parameters (you may want to add additional logic or error handling)
    rospy.set_param("angle", angle.get())
    rospy.set_param("threshold", threshold.get())
    rospy.set_param("scaling", scaling.get())
    rospy.set_param("translation", translation.get())
    rospy.set_param("radial_offset", radial_offset.get())
    rospy.set_param("oclock", oclock.get())

# Initialize rospy without creating a node
rospy.init_node('parameter_tuning_gui', anonymous=True)

# Create the GUI window
root = Tk()
root.title("ROS Parameter Tuning")

# Create a Scale widget for angle
label_angle = Label(root, text="Angle:")
label_angle.pack()
angle = Scale(root, from_=0, to=2*np.pi, orient="horizontal", resolution=2*np.pi/128, length=300, command=update_parameters)
angle.set(3.24)  # Default value is 45
angle.pack()

# Create a Scale widget for angle
label_translation = Label(root, text="Translation:")
label_translation.pack()
translation = Scale(root, from_=0.005, to=0.2, orient="horizontal", resolution=0.0005, length=300, command=update_parameters)
translation.set(0.009)  # Default value is 50
translation.pack()

# Create a Scale widget for threshold
label_threshold = Label(root, text="Threshold:")
label_threshold.pack()
threshold = Scale(root, from_=100, to=255, orient="horizontal", resolution=1, length=300, command=update_parameters)
threshold.set(112)  # Default value is 50
threshold.pack()

# Create a Scale widget for scaling_factor
label_scaling = Label(root, text="Scaling:")
label_scaling.pack()
scaling = Scale(root, from_=0, to=0.00015, orient="horizontal", resolution=0.0000005, length=300, command=update_parameters)
scaling.set(0.0000765)  # Default value is 1.0
scaling.pack()

# Create a Scale widget for radial_offset
label_translation = Label(root, text="Radial offset:")
label_translation.pack()
radial_offset = Scale(root, from_=0.0, to=0.004, orient="horizontal", resolution=0.00005, length=300, command=update_parameters)
radial_offset.set(0.00225)  # Default value is 50
radial_offset.pack()

# Create a Scale widget for oclock
label_translation = Label(root, text="o'clock position:")
label_translation.pack()
oclock= Scale(root, from_=0.0, to=2*np.pi, orient="horizontal", resolution=2*np.pi/128, length=300, command=update_parameters)
oclock.set(2.16)  # Default value is 50
oclock.pack()

# Create a button to trigger parameter update
# update_button = Button(root, text="Update Parameters", command=update_parameters)
# update_button.pack()

# Run the GUI event loop
root.mainloop()
