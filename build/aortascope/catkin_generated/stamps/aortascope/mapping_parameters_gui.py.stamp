import rospy
from tkinter import Tk, Label, Scale, Button, IntVar, filedialog
import numpy as np
import time
import sys

import run_normalizedSpace
import concurrent.futures
from deformation_helpers import *
sys.path.append("/home/tdillon/FUNSR")

def select_folder():
    folder_path = filedialog.askdirectory()  # Open folder selection dialog
    if folder_path:  # If a folder is selected
        print(f"Selected folder: {folder_path}")
        rospy.set_param("dataset", folder_path)
        
    else:
        print("No folder selected")

def start_record():
    
    rospy.set_param("start_record", 1)
   
def save_data():
    
    rospy.set_param("save_data", 1)

def call_funsr():
    

    # GATING
    file_path = self.dataset + '/ungated/ecg_signal/image_times.npy'
    if os.path.isfile(file_path):
        # perform gating
        run_gating(self.dataset)
        # choose diastole
        self.dataset = self.dataset + '/gated/bin_0'

    # Run FUNSR in parallel (non-blocking)
    run_normalizedSpace.run_funsr(dataset)
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(script_dir)

        
def update_progress(progress_bar, label, value):
    # Update the progress bar and the label
    progress_bar['value'] = value
    label.config(text=f"{value}%")
    root.after(100, update_progress, progress_bar, label, value + 1)


def call_register():
    
    #complete this function
    rospy.get_param('dataset', 0)
    register_branches(dataset)
    rospy.set_param('registration_done', 1)
    

def switch_probe():
    rospy.set_param('switch_probe', 1)
    time.sleep(1)
    

def switch_vessel():
    
    pass

def on_closing():
    root.destroy()

# only necessary because two functions access the same variable (i.e., pullback)
def update_parameters(event=None):

    # Update the parameters (you may want to add additional logic or error handling)
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
    root.title("AortaScope")

    # pullback start / stop
    pullback = IntVar(value=0)  # 0 = stopped, 1 = started

    # Add buttons to start and stop pullback

    select_button = Button(root, text="Set Folder Directory", command=select_folder)
    select_button.pack()

    record_button = Button(root, text="Start Recording", command=start_record)
    record_button.pack()

    start_button = Button(root, text="Start Pullback", command=start_pullback)
    start_button.pack()

    stop_button = Button(root, text="Stop Pullback", command=stop_pullback)
    stop_button.pack()

    save_data_button = Button(root,text="Finish Recording", command=save_data)
    save_data_button.pack()
   
    funsr_button = Button(root,text="Extract Surface Geometry", command=call_funsr)
    funsr_button.pack()

    register_button = Button(root,text="Register Preoperative Scan", command=call_register)
    register_button.pack()

    tracking_button = Button(root,text="Switch Endoscopic View", command=switch_probe)
    tracking_button.pack()

    target_button = Button(root,text="Switch Target Vessel", command=switch_vessel)
    target_button.pack()

   

    root.protocol("WM_DELETE_WINDOW", on_closing)
    # Run the GUI event loop
    root.mainloop()



except (rospy.ROSInterruptException, KeyboardInterrupt):
    # Handle both ROSInterruptException and KeyboardInterrupt (Ctrl+C)
    pass





