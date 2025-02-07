import rospy
from tkinter import Tk, Label, Scale, Button, IntVar, filedialog, ttk, BooleanVar, Checkbutton
import numpy as np
import time
import sys

import run_normalizedSpace
import concurrent.futures
from deformation_helpers import *
sys.path.append("/home/tdillon/FUNSR")

def close_gui():
    root.quit()
    root.destroy()

def select_folder():

    initial_dir = os.path.expanduser("/home/tdillon/datasets")

    folder_path = filedialog.askdirectory(initialdir=initial_dir,title = 'Select Folder')  # Open folder selection dialog
   
    if folder_path:  # If a folder is selected
        print(f"Selected folder: {folder_path}")
        rospy.set_param("dataset", folder_path)
        folder_prompt.config(text=f"{folder_path}")  # Update label with the selected folder
    else:
        print("No folder selected")
        folder_prompt.config(text="No folder selected")  # Update label with default message

def start_record():
    
    rospy.set_param("start_record", 1)
   
def save_data():
    
    rospy.set_param("save_data", 1)

def call_funsr():
    
    print("Surface Extraction module called!")
    rospy.set_param("funsr_started", 1)
    dataset = rospy.get_param('dataset', 0)
    if(dataset ==0):
        select_folder()
    dataset = rospy.get_param('dataset', 0)



    # GATING
    file_path = dataset + '/ungated/ecg_signal/image_times.npy'
    if os.path.isfile(file_path):
        # perform gating
        run_gating(dataset)
        # choose diastole
        dataset = dataset + '/gated/bin_0'

    # Run FUNSR in parallel (non-blocking)
    time.sleep(1)
    root.percent.config(text = "Initializing surface ..")
    root.update_idletasks()
    run_normalizedSpace.run_funsr(dataset, root)
    root.percent.config(text = "Surface Initialized!")
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(script_dir)

    # root.after(100, update_progress, progress_bar, percent, root)

def load_previous_surface_geometry():
    
    #complete this function
    dataset = rospy.set_param('funsr_done', 1)

def load_previous_registration():
    
    #complete this function
    dataset = rospy.set_param('registration_done', 1)
        



def call_register():
    
    rospy.set_param("registration_started", 1)
    #complete this function
    dataset = rospy.get_param('dataset', 0)
    if(dataset ==0):
        select_folder()
    dataset = rospy.get_param('dataset', 0)

    # visualize = checkbox_var.get()
    visualize = 0
    root.percent.config(text = "Computing Registration")
    root.update_idletasks()
    call_registration_indirectly(dataset, visualize)
    root.percent.config(text = "Registration Complete")
    rospy.set_param('registration_done', 1)
    root.update_idletasks()
 
    
    
    


def switch_probe():
    rospy.set_param('switch_probe', 1)
    time.sleep(1)
    

def switch_vessel():
    
    pass



# only necessary because two functions access the same variable (i.e., pullback)
def update_parameters(event=None):

    # Update the parameters (you may want to add additional logic or error handling)
    rospy.set_param("pullback", pullback.get())

def on_closing():
    root.quit()
    root.destroy()
    rospy.set_param('shutdown', 1)
    rospy.signal_shutdown('User quitted')

def start_pullback():
    """
    Set the pullback state to 1 (start).
    """
    
    # pullback.set(1)
    rospy.set_param("pullback", 1)
    # update_parameters()

    pullback_check = rospy.get_param("pullback", 0)
    
    
    print("pullback STARTED")
    print("pullback check", pullback_check)

def stop_pullback():
    """
    Set the pullback state to 0 (stop).
    """
    # pullback.set(0)
    # update_parameters()
    # print("pullback is", pullback)
    rospy.set_param("pullback", 0)

    pullback_check = rospy.get_param("pullback", 0)
    print("pullback STOPPED")
    print("pullback check", pullback_check)

# def update_progress(progress_bar, percent, value, root):
#         value = rospy.get_param('funsr_percent', 0)
#         print("fetched value!", value)
#         # Update the progress bar and the label
#         progress_bar['value'] = value
#         percent.config(text=f"{value}%")
#         # root.after(100, update_progress, progress_bar, label, value + 1)
#         root.update_idletasks()  # Update the UI immediately

def quit_aortascope():
    print("closing gui!")
    root.quit()
    root.destroy()
    rospy.set_param('shutdown', 1)
    time.sleep(0.3)
    rospy.signal_shutdown('User quitted')
    
 

try:
    # Initialize rospy without creating a node
    rospy.init_node('mapping_parameters_gui', anonymous=True)

    # Create the GUI window
    root = Tk()
    root.title("AortaScope")

    # single display
    root.geometry("860x1320+10+10") 

    # double display
    # root.geometry("410x527+0+0") 
 

    # pullback start / stop
    pullback = IntVar(value=0)  # 0 = stopped, 1 = started

    # Add buttons to start and stop pullback

    # Define button properties (font size and dimensions)
    standard_font_size = 14
    
    # change to 1 or 0.5 for single or double display
    display_scale_factor = 0.5

    # single display
    standard_font_size = int( 14  * 0.8)
    button_width = int(30 )  # Button width (characters)
    button_height = int(1 )  # Button height (lines)
    padding_y = int(17 * display_scale_factor * 0.4)
    padding_x = int(105 * display_scale_factor)

    button_font = ("Arial", standard_font_size)  # Font size 14

    # Create buttons with larger sizes and left-alignment
    folder_prompt = Label(root, text="No Folder Selected, Select Folder", font=("Arial", int(standard_font_size*0.85)), width=int(button_width*1.4), height=button_height)
    folder_prompt.grid(row=0, column=0, padx=padding_x, pady=padding_y )  # Align left

    # Create buttons with larger sizes and left-alignment
    select_button = Button(root, text="Set Folder Directory", font=button_font, width=button_width, height=button_height, command=select_folder)
    select_button.grid(row=1, column=0, padx=padding_x, pady=padding_y )  # Align left

    record_button = Button(root, text="Start Recording", font=button_font, width=button_width, height=button_height, command=start_record)
    record_button.grid(row=2, column=0, padx=padding_x, pady=padding_y)

    start_button = Button(root, text="Start Pullback Device", font=button_font, width=button_width, height=button_height, command=start_pullback)
    start_button.grid(row=3, column=0, padx=padding_x, pady=padding_y)

    stop_button = Button(root, text="Stop Pullback Device", font=button_font, width=button_width, height=button_height, command=stop_pullback)
    stop_button.grid(row=4, column=0, padx=padding_x, pady=padding_y)

    save_data_button = Button(root, text="Finish Recording", font=button_font, width=button_width, height=button_height, command=save_data)
    save_data_button.grid(row=5, column=0, padx=padding_x, pady=padding_y)

    # Additional buttons
    funsr_button = Button(root, text="Initialize Surface Registration", font=button_font, width=button_width, height=button_height, command = call_funsr)
    funsr_button.grid(row=6, column=0, padx=padding_x, pady=padding_y)

    load_button = Button(root, text="Load Previous Initialization", font=button_font, width=button_width, height=button_height, command = load_previous_surface_geometry)
    load_button.grid(row=7, column=0, padx=padding_x, pady=padding_y)

    register_button = Button(root, text="Register Preoperative Scan", font=button_font, width=button_width, height=button_height, command = call_register)
    register_button.grid(row=8, column=0, padx=padding_x, pady=padding_y)

    # checkbox_var = BooleanVar()
    # checkbox = Checkbutton(root, text="Debug", variable=checkbox_var)
    # checkbox.grid(row=8, column=1,  columnspan=2, padx=0, pady=0)

    load_register = Button(root, text="Load Previously Registered Scan", font=button_font, width=button_width, height=button_height, command = load_previous_registration)
    load_register.grid(row=9, column=0, padx=padding_x, pady=padding_y)

    tracking_button = Button(root, text="Switch Endoscopic View", font=button_font, width=button_width, height=button_height, command =switch_probe)
    tracking_button.grid(row=10, column=0, padx=padding_x, pady=padding_y)

    target_button = Button(root, text="Switch Target Vessel", font=button_font, width=button_width, height=button_height, command = switch_vessel)
    target_button.grid(row=11, column=0, padx=padding_x, pady=padding_y)

    quit_button = Button(root, text="Quit Application", font=button_font, width=button_width, height=button_height, command = quit_aortascope)
    quit_button.grid(row=12, column=0, padx=padding_x, pady=padding_y)

    root.percent = Label(root, text="", font=("Arial", 10))
    root.percent.grid(row=13, column=0)

    root.progress_bar = ttk.Progressbar(root, orient="horizontal", length=(500* display_scale_factor), mode="determinate", 
                                style="TProgressbar")
    root.progress_bar.grid(row=14, column=0, padx=padding_x, pady=padding_y*0.6)
    style = ttk.Style()
    style.configure("TProgressbar", thickness=int(15 * display_scale_factor))  # Increase the thickness of the progress bar

    
    # self.update_progress(progress_bar, percent, percentage_complete, root)

    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()



except (rospy.ROSInterruptException, KeyboardInterrupt):
    # Handle both ROSInterruptException and KeyboardInterrupt (Ctrl+C)
    print("closing")
    close_gui()





