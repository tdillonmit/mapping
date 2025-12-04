import rospy
from tkinter import Tk, Label, Scale, Button, IntVar, filedialog, ttk, BooleanVar, Checkbutton
import numpy as np
import time
import sys
from std_msgs.msg import Int32

import run_normalizedSpace
import concurrent.futures
from deformation_helpers import *
from gating_helpers import *
sys.path.append("/home/tdillon/FUNSR")
from std_msgs.msg import Bool


# publishers for each of the buttons
start_record_pub         = rospy.Publisher('/start_record', Bool, queue_size=1)
save_data_pub            = rospy.Publisher('/save_data', Bool, queue_size=1)
gate_pub                 = rospy.Publisher('/gate', Bool, queue_size=1)
funsr_start_pub          = rospy.Publisher('/funsr_start', Bool, queue_size=1)
funsr_complete_pub       = rospy.Publisher('/funsr_complete', Bool, queue_size=1)
registration_started_pub = rospy.Publisher('/registration_started', Bool, queue_size=1)
registration_done_pub    = rospy.Publisher('/registration_done', Bool, queue_size=1)
global_pause_pub         = rospy.Publisher('/global_pause', Bool, queue_size=1)
motion_capture_pub       = rospy.Publisher('/motion_capture', Bool, queue_size=1)
switch_probe_pub         = rospy.Publisher('/switch_probe', Bool, queue_size=1)
refine_started_pub       = rospy.Publisher('/refine_started', Bool, queue_size=1)
refine_done_pub          = rospy.Publisher('/refine_done', Bool, queue_size=1)
sim_device_pub           = rospy.Publisher('/sim_device', Bool, queue_size=1)
shutdown_pub             = rospy.Publisher('/shutdown', Bool, queue_size=1)
pullback_pub             = rospy.Publisher('/pullback', Int32, queue_size=1)
replay_pub               = rospy.Publisher('/replay', Bool, queue_size=1)
cardiac_pub              = rospy.Publisher('/cardiac', Bool, queue_size=1)


# reg_status_sub           = rospy.Subscriber('/reg_status', Int32, reg_status_cb)

def update_label_from_param():
    try:
        # Get the current value of the 'reg_status' parameter
        reg_status = rospy.get_param('/reg_status')
        
        # Update the label based on the reg_status value
        if reg_status == 1:
            root.percent.config(text="Closing mesh holes...")
        elif reg_status == 2:
            root.percent.config(text="Extracting centerline...")
        elif reg_status == 3:
            root.percent.config(text="Rigid transform of data...")
        elif reg_status == 4:
            root.percent.config(text="Filtering spurious points...")
        elif reg_status == 5:
            root.percent.config(text="FenFit correspondence estimation...")
        elif reg_status == 6:
            root.percent.config(text="Viterbi correspondence estimation...")
        elif reg_status == 7:
            root.percent.config(text="Saving data...")
        elif reg_status == 8:
            root.percent.config(text="Registration complete...")
            return
        elif reg_status == 9:
            root.percent.config(text="Called CPD...")
        
        # Update the GUI once after all changes
        root.update_idletasks()

    except KeyError:
        print("Parameter /reg_status not found yet.")

    return True


def start_parameter_check():
    # Use root.after to check every 500ms (0.5 seconds)
    continue_checking = update_label_from_param()
    if continue_checking:
        root.after(500, start_parameter_check)  # Continue checking every 500ms



def start_record():
    start_record_pub.publish(True)

def save_data():
    save_data_pub.publish(True)
    pullback_pub.publish(0)

def call_gating():
    gate_pub.publish(True)

    time.sleep(1)
    #complete this function
    dataset = rospy.get_param('dataset', 0)
    if(dataset ==0):
        select_folder()
    dataset = rospy.get_param('dataset', 0)

    run_gating(dataset)
    # root.percent.config(text = "Gating Complete")
    # root.update_idletasks()

def call_funsr():
    funsr_start_pub.publish(True)

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
    # root.percent.config(text = "Initializing surface ..")
    # root.update_idletasks()
    run_normalizedSpace.run_funsr(dataset, root)
    # root.percent.config(text = "Surface Initialized!")
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # os.chdir(script_dir)

    # root.after(100, update_progress, progress_bar, percent, root)


def load_previous_surface_geometry():
    funsr_complete_pub.publish(True)

def call_register(notify_mapping=1):
    registration_started_pub.publish(True)

    #complete this function
    dataset = rospy.get_param('dataset', 0)
    if(dataset ==0):
        select_folder()
    dataset = rospy.get_param('dataset', 0)

    # visualize = checkbox_var.get()
    visualize = 0

    print("calling registration indirectly")
    refine=0

    # start checking here?
    start_parameter_check()

    call_registration_indirectly(dataset, visualize, refine, 0)

    print("finished indirect call!")

    if(notify_mapping==1):
        registration_done_pub.publish(True)

# FULL MOTION CAPTURE PIPELINE AUTOMATED FOR USER
def call_motion_capture():

    # bin 3 folder path
    folder_path = rospy.get_param('dataset', 0)
    bin_3_folder_path = folder_path+'/gated/bin_3'

    # do gating here first
    call_gating()

    
    rospy.set_param("dataset", bin_3_folder_path)
    folder_prompt.config(text=f"{bin_3_folder_path}")  # Update label with the selected folder

    # start the replay of bin 3
    replay_pub.publish(True)
    time.sleep(2) # give mapping a chance to register we've pressed the button
    replay_pub.publish(False)

    # check to see if bin 3 is finished mapping
    replay_done = rospy.get_param('replay_done', 0)
    while(replay_done !=1):
        print("looping bin 3")
        time.sleep(1)
        replay_done = rospy.get_param('replay_done', 0)
    
    print("bin 3 complete, starting bin 8! ")

    # bin 8 folder path
    bin_8_folder_path = folder_path+'/gated/bin_8'
    rospy.set_param("dataset", bin_8_folder_path)
    folder_prompt.config(text=f"{bin_8_folder_path}")  # Update label with the selected folder

    # start bin 8 folder path
    replay_pub.publish(True)
    time.sleep(2) # give mapping a chance to register we've pressed the button
    replay_pub.publish(False)

    # check to see if bin 8 is finished mapping
    replay_done = rospy.get_param('replay_done', 0)
    while(replay_done !=1):
        print("looping bin 8")
        time.sleep(1)
        replay_done = rospy.get_param('replay_done', 0)

    # motion_capture_pub.publish(True)
    print("Registering CT mesh to bin 3 (can change to bin 8 with code tweaks)")
    rospy.set_param("dataset", bin_3_folder_path)
    call_register(notify_mapping=0)

    rospy.set_param("dataset", folder_path) # go back to high level folder

    print("initializing motion capture...")
     
    

    cardiac_pub.publish(True)

    #complete this function
    dataset = rospy.get_param('dataset', 0)
    if(dataset ==0):
        select_folder()
    dataset = rospy.get_param('dataset', 0)

    # visualize = checkbox_var.get()
    visualize = 0

    # start checking here?
    start_parameter_check()

    initialize_motion_capture_indirectly(dataset, visualize)

    bin_3_folder_path = folder_path+'/gated/bin_3'
    rospy.set_param("dataset", bin_3_folder_path)
    folder_prompt.config(text=f"{bin_3_folder_path}")  # Update label with the selected folder

    print("Finished!")

    
    # motion_capture_pub.publish(False) 

    registration_done_pub.publish(True)




def load_previous_registration():
    registration_done_pub.publish(True)

def switch_probe():
    switch_probe_pub.publish(True)
    # switch_probe_pub.publish(False)

def call_refine():
    print("publishing refine start")
    refine_started_pub.publish(True)

def save_refine():

    # Publish that refinement is starting
    refine_done_pub.publish(True)
    


    # Get dataset path
    # dataset = rospy.get_param('dataset', 0)
    # if dataset == 0:
    #     select_folder()
    # dataset = rospy.get_param('dataset', 0)

    # visualize = 0
    # refine = 1

    # print("calling refine indirectly")
    # call_refine_indirectly(dataset, visualize, refine)

  
    # Publish that refinement is starting
    # refine_done_pub.publish(True)

def sim_device_deployment():
    sim_device_pub.publish(True)

def toggle_fenestrate():
    fenestrate = rospy.get_param("fenestrate", 0)
    if(fenestrate==1):
        fenestrate=0
    if(fenestrate==0):
        fenestrate=1
    rospy.set_param("fenestrate", fenestrate)
    print("fenestration mode toggled")

def on_closing():
    shutdown_pub.publish(True)
    root.quit()
    root.destroy()
    rospy.signal_shutdown('User quitted')

def start_pullback():
    pullback_pub.publish(1)

def stop_pullback():
    pullback_pub.publish(0)

def call_replay():
    replay_pub.publish(True)
    time.sleep(6)
    replay_pub.publish(False)

def close_gui():
    root.quit()
    root.destroy()
    rospy.signal_shutdown('User quitted')

def quit_aortascope():
    print("closing gui!")
    root.quit()
    root.destroy()
    rospy.signal_shutdown('User quitted')

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


    

try:


    # pullback_pub = rospy.Publisher('/pullback', Int32, queue_size=1)

    # Initialize rospy without creating a node
    rospy.init_node('mapping_parameters_gui', anonymous=True)

    rospy.set_param("fenestrate", 0)

    # pullback_pub.publish(0)

    # Create the GUI window
    root = Tk()
    root.title("AortaScope")

    # single display
    # root.geometry("860x1320+10+10") 



    # change to 0.5 or 1 for single or double display
    # display_scale_factor = 0.5

    time.sleep(3)
    double_display = rospy.get_param('double_display', 0)
    print("gui double display is", double_display)
    if(double_display == 1):
        display_scale_factor =0.5
        # root.geometry("410x527+0+0")
        root.geometry("437x527+0+0")
        

    else:
        display_scale_factor =1
        root.geometry("860x1320+10+10")  
        
    
 


 

    # pullback start / stop
    # pullback = IntVar(value=0)  # 0 = stopped, 1 = started

    # Add buttons to start and stop pullback

    # Define button properties (font size and dimensions)
    standard_font_size = 14
    
    

    # single display
    standard_font_size = int( 14  * 0.8)
    button_width = int(30 )  # Button width (characters)
    button_height = int(0.8 )  # Button height (lines)
    padding_y = int(14 * display_scale_factor * 0.4)
    # padding_x = int(105 * display_scale_factor)
    padding_x = int(119 * display_scale_factor)

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

    # refine_button = Button(root, text="Map EVAR Graft (Start)", font=button_font, width=button_width, height=button_height, command = call_refine)
    # refine_button.grid(row=3, column=0, padx=padding_x, pady=padding_y)

    # refine_stop_button = Button(root, text="Map EVAR Graft (Stop)", font=button_font, width=button_width, height=button_height, command = save_refine)
    # refine_stop_button.grid(row=4, column=0, padx=padding_x, pady=padding_y)

    save_data_button = Button(root, text="Save Data", font=button_font, width=button_width, height=button_height, command=save_data)
    save_data_button.grid(row=5, column=0, padx=padding_x, pady=padding_y)

    # Additional buttons
    # funsr_button = Button(root, text="Initialize Registration", font=button_font, width=button_width, height=button_height, command = call_funsr)
    # funsr_button.grid(row=6, column=0, padx=padding_x, pady=padding_y)

    funsr_button = Button(root, text="Perform Gating", font=button_font, width=button_width, height=button_height, command = call_gating)
    funsr_button.grid(row=6, column=0, padx=padding_x, pady=padding_y)

    load_button = Button(root, text="Replay Dataset", font=button_font, width=button_width, height=button_height, command = call_replay)
    load_button.grid(row=7, column=0, padx=padding_x, pady=padding_y)

    register_button = Button(root, text="3D Preop CT Registration", font=button_font, width=button_width, height=button_height, command = call_register)
    register_button.grid(row=8, column=0, padx=padding_x, pady=padding_y)

    # checkbox_var = BooleanVar()
    # checkbox = Checkbutton(root, text="Debug", variable=checkbox_var)
    # checkbox.grid(row=8, column=1,  columnspan=2, padx=0, pady=0)

    sim_device = Button(root, text="4D Preop CT Registration", font=button_font, width=button_width, height=button_height, command = call_motion_capture)
    sim_device.grid(row=9, column=0, padx=padding_x, pady=padding_y)

    load_register = Button(root, text="Load Previously Registered Scan", font=button_font, width=button_width, height=button_height, command = load_previous_registration)
    load_register.grid(row=10, column=0, padx=padding_x, pady=padding_y)

    tracking_button = Button(root, text="Switch Endoscopic View", font=button_font, width=button_width, height=button_height, command =switch_probe)
    tracking_button.grid(row=11, column=0, padx=padding_x, pady=padding_y)

    # target_button = Button(root, text="Switch Target Vessel", font=button_font, width=button_width, height=button_height, command = switch_vessel)
    # target_button.grid(row=11, column=0, padx=padding_x, pady=padding_y)

    # refine_button = Button(root, text="Refine Abdominal Registration (Start)", font=button_font, width=button_width, height=button_height, command = call_refine)
    # refine_button.grid(row=11, column=0, padx=padding_x, pady=padding_y)

    # refine_stop_button = Button(root, text="Refine Abdominal Registration (Stop)", font=button_font, width=button_width, height=button_height, command = save_refine)
    # refine_stop_button.grid(row=12, column=0, padx=padding_x, pady=padding_y)

    

    # quit_button = Button(root, text="Quit Application", font=button_font, width=button_width, height=button_height, command = quit_aortascope)
    # quit_button.grid(row=13, column=0, padx=padding_x, pady=padding_y)

    
    

    sim_device = Button(root, text="Load IVUS Mesh Only", font=button_font, width=button_width, height=button_height, command = load_previous_surface_geometry)
    sim_device.grid(row=12, column=0, padx=padding_x, pady=padding_y)

    # sim_device = Button(root, text="Simulate Device Deployment", font=button_font, width=button_width, height=button_height, command = sim_device_deployment)
    # sim_device.grid(row=13, column=0, padx=padding_x, pady=padding_y)

    # fenestrate_graft = Button(root, text="Fenestrate Graft", font=button_font, width=button_width, height=button_height, command = toggle_fenestrate)
    # fenestrate_graft.grid(row=13, column=0, padx=padding_x, pady=padding_y)


    # PROGRESS BAR AND PERCENTAGE
    
    root.percent = Label(root, text="", font=("Arial", 10))
    root.percent.grid(row=14, column=0)

    # root.progress_bar = ttk.Progressbar(root, orient="horizontal", length=(500* display_scale_factor), mode="determinate", 
    #                             style="TProgressbar")
    # root.progress_bar.grid(row=14, column=0, padx=padding_x, pady=padding_y*0.6)
    # style = ttk.Style()
    # style.configure("TProgressbar", thickness=int(15 * display_scale_factor))  # Increase the thickness of the progress bar

    
    # self.update_progress(progress_bar, percent, percentage_complete, root)

    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()



except (rospy.ROSInterruptException, KeyboardInterrupt):
    # Handle both ROSInterruptException and KeyboardInterrupt (Ctrl+C)
    print("closing")
    close_gui()





