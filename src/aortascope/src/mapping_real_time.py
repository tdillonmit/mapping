#!/usr/bin/env python3.9

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped
import cv2
import tf2_ros
import open3d as o3d
import numpy as np
import copy
from std_msgs.msg import Header  
from std_msgs.msg import Int32
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Bool
from std_msgs.msg import Int32MultiArray
import math
import yaml
import os
import time
import random
# import threading
import subprocess
from collections import deque
import gc
# import snake as sn
from keras.layers import TFSMLayer
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ReLU, Activation, TimeDistributed
from segmentation_helpers import BlurPool
import psutil, os, gc
from cv_bridge import CvBridge
bridge = CvBridge()

import matplotlib  
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# from segmentation_helpers import build_temporal_segmenter 


from voxblox import (
    BaseTsdfIntegrator,
    FastTsdfIntegrator,
    MergedTsdfIntegrator,
    SimpleTsdfIntegrator,
)

from segmentation_helpers import *
from reconstruction_helpers import *
# from numba import cuda
import tkinter as tk
from tkinter import filedialog






    




class PointCloudUpdater:


            

    def __init__(self):

        self.prev_msg = None

        self.wireframe_gen = WireframeGenerator()

        # for fenestration
        self.absolute_positions = np.empty((0,3))
        self.radial_vectors = np.empty((0,3))


        now = rospy.Time.now()
        self.last_image_time = now


        
        self.dissection_mapping = 0

        
        # INITIALIZE IN MAPPING CONFIGURATION
        if(self.dissection_mapping ==1):
            with open('/home/tdillon/mapping/src/aortascope_mapping_params_dissection.yaml', 'r') as file:
                config_yaml = yaml.safe_load(file)

        else:
            with open('/home/tdillon/mapping/src/aortascope_mapping_params.yaml', 'r') as file:
                config_yaml = yaml.safe_load(file)

        self.load_parameters(config_yaml)

        self.extend = 0

        
        # AORTASCOPE CONSTANTS

        # ------- INITIALIZE DEEPLUMEN ML MODEL ------- #
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
     
        if not hasattr(self, 'model'):
            self.initialize_deeplumen_model()

        # philips
        self.image_width = rospy.get_param('/usb_cam/image_width', default=1280)
        self.image_height = rospy.get_param('/usb_cam/image_height', default=1024)

        # boston scientific avvigo
        # self.image_width = rospy.get_param('/usb_cam/image_width', default=1024)
        # self.image_height = rospy.get_param('/usb_cam/image_height', default=576)

        # boston scientific ilab 
        # self.image_width = rospy.get_param('/usb_cam/image_width', default=1280)
        # self.image_height = rospy.get_param('/usb_cam/image_height', default=1024)

        self.crop_radius=10
        no_points=rospy.get_param('no_points')
        self.previous_no_points=no_points
        

        # what number image callback are we on?
        self.image_call=1

        self.image_number = 1
        
        


        # ---- GUI SPECIFIC ------ #
        # ---- INITIALIZE VISUALIZERS ----- #

        
        # so its accessible to gui script
        rospy.set_param('double_display', self.double_display)
        self.width_scaling = 0.5
        self.height_scaling = (1/2.2222)



        self.vis = o3d.visualization.Visualizer()

        if(self.double_display==0):
            self.vis.create_window(window_name="3D Reconstruction", width = 1700, height = 3000, left = 2200, top = 0)  # Custom window title
        else:
            self.vis.create_window(window_name="3D Reconstruction", width = int(self.width_scaling*1765), height = int(self.height_scaling*3000), left = int(self.width_scaling*2080), top = int(self.height_scaling*0))  # Custom window title

        

        self.vis.get_render_option().mesh_show_back_face = True
        self.vis.poll_events()
        self.vis.update_renderer()

        self.vis2 = o3d.visualization.Visualizer()    
        if(self.double_display==0):    
            self.vis2.create_window(window_name="Simulated Endoscope View", width = 2000, height = 900, left = 0, top = 1500)  # Custom window title
        else:
            self.vis2.create_window(window_name="Simulated Endoscope View", width = int(self.width_scaling*1930), height = int(self.height_scaling*1000), left = int(self.width_scaling*0), top = int(self.height_scaling*1400)) 
        self.vis2.get_render_option().mesh_show_back_face = True
        self.vis2.poll_events()
        self.vis2.update_renderer()
    
        # ----- INITIALIZE RECONSTRUCTIONS ------ #
        self.near_point_cloud = o3d.geometry.PointCloud()
        self.far_point_cloud = o3d.geometry.PointCloud()
        self.dissection_flap_point_cloud = o3d.geometry.PointCloud()
        self.volumetric_far_point_cloud = o3d.geometry.PointCloud() 
        self.volumetric_near_point_cloud = o3d.geometry.PointCloud() 
        self.boundary_near_point_cloud = o3d.geometry.PointCloud() 
        self.point_cloud = o3d.geometry.PointCloud()
        self.orifice_center_point_cloud = o3d.geometry.PointCloud()


        # self.vis.add_geometry(self.near_point_cloud)
        # self.vis.add_geometry(self.far_point_cloud)
        # self.vis.add_geometry(self.dissection_flap_point_cloud)

        if(self.figure_mapping!=1):
            self.vis.add_geometry(self.volumetric_near_point_cloud)
            self.vis.add_geometry(self.orifice_center_point_cloud)
        
        # self.vis.add_geometry(self.volumetric_far_point_cloud)
        self.vis.add_geometry(self.boundary_near_point_cloud)
        # self.vis.add_geometry(self.point_cloud)
        



        # ----- INITIALIZE TRACKER FRAMES ------ #
        self.frame_scaling=0.025
        self.tracker_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.tracker_frame.scale(self.frame_scaling,center=[0,0,0])

        self.baseframe=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.baseframe.scale(self.frame_scaling,center=[0,0,0])

        self.us_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.us_frame.scale(self.frame_scaling,center=[0,0,0])

        self.tracker = o3d.geometry.TriangleMesh.create_cylinder(radius=0.001, height=0.01, resolution=40)
        self.tracker.compute_vertex_normals()
        self.tracker.paint_uniform_color([1,0.5,0])

        # self.catheter = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0015, height=0.01)
        self.catheter = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0015, height=0.007)
        self.catheter.compute_vertex_normals()
        self.catheter.paint_uniform_color([0,0,1])
        # self.catheter.paint_uniform_color([1,0,0])

        self.guidewire_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.000225, height=0.004)
        self.guidewire_cylinder.compute_vertex_normals()
        self.guidewire_cylinder.paint_uniform_color([0,1,0])
        self.guidewire_pointcloud_base = o3d.geometry.PointCloud()
        n = 20
        z = np.linspace(0, 0.03, n)
        x = np.zeros_like(z)
        y = np.zeros_like(z)
        base_points = np.stack((x, y, z), axis=1)
        self.guidewire_pointcloud_base.points = o3d.utility.Vector3dVector(base_points)
        self.guidewire_pointcloud_base.paint_uniform_color([0,1,0])
        self.guidewire_pointcloud = o3d.geometry.PointCloud()

        self.bending_segment = o3d.geometry.TriangleMesh()
        self.bending_segment_color = [0.678, 0.847, 0.902]
        self.bending_segment.paint_uniform_color(self.bending_segment_color)
        # self.steerable_arc_length = 0.015
        # self.steerable_arc_length = 0.027
        # self.steerable_arc_length = 0.028  # aptus tourguide
        self.steerable_arc_length = 0.028 # aptus endoanchor and tourguide

        self.catheter_base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.catheter_base_frame .scale(self.frame_scaling,center=[0,0,0])

        self.catheter_shaft = o3d.geometry.TriangleMesh()
        self.catheter_shaft.paint_uniform_color([0,0,1])
       
        
        
        self.previous_transform=np.eye(4)
        self.previous_transform_1=np.eye(4)
        self.previous_transform_us=np.eye(4)
        self.previous_catheter_transform=np.eye(4)
        self.previous_tracker_transform=np.eye(4)
        self.previous_guidewire_transform=np.eye(4)
        self.previous_bending_transform = np.eye(4)
        self.previous_catheter_base = np.eye(4)

        self.vis.add_geometry(self.catheter)
        self.vis.add_geometry(self.guidewire_cylinder)
        self.vis.add_geometry(self.tracker)

        # print("added us frame")
        self.vis.add_geometry(self.us_frame)
        # self.vis.add_geometry(self.tracker_frame)
        self.vis.add_geometry(self.baseframe)
        self.vis.add_geometry(self.guidewire_pointcloud)
        self.vis.add_geometry(self.bending_segment)
        # self.vis2.add_geometry(self.bending_segment)
        self.vis.add_geometry(self.catheter_shaft)
        # self.vis.add_geometry(self.catheter_base_frame)

    
        # ----- INITIALIZE BOUNDING BOX ----- #
        min_bounds=np.array([0.05,-0.1,-0.1]) 
        max_bounds=np.array([0.3,0.1,0.1]) 
        self.box=get_box(min_bounds,max_bounds)
        self.vis.add_geometry(self.box)

  
        # ------- INITIALIZE IMAGING PARAMETERS ------- #

        self.minimum_thickness = 15

        if(self.machine == 'philips'):
            # CROP IMAGE
            start_x=59
            end_x=840
            start_y=10
            end_y=790
            self.box_crop=[start_x,end_x,start_y,end_y]

            # CROP TEXT AT TOP
            text_start_x=393
            text_end_x=440
            text_start_y=0
            text_end_y=15
            self.text_crop=[text_start_x,text_start_y,text_end_x,text_end_y]
            
            # DEFINE NEW PARAMETERS
            self.new_height=end_x-start_x
            self.new_width=end_y-start_y
            self.centre_x=int(self.new_height/2)
            self.centre_y=int(self.new_width/2)

            # CROP WIRE
            radius_wire=60
            self.wire_crop=make_wire_crop(self.new_height,self.new_width,self.centre_x,self.centre_y, radius_wire)
            # self.circle_crop=make_circle_crop(new_height,new_width,centre_x,centre_y)
            
            # GET RID OF CROSSHAIRS
            crosshair_width=5
            crosshair_height=2
            crosshair_vert_coordinates=[[380,62],[380,128],[380,193],[380,259],[380,324],[394,455],[394,520],[394,585],[394,651],[394,717]]
            crosshair_horiz_coordinates=[[61,395],[127,395],[192,395],[258,395],[323,395],[454,381],[519,381],[585,381],[650,381],[716,381]]
            self.crosshairs_crop=make_crosshairs_crop(self.new_height,self.new_width,crosshair_width,crosshair_height,crosshair_vert_coordinates,crosshair_horiz_coordinates)
            self.circle_crop=make_circle_crop(self.new_height,self.new_width,self.centre_x,self.centre_y)

        if(self.machine=='boston_scientific'):
            start_y=337  # note order of y and x flipped relative to philips
            end_y=1133
            start_x=62
            end_x=858
            self.box_crop=[start_x,end_x,start_y,end_y]

            text_start_y=990
            text_end_y=783
            text_start_x=1138
            text_end_x=855
            self.text_crop=[text_start_x,text_start_y,text_end_x,text_end_y]
            
            self.new_height=end_x-start_x
            self.new_width=end_y-start_y
            self.centre_x=int(self.new_height/2)
            self.centre_y=int(self.new_width/2)
            
            crosshair_width=0
            crosshair_height=0
            crosshair_vert_coordinates=[[self.centre_x, self.centre_y]]
            crosshair_horiz_coordinates=[[self.centre_x, self.centre_y]]
            self.crosshairs_crop=make_crosshairs_crop(self.new_height,self.new_width,crosshair_width,crosshair_height,crosshair_vert_coordinates,crosshair_horiz_coordinates)

            radius_wire=36
            self.wire_crop=make_wire_crop(self.new_height,self.new_width,self.centre_x,self.centre_y, radius_wire)
            self.circle_crop=make_circle_crop(self.new_height,self.new_width,self.centre_x,self.centre_y)

        
        # BUFFER INITIALIZATION
        self.buffer_size = 30
        self.branch_buffer_size = 10
        self.lstm_length = 5

        self.centroid_buffer = deque(maxlen=self.buffer_size)
        self.delta_buffer_x = deque(maxlen=self.buffer_size)
        self.delta_buffer_y = deque(maxlen=self.buffer_size)
        self.delta_buffer_z = deque(maxlen=self.buffer_size)
        self.position_buffer = deque(maxlen=self.buffer_size)
        self.grayscale_buffer = deque(maxlen=self.lstm_length)
        self.mask_1_buffer = deque(maxlen=self.branch_buffer_size)
        self.mask_2_buffer = deque(maxlen=self.branch_buffer_size)
        self.ecg_buffer = deque(maxlen=self.buffer_size)
        self.orifice_angles = deque(maxlen=5)
        self.position_tracker_buffer = deque(maxlen = 10)

        self.raylength_buffer = deque(maxlen=self.buffer_size)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.line, = self.ax.plot([], [], marker='o')
        self.ax.set_xlabel("Frame Index")
        self.ax.set_ylabel("Max Ray Length")
        self.ax.grid(True)
        # plt.show()



        # will be placed in initialization
        # self.mask_2A_buffer = deque(maxlen=self.buffer_size)
        # self.branch_pass = 0
        # # elements in mask_2A_buffer will be [branch_id, component_mask, orifice_angle]
        # # intialize branch id to 0 for mask 2A buffer e.g., [0, np.zeros_like(mask_2), np.nan]

        # self.branch_pass = self.branch_pass + 1 
        # self.mask_2B_buffer = deque(maxlen=self.buffer_size)
        # # initialize branch id to 1 for mask 2B buffer in a similar way

        # self.mask_2_buffers = deque(maxlen=2)
        # self.mask_2_buffers.append(self.mask_2A_buffer)
        # self.mask_2_buffers.append(self.mask_2B_buffer)

        H, W = 224, 224

        # helper to make N zero entries
        

        # initialize two buffers
        self.mask_2A_buffer = self.init_buffer(0, self.branch_buffer_size, (H, W), 0)
        self.mask_2B_buffer = self.init_buffer(1, self.branch_buffer_size, (H, W), 0)
        self.mask_2_buffers = deque([self.mask_2A_buffer, self.mask_2B_buffer], maxlen=2)



        self.transformed_centroids=[]

        self.processed_centreline = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.processed_centreline)

        # ------ GRIDLINES FOR FIRST RETURN --------#
        self.crop_radius=10
        self.gridlines=get_gridlines(self.new_height/2,self.new_width/2,no_points, self.crop_radius)

        # -------- INITIALIZE SDF INTEGRATORS ------#
        

        # # ESDF
        # # near lumen
        # self.voxelCarver=VoxelCarverPybind.VoxelCarverPybind()
        # self.esdf_mesh = o3d.geometry.TriangleMesh()
        # self.vis.add_geometry(self.esdf_mesh)

        # # far lumen
        # self.voxelCarver_2=VoxelCarverPybind.VoxelCarverPybind()
        # self.esdf_mesh_2 = o3d.geometry.TriangleMesh()
        # self.vis.add_geometry(self.esdf_mesh_2)

        # TSDF 
        # whole lumen
        # self.voxel_size = 0.003
        # self.voxel_size = 0.005

        # if(self.gating == 1):
        #     self.voxel_size = 0.0025

        # if(self.gating == 1):
        #     self.voxel_size = 0.005

        self.sdf_trunc = 3.5 * self.voxel_size
        # self.tsdf_volume = SimpleTsdfIntegrator(self.voxel_size, sdf_trunc)
        self.tsdf_volume = FastTsdfIntegrator(self.voxel_size, self.sdf_trunc)
        self.mesh=o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.mesh)


        # near lumen
        # self.tsdf_volume_near_lumen = SimpleTsdfIntegrator(self.voxel_size, sdf_trunc)
        self.tsdf_volume_near_lumen = FastTsdfIntegrator(self.voxel_size, self.sdf_trunc)
        self.mesh_near_lumen=o3d.geometry.TriangleMesh()

        if(self.dissection_mapping==1):
            self.vis.add_geometry(self.mesh_near_lumen)

        # far lumen
        # self.tsdf_volume_far_lumen = SimpleTsdfIntegrator(self.voxel_size, sdf_trunc)
        self.tsdf_volume_far_lumen = FastTsdfIntegrator(self.voxel_size, self.sdf_trunc)
        self.mesh_far_lumen=o3d.geometry.TriangleMesh()

        if(self.dissection_mapping==1):
            self.vis.add_geometry(self.mesh_far_lumen)



        if(self.dissection_mapping!=1):
            self.mesh_near_lumen_lineset = o3d.geometry.LineSet()

            if(self.figure_mapping==1):
                self.tsdf_surface_pc = o3d.geometry.PointCloud()
                self.vis.add_geometry(self.tsdf_surface_pc)
                self.simple_far_pc = o3d.geometry.PointCloud()
                self.vis.add_geometry(self.simple_far_pc)
                
            else:
                self.vis.add_geometry(self.mesh_near_lumen_lineset)
                self.vis.add_geometry(self.volumetric_far_point_cloud)

        

        

    
        self.ray_lengths_global = []
        self.branch_pass = 0
        self.branch_visible_previous = 0
        self.previous_mask = None
        
        # GUI specifics
    
        time.sleep(2)

        window_name = "3D Reconstruction"  # Replace with your window's title
        window_id = get_window_id(window_name)
        bring_window_to_front(window_id)

        window_name = "Simulated Endoscope View"  # Replace with your window's title
        window_id = get_window_id(window_name)
        bring_window_to_front(window_id)

        # Initialize the cv2 window
        # self.cv2_window_height = 224 * 2
        # self.cv2_window_width = self.cv2_window_height
        # self.cv2_window_x = self.cv2_window_height
        # self.cv2_window_y = 0
        # placeholder_image = np.zeros((self.cv2_window_height, self.cv2_window_width, 3), dtype=np.uint8)
 
        # window_name = "ivus"  # Replace with your window's title
        # window_id = get_window_id(window_name)
        # bring_window_to_front(window_id)

        window_name = "AortaScope"  # Replace with your window's title
        window_id = get_window_id(window_name)
        bring_window_to_front(window_id)

        
        # initialize endoscopic view on the target probe
        # setup subscribers and publishers
        # do this at the bottom
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback, queue_size=1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.click_sphere = o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.click_sphere)
       

        if(self.gating ==1):
            # self.lock = threading.Lock()  # A lock to ensure thread safety
            self.ecg_sub = rospy.Subscriber('ecg', Int32, self.ecg_callback, queue_size=1)
            
        self.rgb_image_pub = rospy.Publisher('/rgb_image', Image, queue_size=1)

        # self.pullback_pub = rospy.Publisher('/pullback', Int32, queue_size=1)
        # pullback = rospy.get_param('pullback', 0)
        # self.pullback_pub.publish(pullback)
        # print("pullback check", pullback)

        self.dest_frame = 'target1'


        # initialize view above the phantom
        view_control_1 = self.vis.get_view_control()

        view_control_1.set_up([0,1,0])

        view_control_1.set_front([0,0,-1])

        self.view_control_1 = view_control_1


        self.refine = 0

        # self.branch_sim_index = 866
        self.branch_sim_index = 1034*2

        self.orifice_center_spheres = o3d.geometry.TriangleMesh()

        # device deployment simulation
        self.evar_slide_sim = 0
        self.evar_loft_sim = 0
        self.tavr_sim = 0


        # block the callback of data once replay has started
        self.replay_data = 0


        # replaced rosparam polling with subscribers for efficiency
        rospy.Subscriber('/start_record', Bool, self.start_record_cb)
        rospy.Subscriber('/save_data', Bool, self.save_data_cb)
        rospy.Subscriber('/gate', Bool, self.gate_cb)
        rospy.Subscriber('/funsr_start', Bool, self.funsr_start_cb)
        rospy.Subscriber('/funsr_complete', Bool, self.funsr_complete_cb)
        rospy.Subscriber('/registration_started', Bool, self.registration_started_cb)
        rospy.Subscriber('/registration_done', Bool, self.registration_done_cb)
        rospy.Subscriber('/global_pause', Bool, self.global_pause_cb)
        rospy.Subscriber('/motion_capture', Bool, self.motion_capture_cb)
        rospy.Subscriber('/switch_probe', Bool, self.switch_probe_cb)
        rospy.Subscriber('/refine_started', Bool, self.refine_started_cb)
        rospy.Subscriber('/refine_done', Bool, self.refine_done_cb)
        rospy.Subscriber('/sim_device', Bool, self.sim_device_cb)
        rospy.Subscriber('/shutdown', Bool, self.shutdown_cb)
        rospy.Subscriber('/pullback', Int32, self.pullback_cb)
        rospy.Subscriber('/replay', Bool, self.replay_cb)
        rospy.Subscriber('/evar_click', Int32MultiArray, self.evar_click_cb)

        self.start_record = False
        self.save_data = False
        self.gate = False
        self.funsr_start = False
        self.funsr_complete = False
        self.registration_start = False
        self.reg_complete = False
        self.pause = False
        self.motion_capture = False
        self.switch_probe = False
        self.refine_start = False
        self.refine_complete = False
        self.sim_device = False
        self.shutdown = False
        self.pullback = 0  # Int32
        self.replay = False
        self.refine_done = False

        self.funsr_only = 0
        self.once = 0

        # for speed pruning
        transform_time = rospy.Time.now()
        self.previous_time_in_sec = transform_time.to_sec()
        self.previous_transform_ema = np.eye(4)
        self.smoothed_linear_speed = 0.0


        # FRAME GRABBER AND EM SIMULATOR
        if self.test_image == 1:
            rospy.loginfo("Test image mode active — starting mock publisher.")
            self.test_image_pub = rospy.Publisher('/usb_cam/image_raw', Image, queue_size=1)


            # optionally simulate position also so you can just launch aortascope, publish just once
            if self.test_transform == 1:

                self.static_broadcaster = tf2_ros.TransformBroadcaster()
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "ascension_origin"   # ref_frame
                t.child_frame_id = self.dest_frame       # dest_frame (e.g., 'target1')

                # Identity transform
                t.transform.translation.x = 0.0
                t.transform.translation.y = 0.0
                t.transform.translation.z = 0.0
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0

                self.static_broadcaster.sendTransform(t)


            # doing this instead of rospy spin
            while True:
              
                if self.test_transform == 1:
                    t.header.stamp = rospy.Time.now()
                    self.static_broadcaster.sendTransform(t)

                rgb_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                rgb_image = rgb_image.copy()
                center = (self.centre_x, self.centre_y)  # (x, y) coordinates of the circle center
                radius = int(self.centre_x/2)  # Radius of the circle
                color = (255, 255, 255)  # White color in BGR format
                thickness = 5  # -1 to fill the circle, >0 for border thickness
                cv2.circle(rgb_image, center, radius, color, thickness)
        

                msg = Image()
                msg.header.stamp = rospy.Time.now()
                msg.height = rgb_image.shape[0]
                msg.width = rgb_image.shape[1]
                msg.encoding = 'rgb8'
                msg.is_bigendian = False
                msg.step = rgb_image.shape[1] * 3
                msg.data = rgb_image.tobytes()

                self.test_image_pub.publish(msg)
                time.sleep(0.015)


    # for efficiency rather than rosparam polling
    def start_record_cb(self, msg):        self.start_record = msg.data
    def save_data_cb(self, msg):           self.save_data = msg.data
    def gate_cb(self, msg):                self.gate = msg.data
    def funsr_start_cb(self, msg):         self.funsr_start = msg.data
    def funsr_complete_cb(self, msg):      self.funsr_complete = msg.data
    def registration_started_cb(self, msg):self.registration_start = msg.data
    def registration_done_cb(self, msg):   self.reg_complete = msg.data
    def global_pause_cb(self, msg):        self.pause = msg.data
    def motion_capture_cb(self, msg):      self.motion_capture = msg.data
    def switch_probe_cb(self, msg):        self.switch_probe = msg.data
    def refine_started_cb(self, msg):      self.refine_start = msg.data
    def refine_done_cb(self, msg):         self.refine_complete = msg.data
    def sim_device_cb(self, msg):          self.sim_device = msg.data
    def shutdown_cb(self, msg):            self.shutdown = msg.data
    def pullback_cb(self, msg):            self.pullback = msg.data
    def replay_cb(self, msg):              self.replay = msg.data
    
    # EVAR POSITIONING AND FENESTRATION FUNCTIONS
    def evar_click_cb(self, msg): 

        # only bead geometry parameters should go here         
        
        # self.clicked_pixel = msg.data # change pixel value for sim device location in real time

        # self.fen_colors = [[1,1,0],[1,0,0],[0,0,1],[0,1,0]] # assuming 3 fens in head 4 in abd

        # self.fen_colors = [[0,0,1],[1,1,0],[0,1,0],[1,0,0]] # assuming 3 fens in head 4 in abd top to bottom C SMA LRA RRA

        # self.fen_colors = [[0,0,1],[1,1,0],[0,1,0],[1,0,0]] # assuming 3 fens in head 4 in abd top to bottom C SMA LRA RRA

        self.fen_colors = [[1,1,0],[1,0,0],[0,1,0],[0,0,1]] # Yellow, red, green, blue

        x, y = msg.data
        self.clicked_pixel = [x*(1/ (2.6*0.95)),y*(1/ (2.6*0.95*0.95))]
        self.clicked_pixel = np.asarray(self.clicked_pixel).astype(int)
        print(f"Received pixel: x={x}, y={y}")


        # self.fenestrate = rospy.get_param('fenestrate', 0)

        clicked_point = get_single_point_cloud_from_pixels(
                            self.clicked_pixel, self.scaling)

        pt = np.append(clicked_point, 1.0)              # [x, y, z, 1]
        pt_world = self.most_recent_extrinsic @ pt      # matrix multiply
        clicked_point = pt_world[:3]   

        temp_sphere = get_sphere_cloud([clicked_point], 0.0025, 20, [0,1,0])
        self.click_sphere.vertices = temp_sphere.vertices
        self.click_sphere.triangles = temp_sphere.triangles
        self.click_sphere.paint_uniform_color([0,1,0])
        self.click_sphere.compute_vertex_normals()
        self.vis.update_geometry(self.click_sphere)

        # for EVAR positioning
        closest_point, placeholder_1, placeholder_2 = find_closest_interpolated_values_on_centerline(clicked_point, self.aortic_centreline, self.GD_centreline)

        self.evar_loft_sim = 0 # not doing live updating

        if(self.fenestrate==0):
            
            # clicking a point 22 mm down
            first_bead_distance = -0.021
            absolute_position, radial_vector = convert_clicked_point_to_fen_center(clicked_point, closest_point, self.aortic_centreline, self.GD_centreline, first_bead_distance, self.evar_radius )

            closest_point, placeholder_1, placeholder_2 = find_closest_interpolated_values_on_centerline(absolute_position, self.aortic_centreline, self.GD_centreline)

            self.clicked_extrinsic = np.eye(4)
            self.clicked_extrinsic[:3,3] = absolute_position
            self.clicked_extrinsic[:3,1] = (absolute_position - closest_point ) / np.linalg.norm(absolute_position - closest_point )

            # align click point rotationally
            angle_deg = 60
            T = self.clicked_extrinsic.copy()
            theta = np.deg2rad(angle_deg)

            # Rotation in LOCAL Z axis
            Rz = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,              0,            1]
            ])

            # Apply to rotation block: local rotation
            self.clicked_extrinsic[:3,:3] = T[:3,:3] @ Rz

            # old code clicking the top
            # self.clicked_extrinsic = np.eye(4)
            # self.clicked_extrinsic[:3,3] = clicked_point
            # self.clicked_extrinsic[:3,1] = (clicked_point - closest_point ) / np.linalg.norm(clicked_point - closest_point )

            self.fenestrate=1
            print("self.fenestrate is now 1")

            

            
        elif(self.fenestrate==1):
            # punch out holes based on more clicked points
            bead_distance = 0.003  # change this as needed
            absolute_position, radial_vector = convert_clicked_point_to_fen_center(clicked_point, closest_point, self.aortic_centreline, self.GD_centreline, bead_distance, self.evar_radius )
            self.absolute_positions = np.vstack((self.absolute_positions, absolute_position))
            self.radial_vectors= np.vstack((self.radial_vectors, radial_vector))



        self.fen_distances = None # overriding any manually written fen locations
        self.fen_angles = None

        # self.evar_radius = 0.012
        # self.evar_length = 0.14
        
        print("absolute_position", absolute_position)

        current_evar  = predict_deploy(self.clicked_extrinsic, self.aortic_centreline,self.lofted_cylinder,self.strut_geometry,self.strut_distances,self.evar_length, self.centreline_transforms, self.GD_centreline,self.evar_radius, self.fen_distances, self.fen_angles, self.absolute_positions, self.radial_vectors, fenestration_mode='abs', fen_colors=self.fen_colors)
        # evar_struts_and_rings, evar_wireframe = predict_deploy_wireframe(self.clicked_extrinsic, self.aortic_centreline,self.lofted_cylinder_wireframe,self.strut_geometry,self.strut_distances,self.evar_length, self.centreline_transforms, self.GD_centreline,self.evar_radius, self.fen_distances, self.fen_angles, self.absolute_positions, self.radial_vectors, fenestration_mode='abs', fen_colors=self.fen_colors)
        evar_struts_and_rings, evar_wireframe = predict_deploy_wireframe(self.clicked_extrinsic, self.aortic_centreline,self.lofted_cylinder,self.strut_geometry,self.strut_distances,self.evar_length, self.centreline_transforms, self.GD_centreline,self.evar_radius, self.fen_distances, self.fen_angles, self.absolute_positions, self.radial_vectors, fenestration_mode='abs', fen_colors=self.fen_colors)



        self.evar_graft.vertices = current_evar.vertices
        self.evar_graft.triangles = current_evar.triangles
        self.evar_graft.vertex_colors = current_evar.vertex_colors
        self.evar_graft.compute_vertex_normals()
        
        self.vis.update_geometry(self.evar_graft)
        
        
        # self.vis2.update_geometry(self.evar_graft)

        

        if(self.show_evar_wireframe==1):

            self.evar_wireframe.points = evar_wireframe.points
            self.evar_wireframe.lines = evar_wireframe.lines
            self.evar_wireframe.colors = evar_wireframe.colors

            self.evar_struts_and_rings.vertices = evar_struts_and_rings.vertices
            self.evar_struts_and_rings.triangles = evar_struts_and_rings.triangles
            self.evar_struts_and_rings.vertex_colors = evar_struts_and_rings.vertex_colors
            self.evar_struts_and_rings.compute_vertex_normals()

            self.vis2.update_geometry(self.evar_struts_and_rings)
            self.vis2.update_geometry(self.evar_wireframe)

            self.vis.update_geometry(self.evar_struts_and_rings)
            self.vis.update_geometry(self.evar_wireframe)
        else:
            self.vis.update_geometry(self.evar_graft)

        o3d.io.write_triangle_mesh(self.write_folder  +  "/evar_graft.ply", self.evar_graft)
        o3d.io.write_triangle_mesh(self.write_folder  +  "/evar_struts_and_rings.ply", self.evar_struts_and_rings)
        o3d.io.write_line_set(self.write_folder  +  "/evar_wireframe.ply", self.evar_wireframe)
        np.save(self.write_folder  +  "/clicked_extrinsic.npy", self.clicked_extrinsic)
        np.save(self.write_folder  +  "/absolute_positions.npy", self.absolute_positions)
        np.save(self.write_folder  +  "/radial_vectors.npy", self.radial_vectors)
        


    def load_previous_fevar(self):

        path = os.path.join(self.write_folder, "clicked_extrinsic.npy")
        if os.path.exists(path):
            self.clicked_extrinsic = np.load(self.write_folder  +  "/clicked_extrinsic.npy")
            self.absolute_positions = np.load(self.write_folder  +  "/absolute_positions.npy")
            self.radial_vectors = np.load(self.write_folder  +  "/radial_vectors.npy")

        path = os.path.join(self.write_folder, "evar_graft.ply")

        # if os.path.exists(path):

        #     self.evar_graft = o3d.io.read_triangle_mesh(self.write_folder  +  "/evar_graft.ply")
        #     self.evar_graft.compute_vertex_normals()

        #     # self.vis.add_geometry(self.evar_graft)
        #     # self.vis2.add_geometry(self.evar_graft)

            


        #     if(self.show_evar_wireframe==1):

        #         print("loading and showing1!")
        #         self.evar_struts_and_rings = o3d.io.read_triangle_mesh(self.write_folder  +  "/evar_struts_and_rings.ply")
        #         self.evar_struts_and_rings.compute_vertex_normals()
        #         self.evar_wireframe = o3d.io.read_line_set(self.write_folder  +  "/evar_wireframe.ply")

                

        #         self.vis.add_geometry(self.evar_struts_and_rings)
        #         self.vis.add_geometry(self.evar_wireframe)

        #         self.vis2.add_geometry(self.evar_struts_and_rings)
        #         self.vis2.add_geometry(self.evar_wireframe)

        #         print("done!")

        #         # self.vis.update_geometry(self.evar_struts_and_rings)
        #         # self.vis.update_geometry(self.evar_wireframe)
                
        #     else:
        #         # self.vis.add_geometry(self.evar_graft)

        #         # self.vis.add_geometry(self.evar_graft)

        #         print("loading and showing2!")

        #         self.evar_struts_and_rings = o3d.io.read_triangle_mesh(self.write_folder  +  "/evar_struts_and_rings.ply")
        #         self.evar_struts_and_rings.compute_vertex_normals()
        #         self.evar_wireframe = o3d.io.read_line_set(self.write_folder  +  "/evar_wireframe.ply")

        #         self.vis.add_geometry(self.evar_struts_and_rings)
        #         self.vis.add_geometry(self.evar_wireframe)

        #         self.vis2.add_geometry(self.evar_struts_and_rings)
        #         self.vis2.add_geometry(self.evar_wireframe)

        #         print("done!")

        # temporarily overwrite
        self.fen_distances = None
        self.fen_angles = None

        self.fen_colors = [[0,0,1],[1,1,0],[0,1,0],[1,0,0]]
        
        current_evar  = predict_deploy(self.clicked_extrinsic, self.aortic_centreline,self.lofted_cylinder,self.strut_geometry,self.strut_distances,self.evar_length, self.centreline_transforms, self.GD_centreline,self.evar_radius, self.fen_distances, self.fen_angles, self.absolute_positions, self.radial_vectors, fenestration_mode='abs', fen_colors=self.fen_colors)
        # evar_struts_and_rings, evar_wireframe = predict_deploy_wireframe(self.clicked_extrinsic, self.aortic_centreline,self.lofted_cylinder_wireframe,self.strut_geometry,self.strut_distances,self.evar_length, self.centreline_transforms, self.GD_centreline,self.evar_radius, self.fen_distances, self.fen_angles, self.absolute_positions, self.radial_vectors, fenestration_mode='abs', fen_colors=self.fen_colors)
        evar_struts_and_rings, evar_wireframe = predict_deploy_wireframe(self.clicked_extrinsic, self.aortic_centreline,self.lofted_cylinder,self.strut_geometry,self.strut_distances,self.evar_length, self.centreline_transforms, self.GD_centreline,self.evar_radius, self.fen_distances, self.fen_angles, self.absolute_positions, self.radial_vectors, fenestration_mode='abs', fen_colors=self.fen_colors)

        self.evar_graft = current_evar
        self.evar_struts_and_rings = evar_struts_and_rings
        self.evar_wireframe = evar_wireframe

        self.vis.add_geometry(self.evar_struts_and_rings)
        self.vis.add_geometry(self.evar_wireframe)

        self.vis2.add_geometry(self.evar_struts_and_rings)
        self.vis2.add_geometry(self.evar_wireframe)

             
        self.vis.remove_geometry(self.volumetric_near_point_cloud)
        self.vis.remove_geometry(self.volumetric_far_point_cloud)
        # self.vis.remove_geometry(self.near_pc)
        # self.vis.remove_geometry(self.far_pc)
        self.fenestrate=1






    

    def initialize_deeplumen_model(self):
        
        if(self.deeplumen_on == 1):

            

            # DRN_inputs_3,DRN_outputs_3 = get_DRN_network()
            # model = tf.keras.Model(inputs=DRN_inputs_3, outputs=DRN_outputs_3)

            # model = keras.models.load_model(
            #     self.model_path,
            #     custom_objects={'BlurPool': BlurPool}
            # )

            model = build_mldr_drn(input_shape=(224,224,3), num_classes=3, base=64,
                        blocks_per_stage=(2,2,3,3,3), dilations=(1,2,4),
                        dropout=0.2, upsample_stride=8,
                        return_pyramid=False, name="MLDR_DRN_Large")

            # 2. Use TF-Keras (not standalone Keras 3)
            # 3. Load weights in H5 format, not .keras or SavedModel
            model.load_weights(self.model_path)

            model.summary()

            
            # sub branch segmentation
            # model.load_weights( self.model_path)  


        
            # model.summary()

            # this makes compilation 20x faster!!
            model = tf.function(model, jit_compile=True)
            # tf.config.optimizer.set_jit(True)

            self.model = model

        if(self.deeplumen_slim_on == 1):

            

            model = model7 = build_reduced_cnn_with_blurpool(input_shape=(224,224,3), num_classes=3)

            model.summary()

            
            # sub branch segmentation
            model.load_weights( self.model_path)  


        
            # model.summary()

            # this makes compilation 20x faster!!
            model = tf.function(model, jit_compile=True)

            self.model = model

        if(self.deeplumen_lstm_on==1):
            
            
            # 4 load savedmodel path
            model = TFSMLayer(self.model_path, call_endpoint="serving_default")


            
            # this makes compilation 20x faster!!
            model = tf.function(model, jit_compile=True)
            self.model = model

        if(self.deeplumen_valve_on == 1):


            DRN_inputs_3,DRN_outputs_3 = get_DRN_network()
            model_cusp = tf.keras.Model(inputs=DRN_inputs_3, outputs=DRN_outputs_3)
            model_cusp.load_weights( self.model_path_cusps)  
            model_cusp = tf.function(model_cusp, jit_compile=True)
            self.model_cusp = model_cusp

    def load_parameters(self, config_yaml):

        # GUI FUNCTIONS

        self.savepath = config_yaml['savepath']
        # is there an ECG signal?
        self.gating = config_yaml['gating']

        if(self.gating == 1):
            self.savepath = self.savepath + '/ungated'

        create_folder(self.savepath)

        # save data?
        self.record = config_yaml['record']

        self.record_poses =config_yaml['record_poses']
        
        self.save_replay_data = config_yaml['save_replay_data']

        # healthy first return mapping for reference
        self.tsdf_map = config_yaml['tsdf_map']

        self.voxel_size = config_yaml['voxel_size']

        self.conf_threshold = config_yaml['conf_threshold']

        self.deeplumen_on = config_yaml['deeplumen_on']

        self.deeplumen_slim_on = config_yaml['deeplumen_slim_on']

        self.deeplumen_lstm_on = config_yaml['deeplumen_lstm_on']

        self.deeplumen_valve_on = config_yaml['deeplumen_valve_on']

        self.model_path  = config_yaml['model_path']

        self.model_path_cusps  = config_yaml['model_path_cusps']

        self.endoanchor = config_yaml['endoanchor']

        self.vpC_map = config_yaml['vpC_map']

        self.bpC_map = config_yaml['bpC_map']

        if(self.dissection_mapping == 1):
            self.dissection_track = config_yaml['dissection_track']
            self.dissection_map = config_yaml['dissection_map']

        # dont load extend
        # self.extend = config_yaml['extend']
        

        self.tsdf_map = config_yaml['tsdf_map']

        self.orifice_center_map = config_yaml['orifice_center_map']

        self.registered_ct = config_yaml['registered_ct']

        self.registered_ct_dataset = config_yaml['registered_ct_dataset']

        self.live_deformation = config_yaml['live_deformation']

        self.cardiac_deformation = config_yaml['cardiac_deformation']

        self.animal = config_yaml['animal']

        self.guidewire = config_yaml['guidewire']

        self.steerable = config_yaml['steerable']

        self.double_display = config_yaml['double_display']

        self.test_image = config_yaml['test_image']

        self.test_transform = config_yaml['test_transform']

        self.machine = config_yaml['machine']

        self.vis_red_vessel = config_yaml['vis_red_vessel']

        self.figure_mapping = config_yaml['figure_mapping']

        self.load_evar_graft = config_yaml['load_evar_graft']

        self.show_evar_wireframe = config_yaml['show_evar_wireframe']

        # ---- LOAD ROS PARAMETERS ------ #
        param_names = ['/angle', '/translation','/scaling','/threshold','/no_points','/crop_index','/radial_offset','/oclock', '/constraint_radius','/pullback']

        # fetch these from yaml instead..
        self.default_values=load_default_values()

        # mapping specific parameters
        self.default_values['/no_points'] = 1000

        no_points=self.default_values['/no_points']
        # override previous threshold
        self.default_values['/threshold'] = 50

        self.default_values['/crop_index'] = 60

        self.default_values['/pullback'] = 0


        self.default_values['/constraint_radius'] = 0.006

        self.threshold = self.default_values['/threshold']
        self.no_points = self.default_values['/no_points']
        self.crop_index = self.default_values['/crop_index']
        self.scaling = self.default_values['/scaling']

   

        
        with open('/home/tdillon/mapping/src/calibration_parameters_ivus.yaml', 'r') as file:
            self.calib_yaml = yaml.safe_load(file)

        self.angle = self.calib_yaml['/angle']
        self.translation = self.calib_yaml['/translation']
        self.radial_offset = self.calib_yaml['/radial_offset']
        self.o_clock = self.calib_yaml['/oclock']



        # Set default parameter values if they don't exist
        for param_name in param_names:
            if not rospy.has_param(param_name):
                rospy.set_param(param_name, self.default_values.get(param_name, None))
                rospy.loginfo(f"Initialized {param_name} with default value: {self.default_values.get(param_name, None)}")

    # these should only load / save geometries, change the visualizer, and change pointcloudupdater properties

    def gate_data(self):


        # made redundant by motion capture code
        # perform reconstruction of bin 3 and 8 and then register

        # self.write_folder = rospy.get_param('dataset',0)
        
        # # change directory to gated folder
        # self.write_folder = self.write_folder + '/gated'
        # create_folder(self.write_folder )
        
        # print("new folder after gating is:", self.write_folder )

        # collect all data -> gating -> replay 3, 8 -> register to 3 -> initialize cardiac motion (register to 8) -> load registered scan (with cardiac deformation on)

        pass
        


    def replay_function(self):

        rospy.set_param('replay_done', 0)

        self.test_image = 0
        self.test_transform = 0

        self.write_folder = rospy.get_param('dataset',0)

    
        # load the calibration file for the REPLAYED dataset - note this means you will have that calibration file loaded when you 
        # go back to aortscope real time mapping

        try:
            with open(self.write_folder + '/calibration_parameters_ivus.yaml', 'r') as file:
                self.calib_yaml = yaml.safe_load(file)
        

            print("loaded calibration file for dataset:",  self.write_folder)

            self.angle = self.calib_yaml['/angle']
            self.translation = self.calib_yaml['/translation']
            self.radial_offset = self.calib_yaml['/radial_offset']
            self.o_clock = self.calib_yaml['/oclock']

            print("offset angle is!", self.angle)

        except:
            print("NO CALIBRATION FOUND")
            self.angle = 0
            self.translation=0
            self.radial_offset=0
            self.o_clock = 0

            self.test_transform=1
            


        # rospy.set_param('replay', 0)

        # clear view of any geometries
        try:
            self.vis.remove_geometry(self.processed_centreline)
        except:
            print("processed centreline")

        # try:
        #     self.vis.remove_geometry(self.us_frame)
        # except:
        #     print("no us frame present")

        try:
            self.vis.remove_geometry(self.mesh_near_lumen)
        except:
            print("no near lumen present")

        try:
            self.vis.remove_geometry(self.volumetric_near_point_cloud)

            
                # self.volumetric_near_point_cloud = o3d.geometry.PointCloud()
                # self.vis.add_geometry(self.volumetric_near_point_cloud)

            
        except:
            print("no volumetric near point cloud present")

        # want this
        # try:
        #     self.vis.remove_geometry(self.volumetric_far_point_cloud)
        # except:
        #     print("no volumetric far point cloud present")



        
        self.record=0
        self.extend=1
        self.write_folder = rospy.get_param('dataset',0)

        if(self.write_folder == 0):
            self.write_folder = self.prompt_for_folder()
            
        while(self.write_folder == None):
            self.write_folder = self.prompt_for_folder()

        # for post processing of tracking data
        folder_path = self.write_folder + '/pose_data'
        create_folder(folder_path)

        # load the images from the specified paths
        image_path = self.write_folder + '/grayscale_images/*.npy'
        sorted_images=sort_folder_string(image_path, "grayscale_image_")
        grayscale_images=load_numpy_data_from_folder(sorted_images)
        
        # if replaying just image data from OLD COMPUTER datasets
        # image_paths = sorted(glob.glob(os.path.join(self.write_folder, 'grayscale_images', '*.npy')))
        # grayscale_images = []
        # em_transforms = []
        # for image in image_paths:
        #     grayscale_images.append(np.load(image))
        #     em_transforms.append(np.eye(4))
        
        transform_path = self.write_folder + '/transform_data/*.npy'
        sorted_transforms=sort_folder_string(transform_path, "TW_EM_")
        em_transforms=load_transform_data_from_folder(sorted_transforms)

        print("pulling from folder",self.write_folder)
        print("number of loaded images:", len(grayscale_images))


        starting_index=0
        # starting_index=1200
        ending_index=len(em_transforms)-1

        # if(self.centre_data == 1):
        #     average_transform = get_transform_data_center(em_transforms)

        previous_time = 0

        for i in np.arange(starting_index,ending_index):
        # for i in np.arange(300,ending_index): # for troubleshooting ransac on k8_pva_tom_2
        # for i in np.arange(2350,ending_index): # for troubleshooting branch pass
        # for i in np.arange(2535,ending_index): # for troubleshooting branch pass

            

            grayscale_image=grayscale_images[i]
            TW_EM=em_transforms[i] 
            
            # print("TW_EM:", TW_EM)

            

            # TW_EM = np.eye(4)  # if you want to visualize only the image data

            # if(self.centre_data == 1):
            #     TW_EM =  average_transform @ em_transforms[i]


            # for eves dataset
            # time.sleep(0.1)
            # grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_BGR2GRAY) 

            # grayscale_image=preprocess_ivus_image(grayscale_image, pc_updater.box_crop, pc_updater.circle_crop, pc_updater.text_crop, pc_updater.crosshairs_crop)

            try:
                # pc_updater.image_callback(grayscale_image, TW_EM, i, model, dataset_name, gating, bin_number, pC_map, esdf_map, tsdf_map, killingfusion_save, dissection_parameterize, esdf_smoothing, certainty_coloring, vpC_map)
                self.append_image_transform_pair(TW_EM, grayscale_image) #TURN TRY EXCEPT BACK ON

            except:
                print("image skipped on replay!")

            # except KeyboardInterrupt:
            #     print("Ctrl+C detected, stopping visualizer...")
            #     self.stop()  #fake function that throws an exception

        try:
            self.vis.remove_geometry(self.processed_centreline)
        except:
            print("processed centreline")

        try:
            self.vis.remove_geometry(self.us_frame)
        except:
            print("no us frame present")

        try:
            self.vis.remove_geometry(self.mesh_near_lumen)
        except:
            print("no near lumen present")

        try:
            self.vis.remove_geometry(self.volumetric_near_point_cloud)
        except:
            print("no volumetric near point cloud present")

        try:
            self.vis.remove_geometry(self.volumetric_far_point_cloud)
        except:
            print("no volumetric far point cloud present")

        self.extend=0

        # SAVE THE REPLAYED DATA (NOT IMAGES OR TW_EM)
        if(self.save_replay_data == 1):
           
            self.record = 0

            # if(self.figure_mapping==1):
            #     intentional_fail # was about to save uniform colour point cloud for figure mapping
            
  
            o3d.io.write_point_cloud(self.write_folder +  "/volumetric_near_point_cloud.ply", self.volumetric_near_point_cloud)

            
            o3d.io.write_point_cloud(self.write_folder +  "/volumetric_far_point_cloud.ply", self.volumetric_far_point_cloud)

            o3d.io.write_point_cloud(self.write_folder +  "/orifice_center_pc.ply", self.orifice_center_point_cloud)

            o3d.io.write_triangle_mesh(self.write_folder  +  "/tsdf_mesh_near_lumen.ply", self.mesh_near_lumen)

            # for evaluation purposes
            o3d.io.write_point_cloud(self.write_folder +  "/boundary_near_point_cloud.ply", self.boundary_near_point_cloud)


            # do spline smoothing in post processing
            for transformed_centroid in self.transformed_centroids:
            
                # ----- ADD TO BUFFERS ------ #
                self.centroid_buffer.append(transformed_centroid[:3])
            
                if(len(self.centroid_buffer) >= self.buffer_size):
                    self.delta_buffer_x, self.delta_buffer_y, self.delta_buffer_z,closest_points, centrepoints = process_centreline_bspline(self.centroid_buffer, self.delta_buffer_x,self.delta_buffer_y,self.delta_buffer_z, self.buffer_size)
                    self.processed_centreline.points.extend(o3d.utility.Vector3dVector(centrepoints))
                    self.processed_centreline.paint_uniform_color([1,0,0])

            o3d.io.write_point_cloud(self.write_folder +  "/smoothed_bspline_centreline.ply", self.processed_centreline)

        if(self.gating==1):
            self.lightweight_reinitialize()


        rospy.set_param('replay_done', 1)
        # self.write_folder = rospy.set_param('dataset',0)

        print("saved! restarted aortascope")
        # self.vis.run()


    def lightweight_reinitialize(self):

        # self.orifice_center_pc.clear()
        self.orifice_center_point_cloud.clear()
        self.mesh_near_lumen_lineset.clear()

        if(self.figure_mapping==1):
            self.tsdf_surface_pc.clear()
            self.simple_far_pc.clear()
                
        
        try:
            self.vis.remove_geometry(self.tsdf_surface_pc)
        except:
            print("nothing to remove")
        try:
            self.vis.remove_geometry(self.simple_far_pc)
        except:
            print("nothing to remove")
        try:
            self.vis.remove_geometry(self.mesh_near_lumen_lineset)
        except:
            print("nothing to remove")
        try:
            self.vis.remove_geometry(self.volumetric_near_point_cloud)
        except:
            print("nothing to remove")
        self.volumetric_far_point_cloud.clear()
        try:
            self.vis.remove_geometry(self.volumetric_far_point_cloud)
        except:
            print("nothing to remove")
        self.mesh_near_lumen.clear()
        try:
            self.vis.remove_geometry(self.mesh_near_lumen)
        except:
            print("nothing to remove")
        self.boundary_near_point_cloud.clear()
        try:
            self.vis.remove_geometry(self.boundary_near_point_cloud)
        except:
            print("nothing to remove")

        now = rospy.Time.now()
        self.last_image_time = now



        # what number image callback are we on?
        self.image_call=1

        self.image_number = 1
        


        self.vis.poll_events()
        self.vis.update_renderer()

        self.vis2.poll_events()
        self.vis2.update_renderer()
    
        # ----- INITIALIZE RECONSTRUCTIONS ------ #
        self.near_point_cloud = o3d.geometry.PointCloud()
        self.far_point_cloud = o3d.geometry.PointCloud()
        self.dissection_flap_point_cloud = o3d.geometry.PointCloud()
        self.volumetric_far_point_cloud = o3d.geometry.PointCloud() 
        self.volumetric_near_point_cloud = o3d.geometry.PointCloud() 
        self.boundary_near_point_cloud = o3d.geometry.PointCloud() 
        self.point_cloud = o3d.geometry.PointCloud()
        self.orifice_center_point_cloud = o3d.geometry.PointCloud()
        
        

        self.vis.add_geometry(self.volumetric_near_point_cloud)
        # self.vis.add_geometry(self.volumetric_far_point_cloud)
        self.vis.add_geometry(self.boundary_near_point_cloud)
        # self.vis.add_geometry(self.point_cloud)
        self.vis.add_geometry(self.orifice_center_point_cloud)

        if(self.figure_mapping == 1):
            self.vis.add_geometry(self.tsdf_surface_pc)
            self.vis.add_geometry(self.simple_far_pc)



        # ----- INITIALIZE TRACKER FRAMES ------ #
        self.frame_scaling=0.025
        self.tracker_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.tracker_frame.scale(self.frame_scaling,center=[0,0,0])

        self.baseframe=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.baseframe.scale(self.frame_scaling,center=[0,0,0])

        self.us_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.us_frame.scale(self.frame_scaling,center=[0,0,0])

        self.tracker = o3d.geometry.TriangleMesh.create_cylinder(radius=0.001, height=0.01, resolution=40)
        self.tracker.compute_vertex_normals()
        self.tracker.paint_uniform_color([1,0.5,0])
        


        self.previous_transform=np.eye(4)
        self.previous_transform_1=np.eye(4)
        self.previous_transform_us=np.eye(4)
        self.previous_tracker_transform=np.eye(4)


        # print("added us frame")
        self.vis.add_geometry(self.us_frame)
        self.vis.add_geometry(self.tracker_frame)
        self.vis.add_geometry(self.baseframe)


        
        # BUFFER INITIALIZATION
        self.buffer_size = 30
        self.branch_buffer_size = 10

        self.centroid_buffer = deque(maxlen=self.buffer_size)
        self.delta_buffer_x = deque(maxlen=self.buffer_size)
        self.delta_buffer_y = deque(maxlen=self.buffer_size)
        self.delta_buffer_z = deque(maxlen=self.buffer_size)
        self.position_buffer = deque(maxlen=self.buffer_size)
        self.grayscale_buffer = deque(maxlen=self.lstm_length)
        self.mask_1_buffer = deque(maxlen=self.branch_buffer_size)
        self.mask_2_buffer = deque(maxlen=self.branch_buffer_size)
        self.ecg_buffer = deque(maxlen=self.buffer_size)
        self.orifice_angles = deque(maxlen=5)

        H, W = 224, 224

        # helper to make N zero entries
        

        # initialize two buffers
        self.mask_2A_buffer = self.init_buffer(0, self.branch_buffer_size, (H, W), 0)
        self.mask_2B_buffer = self.init_buffer(1, self.branch_buffer_size, (H, W), 0)
        self.mask_2_buffers = deque([self.mask_2A_buffer, self.mask_2B_buffer], maxlen=2)



        self.transformed_centroids=[]

        self.processed_centreline = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.processed_centreline)


        # -------- INITIALIZE SDF INTEGRATORS ------#
        

        # # ESDF
        # # near lumen
        # self.voxelCarver=VoxelCarverPybind.VoxelCarverPybind()
        # self.esdf_mesh = o3d.geometry.TriangleMesh()
        # self.vis.add_geometry(self.esdf_mesh)

        # # far lumen
        # self.voxelCarver_2=VoxelCarverPybind.VoxelCarverPybind()
        # self.esdf_mesh_2 = o3d.geometry.TriangleMesh()
        # self.vis.add_geometry(self.esdf_mesh_2)

        # TSDF 
        # whole lumen
        # self.voxel_size = 0.003
        # self.voxel_size = 0.005

        # if(self.gating == 1):
        #     self.voxel_size = 0.0025

        # if(self.gating == 1):
        #     self.voxel_size = 0.005

        self.tsdf_volume = FastTsdfIntegrator(self.voxel_size, self.sdf_trunc)
        self.mesh=o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.mesh)


        # near lumen
        # self.tsdf_volume_near_lumen = SimpleTsdfIntegrator(self.voxel_size, sdf_trunc)
        self.tsdf_volume_near_lumen = FastTsdfIntegrator(self.voxel_size, self.sdf_trunc)
        self.mesh_near_lumen=o3d.geometry.TriangleMesh()

        if(self.dissection_mapping==1):
            self.vis.add_geometry(self.mesh_near_lumen)

        # far lumen
        # self.tsdf_volume_far_lumen = SimpleTsdfIntegrator(self.voxel_size, sdf_trunc)
        self.tsdf_volume_far_lumen = FastTsdfIntegrator(self.voxel_size, self.sdf_trunc)
        self.mesh_far_lumen=o3d.geometry.TriangleMesh()

        if(self.dissection_mapping==1):
            self.vis.add_geometry(self.mesh_far_lumen)



        if(self.dissection_mapping!=1):
            self.mesh_near_lumen_lineset = o3d.geometry.LineSet()

            if(self.figure_mapping==1):
                self.tsdf_surface_pc = o3d.geometry.PointCloud()
                self.vis.add_geometry(self.tsdf_surface_pc)
                
            else:
                self.vis.add_geometry(self.mesh_near_lumen_lineset)

        

    
        self.ray_lengths_global = []
        self.branch_pass = 0
        self.branch_visible_previous = 0
        self.previous_mask = None
        

        # initialize view above the phantom
        view_control_1 = self.vis.get_view_control()
        view_control_1.set_up([0,1,0])
        view_control_1.set_front([0,0,-1])
        self.view_control_1 = view_control_1


        self.orifice_center_spheres = o3d.geometry.TriangleMesh()


        # block the callback of data once replay has started
        self.replay_data = 0

        self.funsr_only = 0
        self.once = 0

        # for speed pruning
        transform_time = rospy.Time.now()
        self.previous_time_in_sec = transform_time.to_sec()
        self.previous_transform_ema = np.eye(4)
        self.smoothed_linear_speed = 0.0


        

    def start_recording(self):

        self.write_folder = rospy.get_param('dataset',0)

        if not hasattr(self, 'write_folder'):
            self.write_folder = self.prompt_for_folder()
            
        while(self.write_folder ==None or self.write_folder == 0):
            self.write_folder = self.prompt_for_folder()

        new_path = self.write_folder + '/calibration_parameters_ivus.yaml'
        with open(new_path, 'w') as file:
            yaml.dump(self.calib_yaml, file)

        # self.deeplumen_on = 1

        # turn on extend
        self.record=1
        self.extend=1

        # start appending to image batch
        
        self.image_batch=[]
        self.tw_em_batch=[]

        self.image_tags=[]
        self.starting_index = 1
        
        self.ecg_times=[]
        self.ecg_signal=[]
        self.image_times=[]

        print("creating folders!")

        

        folder_path = self.write_folder + '/grayscale_images'
        create_folder(folder_path)

        folder_path = self.write_folder + '/transform_data'
        create_folder(folder_path)

        if(self.gating==1):

            folder_path = self.write_folder + '/ecg_signal'
            create_folder(folder_path)

        

        print("creating folders!")

        # for post processing of tracking data
        folder_path = self.write_folder + '/pose_data'
        create_folder(folder_path)

        # this is important!
        self.branch_pass = 0


        # initialize view
        view_control_1 = self.vis.get_view_control()

        view_control_1.set_up([0,-1,0])

        view_control_1.set_front([0,0,-1])

        view_control_1.set_zoom(0.5)

        self.view_control_1 = view_control_1

    

    def funsr_started(self):

        rospy.set_param('funsr_started', 0)
        self.write_folder = rospy.get_param('dataset', 0)
        self.deeplumen_on = 0
        self.deeplumen_slim_on = 0
        self.deeplumen_lstm_on = 0
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        del self.model
        # cuda.select_device(0)
        # cuda.close()
        gc.collect()
        
        print("cleared session!")
        funsr_done = 0
        while(funsr_done ==0):
            funsr_done = rospy.get_param('funsr_done', 0)
            time.sleep(1)

    def registration_started(self):

        # rospy.set_param('registration_started', 0)
        print("registration started!")
        self.write_folder = rospy.get_param('dataset', 0)
        self.deeplumen_on = 0
        self.deeplumen_slim_on = 0
        self.deeplumen_lstm_on = 0
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        del self.model
        # cuda.select_device(0)
        # cuda.close()
        gc.collect()
        
        print("cleared session!")
       
        while(self.reg_complete ==0):
            # registration_done = rospy.get_param('registration_done', 0)
            time.sleep(1)

    def motion_capture_pause(self):

        while(self.motion_capture ==True):
            time.sleep(1)

    def init_buffer(self,branch_id, buffer_size, shape, branch_pass_trigger):
            H, W = shape
            zero_mask = np.zeros((H, W), dtype=np.uint8)
            return deque([[branch_id, zero_mask.copy(), np.nan, branch_pass_trigger] for _ in range(buffer_size)],
                        maxlen=buffer_size)

    def prompt_for_folder(self):
        # Create a Tkinter root window (it won't show)
        root2 = tk.Tk()
        root2.withdraw()  # Hide the root window
        
        # Open a folder selection dialog
        initial_dir = '/home/tdillon/datasets'
        folder_selected = filedialog.askdirectory(title="Select a Folder to Save the Data", initialdir=initial_dir)
        
        # If the user selects a folder (i.e., the folder path is not empty)
        if folder_selected:
            print(f"Using folder: {folder_selected}")
            return folder_selected
        else:
            print("No folder selected")
            return None
        

    def save_image_and_transform_data(self):

    

        # turn extend off!
        self.extend = 0
        self.record = 0
        

        # time.sleep(1)

        print("final save!")

        # is self.write_folder the same as these long paths:
        # dataset_directory + pc_updater.dataset_name + bin_name 

        
       
        rospy.set_param('dataset', self.write_folder)   

        # if(self.figure_mapping==1):
        #     intentional_fail # was about to save uniform colour point cloud for figure mapping

        

        o3d.io.write_point_cloud(self.write_folder +  "/volumetric_near_point_cloud.ply", pc_updater.volumetric_near_point_cloud)

        o3d.io.write_point_cloud(self.write_folder +  "/volumetric_far_point_cloud.ply", pc_updater.volumetric_far_point_cloud)

        o3d.io.write_point_cloud(self.write_folder +  "/orifice_center_pc.ply", pc_updater.orifice_center_point_cloud)

        o3d.io.write_triangle_mesh(self.write_folder  +  "/tsdf_mesh_near_lumen.ply", pc_updater.mesh_near_lumen)

        # for evaluation purposes
        o3d.io.write_point_cloud(self.write_folder +  "/boundary_near_point_cloud.ply", pc_updater.boundary_near_point_cloud)


        # do spline smoothing in post processing
        for transformed_centroid in self.transformed_centroids:
        
            # ----- ADD TO BUFFERS ------ #
            self.centroid_buffer.append(transformed_centroid[:3])
        
            if(len(self.centroid_buffer) >= self.buffer_size):
                self.delta_buffer_x, self.delta_buffer_y, self.delta_buffer_z,closest_points, centrepoints = process_centreline_bspline(self.centroid_buffer, self.delta_buffer_x,self.delta_buffer_y,self.delta_buffer_z, self.buffer_size)
                self.processed_centreline.points.extend(o3d.utility.Vector3dVector(centrepoints))
                self.processed_centreline.paint_uniform_color([1,0,0])

        o3d.io.write_point_cloud(self.write_folder +  "/smoothed_bspline_centreline.ply", pc_updater.processed_centreline)


        save_frequency = 1 #doesnt work with ecg etc

        # dataset saving
        # if(self.record==1):

    

        print("started saving")
        # Iterate through the image and TW_EM batches simultaneously
        for (grayscale_image, TW_EM, image_tag) in zip(self.image_batch, self.tw_em_batch, self.image_tags):
            # if i % save_frequency == 0:
            # Save the image
            image_filename = f'{self.write_folder}/grayscale_images/grayscale_image_{self.starting_index + image_tag -1}.npy'
            with open(image_filename, 'wb') as f:
                np.save(f, grayscale_image)
            

            # Save the TW_EM data
            tw_em_filename = f'{self.write_folder}/transform_data/TW_EM_{self.starting_index + image_tag -1}.npy'
            with open(tw_em_filename, 'wb') as f:
                TW_EM = np.array(TW_EM, dtype=np.float64).reshape(4, 4)
                np.save(f, TW_EM)


        # this does not get saved intermittently, only at the end
        if(self.gating ==1):

            image_times = np.asarray(self.image_times) 

            image_times_filename = f'{self.write_folder}/ecg_signal/image_times.npy'
            with open(image_times_filename, 'wb') as f:
                np.save(f, image_times)

            if(self.gating == 1):
                ecg_times = np.asarray(self.ecg_times)
                ecg_times_filename = f'{self.write_folder}/ecg_signal/ecg_times.npy'
                with open(ecg_times_filename, 'wb') as f:
                    np.save(f, ecg_times)

                ecg_signal = np.asarray(self.ecg_signal)

                ecg_signal_filename = f'{self.write_folder}/ecg_signal/ecg_signal.npy'
                with open(ecg_signal_filename, 'wb') as f:
                    np.save(f, ecg_signal)

        print("finished saving images, transform and ecg data (if present)!")

        # turn off the pullback device
        rospy.set_param('pullback', 0)
        print("pullback device stopped")

        return

    def quick_save(self):

        


        start_save = time.time()

        # TEMP BOSTON SCIENTIFIC!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        

        for (grayscale_image, TW_EM, image_tag) in zip(self.image_batch, self.tw_em_batch, self.image_tags):

        # for (grayscale_image, image_tag) in zip(self.image_batch, self.image_tags):

            # if i % save_frequency == 0:
            # Save the image
            # image_filename = f'{self.write_folder}/grayscale_images/grayscale_image_{self.image_call + i}.npy'
            image_filename = f'{self.write_folder}/grayscale_images/grayscale_image_{self.starting_index + image_tag -1}.npy'
            with open(image_filename, 'wb') as f:
                np.save(f, grayscale_image)
            

            # Save the TW_EM data
            
            tw_em_filename = f'{self.write_folder}/transform_data/TW_EM_{self.starting_index + image_tag -1}.npy'
            with open(tw_em_filename, 'wb') as f:
                TW_EM = np.array(TW_EM, dtype=np.float64).reshape(4, 4)
                np.save(f, TW_EM)



        finish_save = time.time()
        difference_time = finish_save - start_save
        
        print("time to save image batch", difference_time)
        

        self.starting_index = self.starting_index + image_tag



    def save_pose_data(self):

    

   
        # Iterate through the image and TW_EM batches simultaneously
        i=0
        for (TW_EM, image_tag) in zip(self.pose_batch, self.image_tags):
       
        

            # Save the TW_EM data
            tw_em_filename = f'{self.write_folder}/pose_data/TW_EM_{self.starting_index + image_tag -1}.npy'
            with open(tw_em_filename, 'wb') as f:
                np.save(f, TW_EM)

            i=i+1

        self.starting_index = self.starting_index + image_tag




        return


    def save_refine_data(self):

    

        # turn extend off!
        self.extend = 0
  
        
        o3d.io.write_point_cloud(self.write_folder +  "/orifice_center_pc_refine.ply", pc_updater.orifice_center_point_cloud)
        o3d.io.write_triangle_mesh(self.write_folder  +  "/tsdf_mesh_refine.ply", pc_updater.mesh_near_lumen)


        # need to RESET TRANSFORMED CENTROIDS

        # do spline smoothing in post processing
        for transformed_centroid in self.transformed_centroids:
        
            # ----- ADD TO BUFFERS ------ #
            self.centroid_buffer.append(transformed_centroid[:3])
        
            if(len(self.centroid_buffer) >= self.buffer_size):
                self.delta_buffer_x, self.delta_buffer_y, self.delta_buffer_z,closest_points, centrepoints = process_centreline_bspline(self.centroid_buffer, self.delta_buffer_x,self.delta_buffer_y,self.delta_buffer_z, self.buffer_size)
                self.processed_centreline.points.extend(o3d.utility.Vector3dVector(centrepoints))
                self.processed_centreline.paint_uniform_color([1,0,0])

        o3d.io.write_point_cloud(self.write_folder +  "/smoothed_bspline_refine.ply", pc_updater.processed_centreline)

        print("finished saving , transform and ecg data (if present)!")

        
        return
        

    # this is now just a function for making do with a rigid TSDF lumen mesh when no CT available
    def funsr_done(self):

        
        self.vpC_map = 0
        self.vis_red_vessel= 1
        self.steerable = 1
        self.guidewire = 0

        self.pose_batch = []
        self.image_batch = []
        self.image_tags = []
        self.starting_index = 0

        self.write_folder = rospy.get_param('dataset', 0) 

        

        if(self.write_folder ==0):
            self.write_folder = self.prompt_for_folder()
            
        while(self.write_folder ==None):
            self.write_folder = self.prompt_for_folder()

        self.centerline_pc = o3d.io.read_point_cloud(self.write_folder + '/smoothed_bspline_centreline.ply')

        self.extend = 0
        self.record = 0

        # usually means doing an animal study so record!!
        self.record_poses = 1

        with open('/home/tdillon/mapping/src/calibration_parameters_guidewire.yaml', 'r') as file:
            calib_yaml_gw = yaml.safe_load(file)

        translation_gw = calib_yaml_gw['/translation']
        radial_offset_gw = calib_yaml_gw['/radial_offset']
        oclock_gw = calib_yaml_gw['/oclock']
        TEM_GW = [[1,0,0,translation_gw],[0,1,0,radial_offset_gw*np.cos(oclock_gw)],[0,0,1,radial_offset_gw*np.sin(oclock_gw)],[0, 0, 0, 1]]
        self.TEM_GW = np.asarray(TEM_GW)

        self.deeplumen_on = 0
        self.deeplumen_slim_on = 0
        self.deeplumen_lstm_on = 0

          
        # superimpose the completed surface geometry

        ivus_funsr_mesh = o3d.io.read_triangle_mesh(self.write_folder + '/tsdf_mesh_near_lumen.ply')
        # ivus_funsr_lineset = create_wireframe_lineset_from_mesh(ivus_funsr_mesh)
        # ivus_funsr_lineset.paint_uniform_color([0,0,0])
        # self.vis.add_geometry(ivus_funsr_lineset)
        # ivus_funsr_mesh.paint_uniform_color([1,0,0])
        # self.vis2.add_geometry(ivus_funsr_lineset)
        # self.vis.add_geometry(ivus_funsr_lineset)
        # ivus_funsr_mesh.paint_uniform_color([1,0,0])
        # self.vis2.add_geometry(ivus_funsr_lineset)

        # superimpose a mesh found from poisson reconstruction
        
        print("getting poisson mesh reconstruction ...")

        ivus_funsr_mesh.compute_vertex_normals()
        pcd = ivus_funsr_mesh.sample_points_uniformly(number_of_points=100000)
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        #     radius=0.001,  # adjust radius based on scale (e.g. 1 cm neighborhood)
        #     max_nn=10     # max number of neighbors
        # ))
        ivus_funsr_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)
        ivus_funsr_mesh.compute_vertex_normals()
        # voxel_size = max(ivus_funsr_mesh.get_max_bound() - ivus_funsr_mesh.get_min_bound()) / 120
        # ivus_funsr_mesh = ivus_funsr_mesh.simplify_vertex_clustering(
        #     voxel_size=voxel_size,
        #     contraction=o3d.geometry.SimplificationContraction.Average)
        ivus_funsr_mesh.paint_uniform_color([1,0,0])

       
        triangle_clusters, cluster_n_triangles, _ = (ivus_funsr_mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        ivus_funsr_mesh.remove_triangles_by_mask(triangles_to_remove)

        funsr_lineset = create_wireframe_lineset_from_mesh(ivus_funsr_mesh)
        funsr_lineset.paint_uniform_color([0,0,0])

        # self.vis.add_geometry(ivus_funsr_mesh)
        # self.vis2.add_geometry(ivus_funsr_mesh)



        # flip_mesh_orientation(ivus_funsr_mesh) 

        if(self.vis_red_vessel!=1):
            self.vis.add_geometry(funsr_lineset)

        else:
            self.vis.add_geometry(ivus_funsr_mesh)
        
        self.vis.get_render_option().mesh_show_back_face = False



        self.vis2.add_geometry(ivus_funsr_mesh)

        self.vis2.get_render_option().mesh_show_back_face = False


        self.vis.remove_geometry(self.us_frame)
        # self.catheter_radius = 0.0025
        self.catheter_radius = 0.0015
        # self.bending_segment_color = [0.2, 0.2, 1.0]

        

        


        

        # adding raw volumetric far point cloud data
        self.far_pc = o3d.io.read_point_cloud(self.write_folder + '/volumetric_far_point_cloud.ply')
        if(self.refine==1):
            self.far_pc = o3d.io.read_point_cloud(self.write_folder + '/volumetric_far_point_cloud_refine.ply')
        self.far_pc.paint_uniform_color([1,0,0])
        self.vis.add_geometry(self.far_pc)
        self.vis2.add_geometry(self.far_pc)

        copy_far_pc = copy.deepcopy(self.far_pc)
        self.far_pc_points = np.asarray(copy_far_pc.points)


        self.vis2.add_geometry(self.tracker)




        # Get current camera parameters
        self.view_control_1 = self.vis.get_view_control()

        self.view_control_1.set_up([0,-1,0])

        self.view_control_1.set_front([0,0,-1])

        
        
        self.view_control_1.set_zoom(0.25)

        # roll 180 degrees
        # view_control_1.rotate(180, 0) 



        self.vis.poll_events()
        self.vis.update_renderer()

        self.vis2.poll_events()
        self.vis2.update_renderer()

        self.funsr_only = 1
        points = np.asarray(ivus_funsr_mesh.vertices)  # Convert point cloud to numpy array
        self.funsr_centroid = np.mean(points, axis=0) 

        self.guidewire = 0

        # self.steerable = 1

        with open('/home/tdillon/mapping/src/calibration_parameters_guidewire.yaml', 'r') as file:
            calib_yaml_gw = yaml.safe_load(file)

        translation_gw = calib_yaml_gw['/translation']
        radial_offset_gw = calib_yaml_gw['/radial_offset']
        oclock_gw = calib_yaml_gw['/oclock']
        
        TEM_GW = [[1,0,0,translation_gw],[0,1,0,radial_offset_gw*np.cos(oclock_gw)],[0,0,1,radial_offset_gw*np.sin(oclock_gw)],[0, 0, 0, 1]]
        self.TEM_GW = np.asarray(TEM_GW)

        # ---- RENDER RINGS INSTEAD OF SPHERES (just comment this if prefer spheres) --- #
        # add ivus centroid clusters
        orifice_pc = o3d.io.read_point_cloud(self.write_folder + "/orifice_center_pc.ply")
        # ivus_centroids = np.load(self.write_folder + "/ivus_centroids.npy")
        ivus_centroids = np.asarray(orifice_pc.points)
        cluster_centroids = visualize_ivus_tsdf_clusters(orifice_pc, ivus_centroids)
        self.ct_centroids = cluster_centroids
        self.ct_centroids = np.vstack(self.ct_centroids)
        

        self.ct_spheres = o3d.geometry.TriangleMesh()
        self.torus_origins = []
        self.torus_normals = []
        self.minor_radius = 0.0005
        self.major_radius = 0.00325  # shrink it a little bit so it doesn't overlap mesh
        self.aortic_centreline = np.asarray(self.centerline_pc.points)

        colors = [
            [0.5, 0.0, 0.0],  # Maroon
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0]   # Yellow
        ]

        num_colors_needed = np.shape(self.ct_centroids)[0]
        extended_colors = (colors * (num_colors_needed // 4)) + colors[:num_colors_needed % 4]
        colors = extended_colors[:num_colors_needed]
            
        self.ct_centroids = np.asarray(self.ct_centroids)
        for centroid, color in zip(self.ct_centroids,colors):
            
            
            origin = centroid
            closest_index = np.argmin(np.linalg.norm(self.aortic_centreline-origin,axis=1))
            closest_point = self.aortic_centreline[closest_index,:]
            normal = origin-closest_point
            normal = normal/np.linalg.norm(normal)

            # torus = create_torus(centroid, normal, major_radius, minor_radius, 30)
            torus = create_torus(origin, normal, self.major_radius, self.minor_radius, 30)
            torus.paint_uniform_color(color)
            self.ct_spheres = self.ct_spheres + torus # replace the ct_spheres

        self.vis.add_geometry(self.ct_spheres)
        self.vis2.add_geometry(self.ct_spheres)



     


        

      

    def load_registetered_ct(self):

        no_reg=0

        self.once = 0

        refine=0
       

        print("loading registration lineset")
        self.registered_ct_lineset = o3d.io.read_line_set(self.write_folder+ '' + '/final_registration.ply')
        self.registered_ct_mesh = o3d.io.read_triangle_mesh(self.write_folder + '/final_registration_mesh.ply')

        # self.registered_ct_mesh = o3d.io.read_triangle_mesh(self.write_folder + '/final_registration_mesh_final.ply')

        # self.registered_ct_lineset = o3d.io.read_line_set(self.write_folder+ '' + '/backup/final_registration.ply')
        # self.registered_ct_mesh = o3d.io.read_triangle_mesh(self.write_folder + '/backup/final_registration_mesh.ply')
      
        if(self.refine==1):
            self.registered_ct_lineset = o3d.io.read_line_set(self.write_folder+ '' + '/final_registration_refine.ply')
            print("loading registration mesh")
            self.registered_ct_mesh = o3d.io.read_triangle_mesh(self.write_folder + '/final_registration_mesh_refine.ply')

        # TSDF LOAD MODULES
        # if self.registered_ct_mesh is None or len(self.registered_ct_mesh.vertices) == 0 or len(self.registered_ct_mesh.triangles) == 0:

        #     self.ct_centroids = np.load(self.write_folder + '/ivus_centroids.npy')
        #     self.ct_spheres = get_sphere_cloud(self.ct_centroids, 0.004, 20, [0,1,0])

        #     self.registered_ct_lineset.paint_uniform_color([0,0,0])

        #     self.registered_ct_mesh = o3d.io.read_triangle_mesh(self.write_folder + '/tsdf_mesh_near_lumen.ply')
        #     self.registered_ct_mesh_2 = copy.deepcopy(self.registered_ct_mesh)
        #     self.registered_ct_mesh_2.compute_vertex_normals()

        #     # DELETED FOR FEVAR
        #     self.vis2.add_geometry(self.ct_spheres)
        #     self.registered_ct_mesh_2.paint_uniform_color([1,0,0])
        #     self.vis2.add_geometry(self.registered_ct_mesh_2)
        #     self.vis2.add_geometry(self.tracker)

            
        #     # self.vis.remove_geometry(self.catheter)

        #     self.registered_ct_lineset = create_wireframe_lineset_from_mesh(self.registered_ct_mesh)
        #     self.vis.add_geometry(self.registered_ct_lineset)

        #     # DELETED FOR FEVAR
        #     # self.vis.add_geometry(self.ct_spheres)

        #     print("LOADING TSDF MESH ONLY!!")

    
        #     # Get current camera parameters
        #     view_control_1 = self.vis.get_view_control()

        #     view_control_1.set_up([0,-1,0])

        #     view_control_1.set_front([0,0,-1])



        #     view_control_1.set_zoom(0.25)

        #     # roll 180 degrees
        #     # view_control_1.rotate(180, 0) 


            
        #     return

        ct_centroid_pc = o3d.io.read_point_cloud(self.write_folder + '/side_branch_centrelines.ply')
        # ct_centroid_pc = o3d.io.read_point_cloud(self.write_folder + '/side_branch_centrelines_final.ply')
        if(self.refine==1):
            ct_centroid_pc = o3d.io.read_point_cloud(self.write_folder + '/side_branch_centrelines_refine.ply')

        self.centerline_pc = o3d.io.read_point_cloud(self.write_folder + '/centerline_pc.ply')
        if(self.refine==1):
            self.centerline_pc = o3d.io.read_point_cloud(self.write_folder + '/centerline_pc_refine.ply')

        # o3d.visualization.draw_geometries([self.centerline_pc, self.registered_ct_lineset])


        # NOte that IVUS centroids have been used here!!!
        # self.ct_centroids = np.load(self.write_folder + '/ct_centroids.npy')
        self.ct_centroids = np.load(self.write_folder + '/ivus_centroids.npy')

        side_branch_centrelines_indices = np.load(self.write_folder + '/side_branch_centrelines_indices.npy')

        # ct_centroid_pc_points = np.asarray(ct_centroid_pc.points)
 
        # side_branch_points_grouped = []
        # for check_index in np.unique(side_branch_centrelines_indices):
        #     relevant_args = np.argwhere(side_branch_centrelines_indices == check_index)[:,0]
        #     relevant_side_branch_centrelines_pc_points = ct_centroid_pc_points[relevant_args,:]
        #     side_branch_points_grouped.append(relevant_side_branch_centrelines_pc_points)

        # self.ct_centroids = np.vstack(side_branch_points_grouped)


        # only load CORRESPONDING ivus_centroids to remove false positives
        load_corres = 1
        print('before crop', self.ct_centroids)
        if(load_corres ==1):
            corres_original = np.load(self.write_folder + '/corres_original.npy')
            corres_original_ivus = corres_original[:,1]
            self.ct_centroids = self.ct_centroids[corres_original_ivus,:]
            print('after crop', self.ct_centroids)
        


        self.registered_ct_mesh.remove_unreferenced_vertices()

        self.registered_ct_mesh_2 = copy.deepcopy(self.registered_ct_mesh)

        avg_edge_length = 999
        desired_edge_length = 0.002

        print("started subdividing!")
        while avg_edge_length > desired_edge_length:
            self.registered_ct_mesh_2 = self.registered_ct_mesh_2.subdivide_loop(number_of_iterations=1)
            
            vertices = np.asarray(self.registered_ct_mesh_2.vertices)
            triangles = np.asarray(self.registered_ct_mesh_2.triangles)
            v0 = vertices[triangles[:, 0]]
            v1 = vertices[triangles[:, 1]]
            v2 = vertices[triangles[:, 2]]
            e0 = np.linalg.norm(v0 - v1, axis=1)
            e1 = np.linalg.norm(v1 - v2, axis=1)
            e2 = np.linalg.norm(v2 - v0, axis=1)
            avg_edge_length = np.mean(np.concatenate([e0, e1, e2]))

        print("subdivision complete, computing coarse to fine mapping!")
        
        self.knn_idxs, self.knn_weights = precompute_knn_mapping(self.registered_ct_mesh, self.registered_ct_mesh_2, k=3)
        
        print("finished knn compute")
        self.coarse_template_vertices = copy.deepcopy(np.asarray(self.registered_ct_mesh.vertices))
        self.fine_template_vertices = copy.deepcopy(np.asarray(self.registered_ct_mesh_2.vertices))
        self.adjacency_matrix = build_adjacency_matrix(self.registered_ct_mesh_2)
        # visualize_knn_mapping(self.registered_ct_mesh_2, self.registered_ct_mesh, self.knn_idxs, sample_indices=[0, 50, 100])
        print("computed coarse to fine mesh node mapping")


        self.constraint_locations = np.asarray(ct_centroid_pc.points)
        
       
        

        colors = [
            [0.5, 0.0, 0.0],  # Maroon
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0]   # Yellow
        ]


        

        print("number of ct centroids are", self.ct_centroids.shape[0])
        num_colors_needed = self.ct_centroids.shape[0]
        extended_colors = (colors * (num_colors_needed // 4)) + colors[:num_colors_needed % 4]
        colors = extended_colors[:num_colors_needed]

        self.ct_spheres = o3d.geometry.TriangleMesh()
        for centroid, color in zip(self.ct_centroids,colors):
            new_sphere = get_sphere_cloud([centroid], 0.00275, 20, [0,1,0])
            new_sphere.paint_uniform_color(color)
            self.ct_spheres = self.ct_spheres + new_sphere


        # ---- RENDER RINGS INSTEAD OF SPHERES (just comment this if prefer spheres) --- #
        simple_normal=1
        self.ct_spheres = o3d.geometry.TriangleMesh()
        self.torus_origins = []
        self.torus_normals = []
        self.minor_radius = 0.0005
        self.major_radius = 0.00325  # shrink it a little bit so it doesn't overlap mesh
        self.aortic_centreline = np.asarray(self.centerline_pc.points)

        if(simple_normal==0):
            side_branch_centrelines_indices = np.load(self.write_folder + '/side_branch_centrelines_indices.npy')
 
            side_branch_points_grouped = []
            for check_index in np.unique(side_branch_centrelines_indices):
                relevant_args = np.argwhere(side_branch_centrelines_indices == check_index)[:,0]
                relevant_side_branch_centrelines_pc_points = self.constraint_locations[relevant_args,:]
                side_branch_points_grouped.append(relevant_side_branch_centrelines_pc_points)

            for centroid, color, relevant_side_branch_centrelines_pc_points  in zip(self.ct_centroids,colors, side_branch_points_grouped):
                
                # get centroid_pc centreline for each centroid
                # major_radius = maximal_inscribed_radius(mesh = self.registered_ct_mesh, point = centroid)  # shrink it a little bit so it doesn't overlap mesh
                
                origin = relevant_side_branch_centrelines_pc_points[2,:]
                normal = relevant_side_branch_centrelines_pc_points[ 10, :] - origin
                normal = normal / np.linalg.norm(normal)

                # torus = create_torus(centroid, normal, major_radius, minor_radius, 30)
                torus = create_torus(origin, normal, self.major_radius, self.minor_radius, 30)
                torus.paint_uniform_color(color)
                self.ct_spheres = self.ct_spheres + torus # replace the ct_spheres

                self.torus_normals.append(normal)
                self.torus_origins.append(origin)

        if(simple_normal==1):
            
            for centroid, color in zip(self.ct_centroids,colors):
                
                
                origin = centroid
                closest_index = np.argmin(np.linalg.norm(self.aortic_centreline-origin,axis=1))
                closest_point = self.aortic_centreline[closest_index,:]
                normal = origin-closest_point
                normal = normal/np.linalg.norm(normal)

                # torus = create_torus(centroid, normal, major_radius, minor_radius, 30)
                torus = create_torus(origin, normal, self.major_radius, self.minor_radius, 30)
                torus.paint_uniform_color(color)
                self.ct_spheres = self.ct_spheres + torus # replace the ct_spheres

                self.torus_normals.append(normal)
                self.torus_origins.append(origin)

        # self.vis_2_spheres = o3d.geometry.TriangleMesh()
        self.vis_2_spheres = copy.deepcopy(self.ct_spheres) #temp for tests

        orig_points = np.asarray(self.ct_spheres.vertices)
        self.ct_spheres_points = copy.deepcopy(orig_points)
        self.vis_2_spheres_points = copy.deepcopy(orig_points)
        self.knn_idxs_spheres, self.knn_weights_spheres = precompute_knn_mapping(self.registered_ct_mesh, self.ct_spheres_points, k=3)
        self.knn_idxs_spheres_dup, self.knn_weights_spheres_dup = precompute_knn_mapping(self.registered_ct_mesh, self.vis_2_spheres_points, k=3)
        
        # ---- END OF SPHERE RENDERING ---- #


        # self.knn_idxs_spheres, self.knn_weights_spheres = precompute_knn_mapping(self.registered_ct_mesh, self.ct_centroids, k=3)  # uncomment this!!




    

        # # SMOOTHING MESH AS IMPORTED!! - not anymore
        # self.registered_ct_mesh = self.registered_ct_mesh.filter_smooth_taubin(number_of_iterations=10)
        # self.registered_ct_lineset = create_wireframe_lineset_from_mesh(self.registered_ct_mesh)
        # #END

        self.registered_ct_lineset.paint_uniform_color([0,0,0])

        # self.registered_ct_mesh_2 = copy.deepcopy(self.registered_ct_mesh)
        
        self.registered_ct_mesh_2.compute_vertex_normals()


        # view inside of vessel wall
        self.vis2.get_render_option().mesh_show_back_face = False

        if(self.vis_red_vessel==1):
            self.vis.get_render_option().mesh_show_back_face = False

        flip_mesh_orientation(self.registered_ct_mesh_2) 

        # view outside of tracker
        # flip_mesh_orientation(self.tracker) 

        # DELETED FOR FEVAR
        # self.vis2.add_geometry(self.ct_spheres)
        self.vis.add_geometry(self.ct_spheres)
        self.vis2.remove_geometry(self.ct_spheres)
        self.vis2.add_geometry(self.vis_2_spheres)
        self.vis2.add_geometry(self.registered_ct_mesh_2)
        self.vis2.add_geometry(self.tracker)

        if(self.vis_red_vessel==1):
            self.vis.add_geometry(self.registered_ct_mesh_2)
        else:
            self.vis.add_geometry(self.registered_ct_lineset)


        # adding raw volumetric far point cloud data
        self.far_pc = o3d.io.read_point_cloud(self.write_folder + '/volumetric_far_point_cloud.ply')
        if(self.refine==1):
            self.far_pc = o3d.io.read_point_cloud(self.write_folder + '/volumetric_far_point_cloud_refine.ply')
        self.far_pc.paint_uniform_color([0,0,1])

        # VISUALIZE BLUE FAR PC
        # self.vis.add_geometry(self.far_pc)

        copy_far_pc = copy.deepcopy(self.far_pc)
        self.far_pc_points = np.asarray(copy_far_pc.points)
        

        
        # self.vis.remove_geometry(self.catheter)

        

        

        # DELETED FOR FEVAR
        # self.vis.add_geometry(self.ct_spheres)

        self.knn_idxs_far_pc, self.knn_weights_far_pc = precompute_knn_mapping(self.registered_ct_mesh, self.far_pc_points, k=3)


        
        print("registered the ct from non rigid icp!!")

        self.scene = o3d.t.geometry.RaycastingScene()
        self.registered_ct_mesh_copy = copy.deepcopy(self.registered_ct_mesh)
        self.registered_ct_mesh_copy = o3d.t.geometry.TriangleMesh.from_legacy(self.registered_ct_mesh_copy)
        _ = self.scene.add_triangles(self.registered_ct_mesh_copy)  # we do not need the geometry ID for mesh

        # Get current camera parameters
        view_control_1 = self.vis.get_view_control()

        points = np.asarray(self.registered_ct_lineset.points)  # Convert point cloud to numpy array
        self.registered_centroid = np.mean(points, axis=0) 
        view_control_1.set_lookat(self.registered_centroid)

        view_control_1.set_up([0,-1,0])


        view_control_1.set_front([0,0,1])
        

        view_control_1.set_zoom(0.25)

        # roll 180 degrees
        # view_control_1.rotate(180, 0) 

        self.tsdf_map = 0

        self.view_control_1 = view_control_1


        self.cardiac_initialized=1


        


        




        

        
        
    
      
    
    def tracking(self):

        self.cardiac_initialized=0

        self.write_folder = rospy.get_param('dataset', 0) 

        self.deeplumen_on = 1
        self.deeplumen_slim_on = 0
        self.deeplumen_lstm_on = 0

        if(self.write_folder ==0):
            self.write_folder = self.prompt_for_folder()
            
        while(self.write_folder ==None):
            self.write_folder = self.prompt_for_folder()

        self.extend = 0
        self.record = 0

        self.image_batch=[]
        self.tw_em_batch=[]

        self.image_tags=[]
        self.starting_index = 1

        # if(self.dest_frame == 'target1'):
        #     print("assuming integrated catheter")
        #     self.vpC_map = 0

        try:
            
            self.vis.remove_geometry(self.processed_centreline)
            self.vis.remove_geometry(ivus_funsr_lineset)
            self.vis.remove_geometry(ivus_funsr_mesh)
            
        except NameError:
            print("no geometries to delete!")


        try:
            self.vis.remove_geometry(self.mesh_near_lumen)
        except:
            print("no near lumen present")
        

        # if IVUS present in tracking catheter
        try:
            self.vis.remove_geometry(self.us_frame)
        except:
            print("no us frame present")
        try:
            self.vis.remove_geometry(self.volumetric_near_point_cloud)
        except:
            print("no volumetric near point cloud present")
        try:
            self.vis.remove_geometry(self.volumetric_far_point_cloud)
        except:
            print("no volumetric far point cloud present")

        try:
            self.vis.remove_geometry(self.mesh_near_lumen_lineset)
        except:
            print("no near lumen mesh present")
        
       

  
        self.volumetric_far_point_cloud = o3d.geometry.PointCloud() 
        self.volumetric_near_point_cloud = o3d.geometry.PointCloud() 
        self.boundary_near_point_cloud = o3d.geometry.PointCloud() 

        self.vis.add_geometry(self.volumetric_near_point_cloud )
        self.vis.add_geometry(self.volumetric_far_point_cloud )
   

        # try healthy lumen tracking for live deformation!!
        
        # load the yaml file - specific to post processing

        # design checkbox for this
        print("loading tracking yaml")
        with open('/home/tdillon/mapping/src/aortascope_tracking_params.yaml', 'r') as file:
        # with open('/home/tdillon/mapping/src/aortascope_tracking_params_dissection.yaml', 'r') as file:
            config_yaml = yaml.safe_load(file)

        self.load_parameters(config_yaml)
        print("loaded tracking yaml")

        with open('/home/tdillon/mapping/src/calibration_parameters_guidewire.yaml', 'r') as file:
            calib_yaml_gw = yaml.safe_load(file)

        translation_gw = calib_yaml_gw['/translation']
        radial_offset_gw = calib_yaml_gw['/radial_offset']
        oclock_gw = calib_yaml_gw['/oclock']
        TEM_GW = [[1,0,0,translation_gw],[0,1,0,radial_offset_gw*np.cos(oclock_gw)],[0,0,1,radial_offset_gw*np.sin(oclock_gw)],[0, 0, 0, 1]]
        self.TEM_GW = np.asarray(TEM_GW)
     
        

        
        if(self.record_poses == 1):
            print("initialized post batches")
            self.pose_batch=[]
            self.image_tags=[]
           

        # GUI SPECIFIC - self.write_folder = rospy.get_param('dataset', 0) 

        

        if(self.write_folder ==0):
            self.write_folder = self.prompt_for_folder()
            
        while(self.write_folder ==None):
            self.write_folder = self.prompt_for_folder()

        self.extend = 0
        self.record = 0

        # if(self.dest_frame == 'target1'):
        #     print("assuming integrated catheter")
        #     self.vpC_map = 0
        try:
            
            self.vis.remove_geometry(self.processed_centreline)
            self.vis.remove_geometry(ivus_funsr_lineset)
            self.vis.remove_geometry(ivus_funsr_mesh)
            
        except NameError:
            print("no geometries to delete!")


        try:
            self.vis.remove_geometry(self.mesh_near_lumen)
        except:
            print("no near lumen present")
        if hasattr(self, 'registered_ct') and self.registered_ct is not None:
            print("registered_ct is", self.registered_ct)
        else:
            print("registered_ct is not defined or is None")
        if(self.registered_ct ==1):

            self.load_registetered_ct()
            

        print("finished registering CT mesh")

        if(self.live_deformation == 1 or self.cardiac_deformation ==1):
            self.constraint_radius=self.default_values['/constraint_radius'] 

            # don't deform the subbranches
            self.free_branch_locations = self.constraint_locations

            # constrain the subbranches
            # self.constraint_locations = np.vstack((self.constraint_locations,np.asarray(self.centerline_pc.points)[0,:],np.asarray(self.centerline_pc.points)[-1,:]))

            # don't constrain the subbranches, aortic endpoints only - need larger constraint radius to capture aortic valve orifice
            self.constraint_locations = np.vstack((np.asarray(self.centerline_pc.points)[0,:],np.asarray(self.centerline_pc.points)[-1,:]))

            

            self.constraint_radius = 0.02
            
            # could also have no constraints if desirable (TAVR)
            # self.constraint_locations = np.empty((0,3))

            self.constraint_indices = get_all_nodes_inside_radius(self.constraint_locations, self.constraint_radius, self.registered_ct_mesh)

            #release the applied deformation to these points but still allow them to move with adjacent nodes, just don't prescribe a deformation to them
            self.free_branch_radius = 0.0065
            self.free_branch_indices = get_all_nodes_inside_radius(self.free_branch_locations, self.free_branch_radius, self.registered_ct_mesh)

            # WE ARE OMMITTING FREE BRANCH
            # self.free_branch_indices = None
            
            # visualize the constraints if desired
            # test_pc= o3d.geometry.PointCloud() 
            # test_pc.points = o3d.utility.Vector3dVector(np.asarray(self.registered_ct_mesh.vertices)[self.constraint_indices,:])
            # constraint_seeds = get_sphere_cloud(self.constraint_locations, 0.0025, 12, [0,0,1])
            # o3d.visualization.draw_geometries([test_pc, self.registered_ct_lineset, constraint_seeds])

        

        if(self.cardiac_deformation==1):

            




            # uncomment if crashes quicker
            # self.vertices_before = copy.deepcopy(np.asarray(self.ct_spheres.vertices))
            # self.knn_idxs_spheres, self.knn_weights_spheres = precompute_knn_mapping(self.registered_ct_mesh, self.vertices_before, k=3)

            # get the ecg calibration
            t_start = rospy.Time.now()
            t_start = t_start.to_sec()
            delta = 0

            

            # initialize ecg parameters
            # if(self.animal == 1):
            ecg_calibration_data = []
            while delta < 4.0:
            # or while reg status not there

                t_current = rospy.Time.now()
                t_current = t_current.to_sec()
                delta = t_current  - t_start
                ecg_calibration_data.append(self.ecg_latest)
            self.ecg_threshold = calibrateThreshold(ecg_calibration_data)

            # else:
            #     self.ecg_threshold = 620

            print("ecg calibrated threshold is: ", self.ecg_threshold)

            # for initialize
            self.previous_peak = rospy.Time.now()
            self.previous_peak = self.previous_peak.to_sec()
            self.period = rospy.Duration(1.0) 
            self.period = self.period.to_sec()
            self.ecg_previous = self.ecg_latest
            self.ecg_state = 1

            self.deformed_mesh = copy.deepcopy(self.registered_ct_mesh)

            

            # load cardiac motion initialization
            self.systole_locations = np.load(self.write_folder + '/systole_locations.npy')
            self.diastole_locations = np.load(self.write_folder + '/diastole_locations_before_pulsatile.npy')

            # precompute values
            vertsTransformed_full = np.load(self.write_folder+ "/vertsTransformed_full.npy")
            C =vertsTransformed_full
            segment_lengths = np.linalg.norm(C[1:] - C[:-1], axis=1)
            d_cum_centerline = np.concatenate(([0], np.cumsum(segment_lengths)))
            tree = cKDTree(C)
            _, closest_idx = tree.query(self.diastole_locations)
            self.d_cum = d_cum_centerline[closest_idx]  # (K,)
            self.direction_vectors = self.diastole_locations - self.systole_locations  # shape (K,3)
            self.PWV = 2.5

            # VALIDATE MOTION
            mesh_1 = o3d.io.read_triangle_mesh('/home/tdillon/datasets/record_test/gated/bin_3/tsdf_mesh_near_lumen.ply')
            mesh_1_lineset = create_wireframe_lineset_from_mesh(mesh_1)
            mesh_1_lineset.paint_uniform_color([1,0,0])
            
            mesh_2 = o3d.io.read_triangle_mesh('/home/tdillon/datasets/record_test/gated/bin_8/tsdf_mesh_near_lumen.ply')
            mesh_2_lineset = create_wireframe_lineset_from_mesh(mesh_2)
            mesh_2_lineset.paint_uniform_color([0,0,1])

            # self.vis.add_geometry(mesh_1_lineset)
            # self.vis.add_geometry(mesh_2_lineset)


        if(self.dissection_mapping == 1):
            if(self.dissection_track == 1):

                self.dissection_lumen_1 = o3d.io.read_triangle_mesh(self.write_folder + '/near_lumen_mesh.ply')
                self.dissection_lumen_2 = o3d.io.read_triangle_mesh(self.write_folder + '/far_lumen_mesh.ply')

                self.dissection_lumen_1.compute_vertex_normals()
                self.dissection_lumen_2.compute_vertex_normals()

                self.dissection_lumen_1.paint_uniform_color([1,0,0])
                self.dissection_lumen_2.paint_uniform_color([0,0,1])

                self.vis.add_geometry(self.dissection_lumen_1)
                self.vis.add_geometry(self.dissection_lumen_2)

                self.vis2.add_geometry(self.dissection_lumen_1)
                self.vis2.add_geometry(self.dissection_lumen_2)

                

                self.vis2.add_geometry(self.tracker)


        if(self.deeplumen_on ==1):
    

            if not hasattr(self, 'model'):
                self.initialize_deeplumen_model()

        if(self.deeplumen_slim_on ==1):
    

            if not hasattr(self, 'model'):
                self.initialize_deeplumen_model()

        if(self.deeplumen_lstm_on ==1):
    

            if not hasattr(self, 'model'):
                self.initialize_deeplumen_model()


        self.vis.poll_events()
        self.vis.update_renderer()

        self.vis2.poll_events()
        self.vis2.update_renderer()

        # remove this in future
        self.simulate_device()
        self.evar_loft_sim = 0

        self.catheter_radius = 0.0015

        # integrated IVUS and steering
        self.bend_plane_angle = -1*(np.pi / 180)*12 # not 018 endoanchor at least, think its 035 endoanchor, but not necessarily normal aptus..
        if(self.endoanchor==1):
            self.vpC_map = 1
            self.vis.add_geometry(self.volumetric_near_point_cloud)
            self.vis.add_geometry(self.volumetric_far_point_cloud)
            self.vis.add_geometry(self.simple_far_pc)

            self.bend_plane_angle = -2*(np.pi / 180)*12 # 018 endoanchor

            # self.catheter_radius = 0.0025
            # self.bending_segment_color = [0.2, 0.2, 1.0]
            

            # self.catheter.paint_uniform_color([1,0.5,0])
            # self.vis.add_geometry(self.catheter)

            # self.vis2.add_geometry(self.catheter) # viewing from ivus probe now
            # self.vis2.remove_geometry(self.tracker)

            #true location of shaft not being represented, but true location of IVUS is
            # self.vis.remove_geometry(self.tracker)



            # LOAD THE ENDOANCHOR CALIBRATION SO DON'T GET MIXED UP
            # with open('/home/tdillon/mapping/src/calibration_parameters_endoanchor.yaml', 'r') as file: # 035
            with open('/home/tdillon/mapping/src/calibration_parameters_018_endoanchor_2.yaml', 'r') as file: #018
                self.calib_yaml = yaml.safe_load(file)

            self.angle = self.calib_yaml['/angle']
            self.translation = self.calib_yaml['/translation']
            self.radial_offset = self.calib_yaml['/radial_offset']
            self.o_clock = self.calib_yaml['/oclock']



            self.deeplumen_on = 1
            self.deeplumen_slim_on = 0
            self.deeplumen_lstm_on = 0

            print("endoanchor active")

        flip_mesh_orientation(self.tracker) 

        if(self.load_evar_graft==1):
            self.load_previous_fevar()

            
            
            
  

            



        

    


    def switch_probe_view(self):
        print("switched_frames")
        # just switch the transform you fetch
        if(self.dest_frame == 'target1'):

            print("switching")
            self.dest_frame = 'target2'

            # turn off ML components
            self.deeplumen_on = 0
            self.deeplumen_slim_on = 0
            self.deeplumen_lstm_on = 0


            self.vis.remove_geometry(self.catheter)
            # self.vis.remove_geometry(self.tracker_frame) # use this and catheter base frame to make sure you're getting reasonable instrument tracking

            # try:
            # self.vis.remove_geometry(self.volumetric_near_point_cloud)

        self.switch_probe = False

        # self.vis2.add_geometry(self.bending_segment)
        
        self.vis.remove_geometry(self.volumetric_near_point_cloud)
        self.deeplumen_on=0
        self.vpC_map=0

            
         

        # elif (self.dest_frame == 'target2'):
        #     self.dest_frame = 'target1'

       


  

    def refine_record(self):

        # setting this to EVAR record instead

        

        self.refine=1
        self.extend=1
        self.tsdf_map =1
        self.vpC_map = 1
        self.deeplumen_on=1
        self.orifice_center_map = 0

        

        

        # really should not try to define "refine" objects and just use existing code, resetting geometries here
        # you can still SAVE them under a different name though
        self.tsdf_volume_refine = FastTsdfIntegrator(self.voxel_size, self.sdf_trunc)
        self.mesh_near_lumen=o3d.geometry.TriangleMesh()
        self.mesh_near_lumen_lineset  = o3d.geometry.LineSet()
     
        self.vis.add_geometry(self.mesh_near_lumen_lineset)

        self.mesh_near_lumen = o3d.geometry.TriangleMesh()

        self.orifice_center_point_cloud=o3d.geometry.PointCloud()

        self.volumetric_near_point_cloud = o3d.geometry.PointCloud()
        self.volumetric_far_point_cloud = o3d.geometry.PointCloud()

        # self.vis.add_geometry(self.volumetric_near_point_cloud)
        # self.vis.add_geometry(self.volumetric_far_point_cloud)  # removed for EVAR record

        

        # self.vis.add_geometry(self.orifice_center_point_cloud)


        
        # self.orifice_center_spheres = o3d.geometry.TriangleMesh()
        # self.vis.add_geometry(self.orifice_center_spheres)

        
        

        print("starting tsdf refinement mapping")

        self.transformed_centroids=[]
        self.centroid_buffer = deque(maxlen=self.buffer_size)
        self.delta_buffer_x = deque(maxlen=self.buffer_size)
        self.delta_buffer_y = deque(maxlen=self.buffer_size)
        self.delta_buffer_z = deque(maxlen=self.buffer_size)

        view_control_1 = self.vis.get_view_control()
        view_control_1.set_up([0,-1,0])
        view_control_1.set_front([0,0,-1])
        view_control_1.set_zoom(0.25)

        

        
  

    def call_refine_reg(self):



        self.save_refine_data()

        print("SAVED REFINE DATA!!!")

        self.extend = 0
        self.refine=0



        # since this is now EVAR the code below is temporarily not relevant

        # while(self.refine_done ==0):

        #     # refine_done = rospy.get_param('refinement_computed', 0)
        #     print("waiting for refinement to compute...")
        #     time.sleep(1)

        # print("refinement complete (check if result is reasonable)")

        # # initialize tracking type code
        # self.refine=1

        # # get rid of old geometry
        # self.vis2.remove_geometry(self.registered_ct_mesh_2)
        # self.vis.remove_geometry(self.registered_ct_lineset)
        # self.tracking()

        # self.tsdf_map = 0
        # # self.refine = 0 # want to call the refined centerline for device simulation
        # self.extend = 0

    def simulate_device(self):

        self.fenestrate = 0

        # do all initialization here
        self.evar_slide_sim = 0
        self.evar_loft_sim = 1
        self.tavr_sim = 0

        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        self.deeplumen_on = 0
        self.deeplumen_slim_on = 0
        self.deeplumen_lstm_on = 0
        self.extend = 0
        self.vpC_map = 0
        self.live_deform = 0
        # self.refine = 0 # want to call the refined centerline

        if(self.registered_ct == 1):
            self.vis2.remove_geometry(self.registered_ct_mesh_2)
            self.vis2.remove_geometry(self.tracker)
            # self.vis.remove_geometry(self.catheter)
            self.vis.remove_geometry(self.registered_ct_lineset)
            self.vis.remove_geometry(self.ct_spheres)
            # self.vis.remove_geometry(self.volumetric_near_point_cloud)
            self.vis.remove_geometry(self.volumetric_far_point_cloud)

            

        if(self.tavr_sim == 1):
            self.evar_graft = o3d.io.read_triangle_mesh('/home/tdillon/Downloads/sapient_stent_frame.stl')
            # fill in rest here

        if(self.evar_loft_sim):

            # precomputed
            # self.evar_radius = 0.014
            self.evar_radius = 0.012

            # deployed inside FEVAR
            # self.evar_radius = 0.009

            # self.evar_length = 0.14
            # self.evar_length = 0.15
            self.evar_length = 0.15
            # self.evar_length = 0.17
            # amplitude = 0.014

            amplitude = 0.006 # half the amplitude
            # num_struts = 5
            # num_struts = 12
            num_struts = 6
            # axial_spacing = 0.015
            axial_spacing = 0.011  # distance to same point on next ring
            # axial_spacing = 0.018 # 0.007*2 + 0.004
            # axial_spacing = 0.014
            self.no_graft_points = 12

            self.load_registetered_ct()
            
            self.aortic_centreline = np.asarray(self.centerline_pc.points)
            

            # IF YOU HAVE DONE VESSEL STRAIGHTENING
            # x_points, y_points, z_points = fit_3D_bspline(self.aortic_centreline, 0.0001)
            # self.aortic_centreline = np.column_stack((x_points,y_points,z_points))


            self.lofted_cylinder, self.strut_geometry, self.strut_distances, self.aortic_centreline, self.centreline_transforms, self.GD_centreline = get_evar_template_geometries(self.aortic_centreline, self.evar_radius, self.evar_length, amplitude, num_struts, axial_spacing, self.no_graft_points)


            self.lofted_cylinder_wireframe, placeholder_1, placeholder_2, placeholder_3, placeholder_4, placeholder_5 = get_evar_template_geometries(self.aortic_centreline, self.evar_radius, self.evar_length, amplitude, num_struts, axial_spacing, self.no_graft_points, circle_resolution=15)

            # for fenestrated evar (FEVAR)


            # self.fen_distances =np.asarray([0.044, 0.056, 0.1155, 0.07])
            # self.fen_angles = np.asarray([2*3.24, 2*3.24, 3.12/2, 2*2.7])

            self.lofted_cylinder = o3d.t.geometry.TriangleMesh.from_legacy(self.lofted_cylinder)
            self.evar_graft = o3d.geometry.TriangleMesh()

            self.evar_wireframe = o3d.geometry.LineSet()
            self.evar_struts_and_rings = o3d.geometry.TriangleMesh()

            if(self.show_evar_wireframe==1):
                self.vis.add_geometry(self.evar_struts_and_rings)
                self.vis.add_geometry(self.evar_wireframe)
            else:
                self.vis.add_geometry(self.evar_graft)

            # FOR FEVAR NAVIGATION
            self.vis2.add_geometry(self.evar_graft)

        return

        


    def get_catheter_transform(self,TW_EM):

        # T_shift = np.eye(4)
        # T_shift[:3,3] = np.asarray([0.002,0.0012,0.01])

    
        # direction = TW_EM[:3, 0]
        # cylinder_z_axis = np.array([0, 0, 1])
        # rotation_axis = np.cross(cylinder_z_axis, direction)
        # rotation_angle = np.arccos(np.dot(cylinder_z_axis, direction))
        # rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        # rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

        roll_axis = TW_EM[:3, 0]  # This will be aligned with the cylinder's roll axis
        short_axis_1 = TW_EM[:3, 1]  # This will be aligned with the first short axis of the cylinder
        short_axis_2 = TW_EM[:3, 2]  # This will be aligned with the second short axis of the cylinder

        # Normalize the axes to ensure they are unit vectors
        roll_axis = roll_axis / np.linalg.norm(roll_axis)
        short_axis_1 = short_axis_1 / np.linalg.norm(short_axis_1)
        short_axis_2 = short_axis_2 / np.linalg.norm(short_axis_2)

        # Construct the rotation matrix using the normalized basis vectors
        rotation_matrix = np.column_stack((short_axis_2, short_axis_1, roll_axis))

        T_transformation = np.eye(4)
        T_transformation[:3,:3]= rotation_matrix
        T_transformation[:3,3] = TW_EM[:3,3]
        
        T_catheter = T_transformation

        return T_catheter
    

    def close_app(self):
        cv2.destroyAllWindows()
        print("app closing")
        # self.save_pose_data()
        rospy.signal_shutdown('User quitted')

    
    def image_pause(self):
        print("CPU OVERHEATING, PAUSING FOR 15 SECONDS, TEMP ABOVE 90C")
        time.sleep(15)


    def ecg_callback(self,ecg_msg):

        # if(self.replay_data==1):
        #     # print("not called")
        #     return

        current_time = rospy.get_rostime()
        ecg_timestamp_secs = current_time.secs
        ecg_timestamp_nsecs = current_time.nsecs
        
        ecg_timestamp_in_seconds = ecg_timestamp_secs + (ecg_timestamp_nsecs * 1e-9)

        ecg_value = ecg_msg.data

        self.ecg_latest = ecg_value

        self.ecg_buffer.append(self.ecg_latest)

        # print("ecg_state", self.ecg_state)
        # print("ecg_ecg_latest", self.ecg_latest)
        # print("ecg_threshold", self.ecg_threshold)

        if(self.cardiac_deformation==1 and self.cardiac_initialized==1):
            # if(self.animal==1):
            #     if (self.ecg_latest < self.ecg_threshold) and (self.ecg_previous >= self.ecg_threshold) and self.ecg_state==1 and (rospy.Time.now().to_sec() - self.previous_peak > 0.1):

                    
            #         self.ecg_state=0

            # if(self.animal==0):

            

            # if (self.ecg_previous < self.ecg_threshold) and (self.ecg_latest >= self.ecg_threshold) and self.ecg_state==1 and (rospy.Time.now().to_sec() - self.previous_peak > 0.1):
            if np.any(np.array(self.ecg_buffer) < self.ecg_threshold) and (self.ecg_latest >= self.ecg_threshold) and self.ecg_state==1 and (rospy.Time.now().to_sec() - self.previous_peak > 0.3):
                self.ecg_state = 0 # won't be changed back until cardiac period is modified
                # print("beat!")

                

        if(self.record==1):
            self.ecg_times.append( ecg_timestamp_in_seconds)
            self.ecg_signal.append(ecg_value)

        
    def image_callback(self, msg):

        # print("step   :", msg.step)
        # print("data bytes:", len(msg.data))

        

        # rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8").copy()


        # enc = msg.encoding.lower()

        # if enc in ["rgb8", "bgr8"]:
        rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8").copy()
        
        # elif enc in ["yuyv", "yuyv422"]:
        #     data = np.frombuffer(msg.data, np.uint8)
        #     yuyv = data.reshape((msg.height, msg.width, 2))
        #     img = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
        #     cv2.imshow("frame", img)
        #     cv2.waitKey(0)
        #     print("DIFFERENT")

        # else:
        #     print("UNKNOWN ENCODING", enc)
        #     return

        


        # convert YUYV -> BGR
        # rgb_image = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)

        # print("Converted image:", rgb_image.shape, rgb_image.dtype)

        

        # h, w, c = rgb_image.shape
        # print("h", h)
        # print("w", w)
        # if (h, w) != (1024, 1280):
        #     print(f"WARNING: Unexpected image size: {rgb_image.shape}")
        #     ERROR # wrong image size!

        # if msg.height != 1024 or msg.width != 1280:
        #     print(f"ROS message is wrong size: {msg.height}x{msg.width}")

        # print(msg.encoding)


        # don't process identical message data
        # Assuming RGB format
        # ORIGINAL METHOD
        # rgb_image_data = np.frombuffer(msg.data, dtype=np.uint8)
        # rgb_image = rgb_image_data.reshape((self.image_height, self.image_width, 3))

        # note there are duplicate images coming in because the ultrasound machine is slow at updating!!
        # fix that in future to reduce computational load, detect identical images efficiently!

        
        
        now = rospy.Time.now()
        self.start_entire = now.to_sec()
        delta = now - self.last_image_time  
        # print("rospy difference time", delta.to_sec())

        if delta < rospy.Duration(0.02):  # max ~50 Hz
            print("throttling to cool CPU!")
            return


        # if(self.endoanchor!=1):
        #     if delta < rospy.Duration(0.02):  # max ~50 Hz
        #         print("throttling!")
        #         return
        # else:
        #     if delta < rospy.Duration(0.0427):  # max 20 Hz - Nyquist for 11.7 Hz
        #         # print("skipping!")
        #         return

        
            
        self.last_image_time = now

        # self.start_entire = time.time()

        

        # self.replay_data = rospy.get_param('replay', 0 )
        if(self.replay ==1):
            self.replay_function()
            self.replay = 0
            self.replay = False
            # rospy.set_param('replay', 0)

        # print("self.replay data", self.replay_data)
        # if(self.replay_data==1):
        #     # print("not called")
        #     return
        

        start_total_time = time.time()
        # this is the shortest number of lines it takes to grab the transform once image arrives
        # transform_time = rospy.Time(0) #get the most recent transform
        self.transform_time = msg.header.stamp #get the most recent transform

        if(self.test_transform ==1): 
            self.transform_time = rospy.Time(0)
        
        # Assuming you have the frame IDs for your transform
        ref_frame = 'ascension_origin'
        dest_frame = self.dest_frame

        # timing_delta = rospy.Duration(0.066) #ultrasound machine + frame grabber

        
        # if(self.registered_ct != 1 and self.test_transform!=1):

        

        # assuming dest frame 1 is the IVUS probe of some form
        if(self.dest_frame=='target1' and self.test_transform!=1):
            # print("lagging")
            # timing_delta = rospy.Duration(0.125) #ultrasound machine + frame grabber TURN OFF FOR NAVIGATION!
            timing_delta = rospy.Duration(0.375) #ultrasound machine + frame grabber TURN OFF FOR NAVIGATION!
            if(self.animal==1): # assuming 0.018 probe
                timing_delta = rospy.Duration(0.05) 
            self.transform_time = self.transform_time - timing_delta
            # self.transform_time = self.transform_time 
        


        try:
            # Lookup transform
            TW_EM = self.tf_buffer.lookup_transform(ref_frame, dest_frame, self.transform_time)
        except (rospy.ROSException, tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Failed to lookup transform")
            TW_EM = None

        
        




        # ----- TEMP BOSTON SCIENTIFIC -----#

       

        # # run these functions once when triggered and update the visualizer accordingly
        # start_record = rospy.get_param('start_record', 0 )
        # if(start_record ==1):
        #     self.start_recording()
        #     rospy.set_param('start_record', 0)

        # save_dataset = rospy.get_param('save_data', 0 )
        # if(save_dataset ==1):
        #     self.save_image_and_transform_data()
        #     rospy.set_param('save_data', 0)
    
        # rgb_image_data = np.frombuffer(msg.data, dtype=np.uint8)

        # rgb_image = rgb_image_data.reshape((self.image_height, self.image_width, 3))

        # rgb_image_msg = Image(
              
        #         height=np.shape(rgb_image)[0],
        #         width=np.shape(rgb_image)[1],
        #         encoding='rgb8',
        #         is_bigendian=False,
        #         step=np.shape(rgb_image)[1] * 3,
        #         data=rgb_image.tobytes()
        #     )
        # self.rgb_image_pub.publish(rgb_image_msg)


        # if(self.record==1):
        #     # code for saving images

        #     self.image_batch.append(rgb_image)

        #     self.image_tags.append(self.image_number)

        #     self.image_number = self.image_number+1


        # # reset memory for efficiency
        # if(self.record == 1):
        #     if(len(self.image_batch)>150):

        #         # consider using threading while saving to save time
        #         self.quick_save()

        #         #clear the image batch and save the callback number
        #         self.image_batch = []
        #         self.image_tags = []
        #         self.image_number = 1

                


        # ------- END OF TEMP BOSTON SCIENTIFIC ------#

        if(self.test_transform==1):
            TW_EM = np.eye(4)
        else:
            TW_EM=transform_stamped_to_matrix(TW_EM)

        # prune fast probe moves
        current_time_in_sec = self.transform_time.to_sec()
        self.smoothed_linear_speed = update_linear_speed_ema(TW_EM, current_time_in_sec, self.previous_transform_ema, self.previous_time_in_sec, self.smoothed_linear_speed)
        self.previous_time_in_sec = current_time_in_sec
        self.previous_transform_ema = TW_EM

        # print("smoothed linear speed:",self.smoothed_linear_speed)
        # if self.registered_ct !=1 and self.smoothed_linear_speed > 0.2:  # max ~50 Hz
        #     print("probe speed too fast! omit image")
        #     return




        # now get the original image's timestamp
        timestamp_secs = msg.header.stamp.secs
        timestamp_nsecs = msg.header.stamp.nsecs
        image_timestamp_in_seconds = timestamp_secs + (timestamp_nsecs * 1e-9)

        # Create a rospy.Time object using the truncated timestamp information
        timestamp = rospy.Time(secs=timestamp_secs, nsecs=timestamp_nsecs)

        
        


        # TEST IMAGE test_image
        #circular test image
        # rgb_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        # center = (self.centre_x, self.centre_y)  # (x, y) coordinates of the circle center
        # radius = int(self.centre_x/2)  # Radius of the circle
        # color = (255, 255, 255)  # White color in BGR format
        # thickness = 5  # -1 to fill the circle, >0 for border thickness
        # cv2.circle(rgb_image, center, radius, color, thickness)

        
      
        grayscale_image=preprocess_ivus_image(rgb_image,self.box_crop,self.circle_crop,self.text_crop,self.crosshairs_crop)

        
        # BRANCH IMAGE SIMULATOR
        # self.branch_sim_index = self.branch_sim_index + 1

        # subbed_in = self.branch_sim_index//2
        # # subbed_in = self.branch_sim_index

        # rgb_image = cv2.imread("/home/tdillon/datasets/k8_tom_3/rgb_jpgs/grayscale_image_" + str(subbed_in) + ".jpg")
        # grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


        # # if(self.branch_sim_index > 1076):
        # #     self.branch_sim_index = 866

        # if(self.branch_sim_index > 1066*2):
        #     self.branch_sim_index = 1034*2
        
            
 
        # print("image number is", self.image_number)
        # recording
        if(self.record==1):
            # code for saving images

            self.image_batch.append(grayscale_image)
            self.tw_em_batch.append(TW_EM)
            self.image_times.append(image_timestamp_in_seconds)

            self.image_tags.append(self.image_number)

            self.image_number = self.image_number+1

        if(self.record_poses==1):

            self.pose_batch.append(TW_EM)

            self.image_tags.append(self.image_number)

            self.image_number = self.image_number+1

        # reset memory for efficiency
        if(self.record == 1):
            if(len(self.image_batch)>150):

                # consider using threading while saving to save time
                self.quick_save()

                #clear the image batch and save the callback number
                self.image_batch = []
                self.tw_em_batch = []
                self.image_tags = []
                self.image_number = 1
                

        if(self.record_poses == 1):
            # if(len(self.image_batch)>500):
            if(len(self.pose_batch)>500):
                print("quick save!")
                self.save_pose_data()
                self.pose_batch = [] 
                self.image_tags = [] 
                self.image_number = 1

        

        # for computational efficiency, stop here and do all processing / reconstruction later
        if(self.tsdf_map != 1 and self.vpC_map != 1 and self.bpC_map!=1):
            pass

        self.append_image_transform_pair(TW_EM, grayscale_image)

    

    def append_image_transform_pair(self, TW_EM, grayscale_image):



        # print("TW_EM:", TW_EM)

        # cv2.imshow("rgb_image", grayscale_image)
        # cv2.waitKey(0)

        original_image = grayscale_image.copy()
        
        # fetch rospy parameters for real time mapping (could be placed in initialization)
        threshold = self.threshold
        no_points = self.no_points
        crop_index = self.crop_index
        scaling = self.scaling
        angle = self.angle
        translation = self.translation
        radial_offset = self.radial_offset
        oclock = self.o_clock

        # GUI LOGIC
        # run these functions once when triggered and update the visualizer accordingly
        if self.start_record:
            self.start_recording()
            self.start_record = False  # reset flag

        if self.save_data:
            self.save_image_and_transform_data()
            self.save_data = False

        if self.gate:
            self.gate_data()
            self.gate = False

        if self.funsr_start:
            self.funsr_started()
            self.funsr_start = False

        if self.funsr_complete:
            print("loading IVUS geometry")
            self.funsr_done()
            self.funsr_complete = False

        if self.registration_start:
            self.registration_started()
            self.registration_start = False

        if self.reg_complete:
            self.tracking()
            self.reg_complete = False

        if self.pause:
            print("detected temp rise")
            self.image_pause()
            self.pause = False

        if self.motion_capture:
            self.motion_capture_pause()
            self.motion_capture = False

        if self.switch_probe:
            self.switch_probe_view()
            self.switch_probe = False

        if self.refine_start:
            self.refine_record()
            self.refine_start = False

        if self.refine_complete:
            self.call_refine_reg()
            self.refine_complete = False

        if self.sim_device:
            self.simulate_device()
            self.sim_device = False


        # interact with hardware over rospy
        # pullback = rospy.get_param('pullback', 0)
        # self.pullback_pub.publish(pullback)



        centre_x=self.centre_x
        centre_y=self.centre_y

        

        # ------ FIRST RETURN SEGMENTATION -------- #

        final_component_data = []

        mask_1=None
        mask_2=None
       

        # first return segmentation
        # if(self.deeplumen_on == 0 and self.deeplumen_slim_on == 0 and self.deeplumen_lstm_on == 0 and self.dest_frame== 'target1' and self.endoanchor!=1):

        #     relevant_pixels=first_return_segmentation(grayscale_image,threshold, crop_index,self.gridlines)
        #     relevant_pixels=np.asarray(relevant_pixels).squeeze()

        #     # EM TRACKER ONLY MODE
        #     # relevant_pixels = np.asarray([[int(centre_x)-10, int(centre_y)], [int(centre_x)+10, int(centre_y)], [int(centre_x), int(centre_y)], [int(centre_x), int(centre_y)+2], [int(centre_x)+1, int(centre_y)]])
            
            
        #     # ellipse fitting to first return
            
        #     # For compatibility with subsequent deeplumen assumed functions
        #     mask_1 = np.zeros_like(grayscale_image, dtype=np.uint8)  # Create a black mask
        #     mask_2 = np.zeros_like(grayscale_image, dtype=np.uint8)  # Create a black mask
        #     original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)


        #     if(relevant_pixels!=[]):
        #         ellipse_model = cv2.fitEllipse(relevant_pixels[:,[1,0]].astype(np.float32))  
        #         ellipse_contour= cv2.ellipse2Poly((int(ellipse_model[0][0]), int(ellipse_model[0][1])),
        #                                         (int(ellipse_model[1][0]/2), int(ellipse_model[1][1]/2)),
        #                                         int(ellipse_model[2]), 0, 360, 5)
        #         cv2.fillPoly(mask_1, [ellipse_contour], 255)  # Filled white ellipse on black mask
        #         cv2.drawContours(original_image, [ellipse_contour], -1, (0, 0, 255), thickness = 1)
            
        #     # insert IVUSProcSegmentation instead
            
            


        #     mask_1 = cv2.resize(mask_1, (224, 224))
        #     mask_2 = cv2.resize(mask_2, (224, 224))

        #     mask_1_contour,hier = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     mask_2_contour,hier = cv2.findContours(mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        #     rgb_image_msg = Image(
              
        #         height=np.shape(original_image)[0],
        #         width=np.shape(original_image)[1],
        #         encoding='rgb8',
        #         is_bigendian=False,
        #         step=np.shape(original_image)[1] * 3,
        #         data=original_image.tobytes()
        #     )
        #     self.rgb_image_pub.publish(rgb_image_msg)

            

        # if((self.deeplumen_on == 1 or (self.deeplumen_slim_on == 1 or self.deeplumen_lstm_on == 1)) and self.dest_frame=='target1'):

        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        # print("deeplumen", self.deeplumen_on)

        if(((self.deeplumen_on == 1 or (self.deeplumen_slim_on == 1 or self.deeplumen_lstm_on == 1)) and self.dest_frame=='target1') or self.endoanchor==1):

            # print("segmenting")

            if(self.deeplumen_on == 1 or self.deeplumen_slim_on == 1 or self.endoanchor == 1):
            
                # FOR DISSECTION IMPLEMENTATION LATER

            
                

                # # note 224,224 image for compatibility with network is hardcoded
                grayscale_image = cv2.resize(grayscale_image, (224, 224))
                image = cv2.cvtColor(grayscale_image,cv2.COLOR_GRAY2RGB)


                # #---------- SEGMENTATION --------------#


                start_time = time.time()

                

                # mask_1, mask_2 = deeplumen_segmentation(image,self.model)
                # pred, conf_class2= deeplumen_segmentation(image,self.model)
                # pred, conf_class2, conf_colormap, overlay = deeplumen_segmentation(image, self.model)
                pred, conf_class2 = deeplumen_segmentation(image, self.model)
                conf_class2 = conf_class2.numpy()
                # conf_colormap, overlay = build_colormap(conf_class2)
                raw_data = pred[0].numpy()
                mask_1, mask_2, largest_two_masks, spline_pixels = post_process_deeplumen(raw_data, conf_class2, self.conf_threshold)

            

                # now you can safely do numpy conversions
                # if conf_class2 is not None:
                #     max_conf_class2 = tf.reduce_max(conf_class2).numpy()
                #     print("max conf:", max_conf_class2)
                #     conf_class2_np = conf_class2.numpy()  # float confidence map
                #     cv2.imshow("Confidence", conf_colormap.numpy())
                #     cv2.waitKey(0)

                        
                

                # get_gpu_temp()
                # get_cpu_temp()

                end_time=time.time()
                diff_time=end_time-start_time
                print("segmentation time:", diff_time)


                # this was coded for when deeplumen valve wasn't coded properly
                # if(self.deeplumen_valve_on == 1):
                #     pred= deeplumen_segmentation(image,self.model_cusp)
                #     raw_data = pred[0].numpy()
                #     mask_1_opening, mask_2_cusp = post_process_deeplumen(raw_data)


                #     # get mask_2 as subtraction of mask_1_opening from mask_1
                #     # overwriting output from ML model!!
                    
                #     # mask_2 = np.logical_and(mask_1 == 1, mask_1_opening == 0).astype(np.uint8)
                #     mask_2 = np.clip(mask_1 - mask_1_opening, 0, 1)
                #     mask_1 = mask_1_opening

                #     # cv2.imshow("mask 2 example", mask_2)
                #     # cv2.waitKey(0)

                    

            if(self.deeplumen_lstm_on == 1):

                # add most recent image to buffer
                grayscale_image = cv2.resize(grayscale_image, (224, 224))
                image = cv2.cvtColor(grayscale_image,cv2.COLOR_GRAY2RGB)
                self.grayscale_buffer.append(image)
                
                # fetch sequence of images from buffer
                if(len(self.grayscale_buffer) >= self.lstm_length):
                

                    # get mask_1 and mask_2 as usual
                    start_time = time.time()

                    sequence = np.stack(self.grayscale_buffer, axis=0) 

                    pred= deeplumen_lstm_segmentation(sequence,self.model)
                    raw_data = pred[0].numpy()
                    mask_1, mask_2 = post_process_lstm_deeplumen(raw_data)


                    end_time=time.time()
                    diff_time=end_time-start_time
                    print("segmentation LSTM time:", diff_time)

                else:
                    return
            
                
            start_half = time.time()


            if(self.orifice_center_map):

                final_component_data = []
                # every point for orifice detection
                mask_1_contour_every_point, hier = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                indices_added = []

                buffers_touched = set()

                new_check=0

                empty_mask = np.zeros_like(mask_2, dtype=np.uint8)
                self.mask_2_buffers[0].append([self.mask_2_buffers[0][-1][0], empty_mask, np.nan, 0])
                self.mask_2_buffers[1].append([self.mask_2_buffers[1][-1][0], empty_mask, np.nan, 0])

                

                for component_mask in largest_two_masks:
                    
                
                    # ------ FIND ORIFICE PIXELS ------- #
                    kernel = np.ones((self.minimum_thickness, self.minimum_thickness), np.uint8)
                    dilated_mask_1 = cv2.dilate(mask_1, kernel, iterations=1)
                    touching_pixels = cv2.bitwise_and(dilated_mask_1, component_mask)
                    non_zero_pixels = np.column_stack(np.where(touching_pixels > 0))
                    contour_points = np.vstack(mask_1_contour_every_point).squeeze()
                    non_zero_pixels_xy = non_zero_pixels[:, [1, 0]]
                    distances = cdist(non_zero_pixels_xy, contour_points, metric='euclidean')
                    nearest_indices = distances.argmin(axis=1)

                    
                    # if branch touches mask 1
                    if nearest_indices.size > 0:

                        # start_time = time.time()
                        

                        # if so, find the ORIFICE CENTER
                        contiguous_indices, contiguous_block_points = get_contiguous_block_from_contour(
                            contour_points, nearest_indices
                        )


                        orifice_mask = component_mask
                        contour_points = np.asarray(contiguous_block_points)
                        normals = visualize_contour_normals(orifice_mask, contour_points)


                        
                        raycast_hits, ray_lengths = compute_branch_raycast_hits(component_mask, contour_points, normals)


                        mid_index = len(raycast_hits) // 2
                        orifice_center_three_d_points = get_single_point_cloud_from_pixels(
                            [contiguous_block_points[mid_index]], scaling
                        )

                        # DETERMINE IF OVERLAP EXISTS WITH BUFFERS
                        

                        
                    
                        orifice_two_d = np.asarray(contiguous_block_points[mid_index]).squeeze()

                        num_overlaps=[]
                        average_orifice_angles=[]


                        # end_time = time.time()
                        # diff_time = end_time-start_time
                        # print("raycast time", diff_time)
                 

                        
                        
                        for mask_2_buffer in self.mask_2_buffers:
                            
                
                            masks = [entry[1] for entry in list(mask_2_buffer)[-self.branch_buffer_size-1:-1]] # -1 doesn't include the last
                    
                            combined_previous_mask = np.logical_or.reduce(np.stack(masks, axis=0))
                

                            combined_previous_mask = (combined_previous_mask * 255).astype(np.uint8)                            
                            overlap = np.logical_and(combined_previous_mask == 255, component_mask == 255)
                            
                            num_overlap_pixels = np.sum(overlap)
                            
                
                            num_overlaps.append(num_overlap_pixels)


                            # adding angle component
                            angles = []
                            for entry in list(mask_2_buffer)[-self.branch_buffer_size-1:-1]:
                                angle = np.array(entry[2], dtype=float).ravel()
                                if angle.shape[0] == 2:
                                    angles.append(angle)
                                else:
                                    angles.append(np.array([np.nan, np.nan]))

                            if angles:
                                angles = np.vstack(angles)  # shape (k, 2)
                                average_orifice_angle = np.nanmean(angles, axis=0)
                            else:
                                average_orifice_angle = np.array([np.nan, np.nan])
                            
                            # average_orifice_angle = np.nanmean(list(mask_2_buffer)[-n-1:-1, 3], axis=0)
                            average_orifice_angles.append(average_orifice_angle)

                        average_orifice_angles = np.asarray(average_orifice_angles)

                        
                        threshold = 10
                        angle_threshold = 25
                        overlap_1, overlap_2 = None, None


                        if(num_overlaps[0] > threshold and np.linalg.norm((orifice_two_d-average_orifice_angles[0,:]),axis=0) < angle_threshold):
                            overlap_1 = num_overlaps[0]
                            

                        if(num_overlaps[1]> threshold and np.linalg.norm((orifice_two_d-average_orifice_angles[1,:]),axis=0) < angle_threshold):
                            overlap_2 = num_overlaps[1]

                        # if num_overlaps[0] > threshold:
                        #     overlap_1 = num_overlaps[0]

                        # if num_overlaps[1] > threshold:
                        #     overlap_2 = num_overlaps[1]

                        # determine which branch id component overlaps with more
                        if overlap_1 is not None and overlap_2 is not None:
                            if overlap_1 > overlap_2:
                                # print("both overlap, 1 greater")
                                branch_pass_id = self.mask_2_buffers[0][-1][0]
                                relevant_buffer = 0
                                # print("fetched branch_pass_id", branch_pass_id)
                            else:
                                # print("both overlap, 2 greater")
                                branch_pass_id = self.mask_2_buffers[1][-1][0]
                                relevant_buffer = 1
                                # print("fetched branch_pass_id", branch_pass_id)

                        elif overlap_1 is not None:
                            # print("1 overlap only")
                            branch_pass_id = self.mask_2_buffers[0][-1][0]
                            relevant_buffer = 0
                            # print("fetched branch_pass_id", branch_pass_id)

                        elif overlap_2 is not None:
                            # print("2 overlap only")
                            branch_pass_id = self.mask_2_buffers[1][-1][0]
                            relevant_buffer = 1
                            # print("fetched branch_pass_id", branch_pass_id)


                        else:
                            

                            # print("no overlap")
                            # no overlap at all → new branch
                            
                            self.branch_pass = self.branch_pass + 1
                            branch_pass_id = self.branch_pass
                            
                
                            new_mask_2_buffer = self.init_buffer(0, self.branch_buffer_size, (224, 224), 0)
                            self.mask_2_buffers.append(new_mask_2_buffer)
                            relevant_buffer = len(self.mask_2_buffers) - 1

                            branch_pass_trigger = 0


                        # AREA - DIP LOGIC (note that branch trigger logic outside of this is harmless if you just comment this block out)
                        # special case to split branch passes if they are close together and there is no complete loss of overlap
                        # detect a dip followed by a rise

                        # detect dip
                        buffer_masks = [entry[1] for entry in self.mask_2_buffers[relevant_buffer]]
                        max_branch_pass_area = max(np.count_nonzero(m) for m in buffer_masks)

                        last_branch_pass_trigger = self.mask_2_buffers[relevant_buffer][-2][3]

           

                        if(np.count_nonzero(component_mask) < 0.66 * max_branch_pass_area):
                            branch_pass_trigger = 1
                            # print("triggered!")
                            
                        
                        else:
                            branch_pass_trigger = 0
                        
                        if last_branch_pass_trigger == 1:
                            branch_pass_trigger = 1

                        # detect rise
                        
                        if(np.count_nonzero(component_mask) > 0.825 * max_branch_pass_area and last_branch_pass_trigger ==1):
                            self.branch_pass = self.branch_pass + 1
                            branch_pass_id = self.branch_pass
                            
                
                            new_mask_2_buffer = self.init_buffer(0, self.branch_buffer_size, (224, 224), 0)
                            self.mask_2_buffers.append(new_mask_2_buffer)
                            relevant_buffer = len(self.mask_2_buffers) - 1

                            branch_pass_trigger = 0

                            # print("detected rise!")

                        # END OF AREA DIP LOGIC

                        
                        # if(branch_pass_id !=None or self.branch_pass<=1):
                        # append results
                        volumetric_three_d_points_far_lumen = get_single_point_cloud_from_mask(component_mask, scaling)
                        branch_pixels = np.count_nonzero(component_mask)
                        
                        # print("branch_pass_id being added", branch_pass_id)
                        final_component_data.append(
                            [branch_pass_id, orifice_center_three_d_points, volumetric_three_d_points_far_lumen, branch_pixels]
                        )

                        # self.mask_2_buffers[relevant_buffer].append([branch_pass_id, component_mask, orifice_two_d])

                        
                        self.mask_2_buffers[relevant_buffer][-1] = [branch_pass_id, component_mask, orifice_two_d, branch_pass_trigger]

                   
            
                
            # end_time = time.time()

            # diff_time = end_time - start_half
            # print("diff time", diff_time)
                

            # visualize the buffers with this!
            # show_masks_from_buffer(self.mask_2_buffers[0], n=self.branch_buffer_size, win_name="Buffer 0")
            # show_masks_from_buffer(self.mask_2_buffers[1], n=self.branch_buffer_size, win_name="Buffer 1")
           

            
    
            # ---- PUBLISH IMAGE (1ms) ----- #


            mask_1_contour,hier = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_2_contour,hier = cv2.findContours(mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # visualize segmentations
            grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(grayscale_image, mask_1_contour, -1, (0, 0, 255), thickness=1)
            cv2.drawContours(grayscale_image, mask_2_contour, -1, (255, 0, 0), thickness=1)

            # original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

            # for visualizing ransac
            # print("spline pixels", spline_pixels)

            # if spline_pixels is not None and len(spline_pixels) > 0:
            #     # rescale spline points to match original_image size
            #     spline_pixels_scaled = spline_pixels.copy().astype(np.float32)
            #     spline_pixels_scaled[:, 0] *= (np.shape(original_image)[0] / mask_1.shape[0])   # scale x
            #     spline_pixels_scaled[:, 1] *= (np.shape(original_image)[1] / mask_1.shape[1])   # scale y

            #     # reshape into OpenCV contour format
            #     spline_contour_send = spline_pixels_scaled.astype(np.int32).reshape((-1, 1, 2))

            #     # draw
            #     cv2.drawContours(original_image, [spline_contour_send], -1, (255, 0, 255), thickness=2)

            
            # header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
            # rgb_image_msg = Image(
            #     header=header,
            #     height=224,
            #     width=224,
            #     encoding='rgb8',
            #     is_bigendian=False,
            #     step=self.new_width * 3,
            #     data=grayscale_image.tobytes()
            # )
            # self.rgb_image_pub.publish(rgb_image_msg)

    
            mask_1_send = cv2.resize(mask_1, (np.shape(original_image)[0], np.shape(original_image)[1]))
            mask_2_send = cv2.resize(mask_2, (np.shape(original_image)[0], np.shape(original_image)[1]))

            # rather than finding contours, just colour mask to save time
            mask_1_contour_send,hier = cv2.findContours(mask_1_send, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_2_contour_send,hier = cv2.findContours(mask_2_send, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(original_image, mask_1_contour_send, -1, (0, 0, 255), thickness=2)
            cv2.drawContours(original_image, mask_2_contour_send, -1, (255, 0, 0), thickness=2)

        # cv2.imshow("original_image", original_image)
        # cv2.waitKey(0)
        
        # publish an image no matter what
    
        rgb_image_msg = Image(
            # header=header,
            height=np.shape(original_image)[0],
            width=np.shape(original_image)[1],
            encoding='rgb8',
            is_bigendian=False,
            step=np.shape(original_image)[1] * 3,
            data=original_image.tobytes()
        )
        self.rgb_image_pub.publish(rgb_image_msg)

       

        


        # FOT INTEGRATED IVUS AND STEERING GET RID OF THIS!
        # if(self.dest_frame=='target2'):
        #     mask_1 = np.zeros_like(grayscale_image)
        #     mask_2 = np.zeros_like(grayscale_image)

        #     original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        #     rgb_image_msg = Image(
        #         # header=header,
        #         height=np.shape(original_image)[0],
        #         width=np.shape(original_image)[1],
        #         encoding='rgb8',
        #         is_bigendian=False,
        #         step=np.shape(original_image)[1] * 3,
        #         data=original_image.tobytes()
        #     )
        #     self.rgb_image_pub.publish(rgb_image_msg)



            



        scaling=self.default_values['/scaling'] 
        # if dissection_parameterize == 0:
        #     three_d_points, three_d_points_near_lumen,three_d_points_far_lumen,three_d_points_dissection_flap = get_point_cloud_from_masks(combined_mask, scaling, mask_1_contour,mask_2_contour)

        # if dissection_parameterize == 1:
        #     three_d_points, three_d_points_near_lumen,three_d_points_far_lumen,three_d_points_dissection_flap = get_point_cloud_from_masks(combined_mask, scaling, mask_1_contour,mask_2_contour, dissection_flap_skeleton)


        
        if self.vpC_map == 1:
            volumetric_three_d_points_near_lumen = get_single_point_cloud_from_mask(mask_1, scaling)
            volumetric_three_d_points_far_lumen = get_single_point_cloud_from_mask(mask_2, scaling) 
        
            

        if self.bpC_map == 1:
            #for evaluation of surface accuracy at the end
            boundary_three_d_points_near_lumen = get_single_point_cloud_from_pixels(mask_1_contour, scaling)

        # ---- KINEMATICS ---- #

        # angle=self.default_values['/angle'] 
        # translation=self.default_values['/translation'] 
        # radial_offset=self.default_values['/radial_offset'] 
        # oclock=self.default_values['/oclock'] 

        angle=self.calib_yaml['/angle'] 
        translation=self.calib_yaml['/translation'] 
        radial_offset=self.calib_yaml['/radial_offset'] 
        oclock=self.calib_yaml['/oclock'] 

       
        TEM_C = [[1,0,0,translation],[0,np.cos(angle),-np.sin(angle),radial_offset*np.cos(oclock)],[0,np.sin(angle),np.cos(angle),radial_offset*np.sin(oclock)],[0, 0, 0, 1]]

        
        TEM_C = np.asarray(TEM_C)

        extrinsic_matrix=TW_EM @ TEM_C
        self.most_recent_extrinsic = extrinsic_matrix

        # ----- BSPLINE SMOOTHING FUNCTIONS  ----- #
        # this is early in the function so that the pose and masks get adjusted if they needs to be

        # ---- ADD TO BUFFERS ----- #
        if(self.dest_frame == 'target2'):
            mask_1 = np.zeros_like(grayscale_image)
            mask_2 = np.zeros_like(grayscale_image)


        if(self.dest_frame == 'target1' and np.count_nonzero(mask_1)>0):
            self.mask_1_buffer.append(mask_1)
            self.mask_2_buffer.append(mask_2)

            # ---- 2D CENTROID CALCULATIONS ----- #
            

            moments = cv2.moments(mask_1)
            centroid = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            centre_x = 224.0/2.0
            centre_y = 224.0/2.0
            centred_centroid=np.array(centroid)-[centre_x,centre_y]
            # centred_centroid=np.array(centroid)
            two_d_centroid=centred_centroid*scaling

        

        # if(self.extend==1 and self.dissection_mapping != 1):

        #         extrinsic_matrix = TW_EM @ TEM_C
                
        #         # ---- GET 3D CENTROIDS ----- #
        #         three_d_centroid = np.hstack((0,two_d_centroid))
        #         three_d_centroid=np.hstack((three_d_centroid,1)).T
        #         transformed_centroid = extrinsic_matrix @ three_d_centroid
                
        #         # ----- ADD TO BUFFERS ------ #
        #         self.centroid_buffer.append(transformed_centroid[:3])
        #         self.position_buffer.append(extrinsic_matrix)

            
        #         if(len(self.centroid_buffer) >= self.buffer_size):
        #             self.delta_buffer_x, self.delta_buffer_y, self.delta_buffer_z,closest_points, centrepoints = process_centreline_bspline(self.centroid_buffer, self.delta_buffer_x,self.delta_buffer_y,self.delta_buffer_z, self.buffer_size)
        #             self.processed_centreline.points.extend(o3d.utility.Vector3dVector(centrepoints))
        #             self.processed_centreline.paint_uniform_color([1,0,0])
        #             self.vis.update_geometry(self.processed_centreline)


        # post process the bspline smoothing instead - this prevents overheating
        if(self.extend==1 and self.dissection_mapping != 1):

                extrinsic_matrix = TW_EM @ TEM_C
                
                # ---- GET 3D CENTROIDS ----- #
                three_d_centroid = np.hstack((0,two_d_centroid))
                three_d_centroid=np.hstack((three_d_centroid,1)).T
                transformed_centroid = extrinsic_matrix @ three_d_centroid
                self.transformed_centroids.append(transformed_centroid)

                
                       


        # ------ NON SMOOTHED ESDF MESHING ------ #
        # colors_certainty_TL = update_esdf_mesh(self.voxelCarver, self.esdf_mesh,TW_EM @ TEM_C, mask_1, [1,0,0])
        # colors_certainty_FL = update_esdf_mesh(self.voxelCarver_2, self.esdf_mesh_2,TW_EM @ TEM_C, mask_2, [0,0,1])
        # if(certainty_coloring  == 1):
        #     self.esdf_mesh.color = colors_certainty_TL
        #     self.esdf_mesh_2.color = colors_certainty_FL
        # self.vis.update_geometry(self.esdf_mesh)
        # self.vis.update_geometry(self.esdf_mesh_2)


        

        # ------- VOXBLOX TSDF MESHING -------- #
        
        if(self.extend == 1):
            if(self.tsdf_map == 1 ):
            #if(self.tsdf_map == 1 and self.refine==0):
                combined_mask = cv2.bitwise_or(mask_1, mask_2)
                combined_mask = np.uint8(combined_mask)

                

                three_d_points, three_d_points_near_lumen, three_d_points_far_lumen, three_d_points_dissection_flap = get_point_cloud_from_masks(combined_mask, scaling, mask_1_contour,mask_2_contour)

                
                

                if(three_d_points_near_lumen is not None):
                    # if(self.gating==1): 
                    #     update_tsdf_mesh(self.vis, self.tsdf_volume_near_lumen,self.mesh_near_lumen,three_d_points_near_lumen, extrinsic_matrix,[1,0,0], keep_largest=False)
                    # else:

                    

                    update_tsdf_mesh(self.vis, self.tsdf_volume_near_lumen,self.mesh_near_lumen,three_d_points_near_lumen, extrinsic_matrix,[1,0,0], keep_largest=False)

                    
                    

                    if(self.dissection_mapping!=1 and np.shape(np.array(self.mesh_near_lumen.vertices))[0]>0):
             
                        # tsdf_time = time.time()
                        # temp_lineset = create_wireframe_lineset_from_mesh(self.mesh_near_lumen) 
                        temp_lineset = self.wireframe_gen.update_from_mesh(self.mesh_near_lumen)
                        # end_time = time.time()
                        # diff_time = end_time - tsdf_time
                        # print("diff time", diff_time)
                 
                        self.mesh_near_lumen_lineset.points = temp_lineset.points
                        self.mesh_near_lumen_lineset.lines = temp_lineset.lines
                        if(self.refine==1):
                            self.mesh_near_lumen_lineset.paint_uniform_color([0.1, 0.7, 0.8])
                            self.vis.update_geometry(self.mesh_near_lumen_lineset)
                        # else:

                        # if(self.figure_mapping==1):

                        #     self.mesh_near_lumen_lineset.paint_uniform_color([0,0,1])

                        if(self.refine ==0):
                            self.vis.update_geometry(self.mesh_near_lumen_lineset)

                            if(self.figure_mapping==1):
                                self.tsdf_surface_pc.points = self.mesh_near_lumen_lineset.points
                                self.tsdf_surface_pc.paint_uniform_color([0,0,1])
                                self.vis.update_geometry(self.tsdf_surface_pc)

                

                # if(three_d_points_far_lumen is not None):
                #     update_tsdf_mesh(self.vis,self.tsdf_volume_far_lumen,self.mesh_far_lumen,three_d_points_far_lumen, extrinsic_matrix,[0,0,1], keep_largest =False)
                    # if(self.gating==1): 
                    #     update_tsdf_mesh(self.vis,self.tsdf_volume_far_lumen,self.mesh_far_lumen,three_d_points_far_lumen, extrinsic_matrix,[0,0,1], keep_largest =False)
                    # else:
                    #   update_tsdf_mesh(self.vis,self.tsdf_volume_far_lumen,self.mesh_far_lumen,three_d_points_far_lumen, extrinsic_matrix,[0,0,1], keep_largest =True)

            
                # dissection flap parameterization no longer needed
                # if(tsdf_map == 1):
                #     if(three_d_points_dissection_flap is not None):
                #         update_tsdf_mesh(self.vis,self.tsdf_volume_dissection_flap,self.mesh_dissection_flap,three_d_points_dissection_flap, extrinsic_matrix,[0,1,0])

        

        
              

        # ------- VOLUMETRIC POINT CLOUD 3D ----- #
        if(self.vpC_map == 1):
            
            
            if(volumetric_three_d_points_near_lumen is not None):
                near_vpC_points=o3d.geometry.PointCloud()
                near_vpC_points.points=o3d.utility.Vector3dVector(volumetric_three_d_points_near_lumen)

                # downsample volumetric point cloud
                near_vpC_points = near_vpC_points.voxel_down_sample(voxel_size=0.0005)

                near_vpC_points.transform(TW_EM @ TEM_C)

                # IVUS BASED DEFORMATION ARAP
                # if((self.live_deformation == 1 and self.registered_ct == 1 and self.dest_frame == 'target1') or (self.live_deformation==1 and self.endoanchor==1)):
                     
                #     #  coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])
                #     #  coordinate_frame.transform(TW_EM @ TEM_C)
                #     #  o3d.visualization.draw_geometries([self.registered_ct_mesh, self.centerline_pc, coordinate_frame, near_pC_points])

                #     #  try:
                        
                #     deformed_mesh = live_deform(self.registered_ct_mesh, self.constraint_indices, self.centerline_pc, TW_EM @ TEM_C, near_vpC_points )
                #     temp_lineset = create_wireframe_lineset_from_mesh(deformed_mesh)
                #     self.registered_ct_lineset.points = temp_lineset.points
                #     self.registered_ct_lineset.lines = temp_lineset.lines
                #     self.vis.update_geometry(self.registered_ct_lineset)

                #     # mapping from coarse deformed mesh to fine deformed mesh nodes for endoscopic view
                #     fine_deformed_vertices = deform_fine_mesh_using_knn(self.registered_ct_mesh, deformed_mesh, self.registered_ct_mesh_2, self.knn_idxs, self.knn_weights, self.coarse_template_vertices, self.fine_template_vertices, self.adjacency_matrix)

                #     # self.registered_ct_mesh_2.vertices = deformed_mesh.vertices
                #     self.registered_ct_mesh_2.vertices = o3d.utility.Vector3dVector(fine_deformed_vertices)
                #     self.registered_ct_mesh_2.compute_vertex_normals()
                #     self.vis2.update_geometry(self.registered_ct_mesh_2)


                #     # deform the green spheres 
                #     # ct_spheres_deformed_points = deform_points_using_knn(self.registered_ct_mesh, deformed_mesh,  self.knn_idxs_spheres, self.knn_weights_spheres, self.coarse_template_vertices, self.ct_centroids, self.adjacency_matrix)
                #     # ct_spheres_temp = get_sphere_cloud(ct_spheres_deformed_points , 0.00225, 20, [0,1,0])
                #     # self.ct_spheres.vertices =  ct_spheres_temp.vertices
                #     # self.vis.update_geometry(self.ct_spheres)
                #     # self.vis2.update_geometry(self.ct_spheres)

                #     # deform the rings instead - don't use make sphere point cloud
                #     # toruses = []
                #     # for deformed_point, normal in (ct_spheres_deformed_points:   
                #     #     torus = create_torus(deformed_point, self., self.major_radius, self.minor_radius, 30)
                #     #     toruses = toruses + torus

                    
                #     ct_spheres_deformed_points = deform_points_using_knn(self.registered_ct_mesh, deformed_mesh,  self.knn_idxs_spheres, self.knn_weights_spheres, self.coarse_template_vertices, self.ct_spheres_points, self.adjacency_matrix)
                #     clean_geom = o3d.geometry.TriangleMesh()
                #     clean_geom.vertices = o3d.utility.Vector3dVector(ct_spheres_deformed_points)
                #     # self.ct_spheres.vertices =  o3d.utility.Vector3dVector(ct_spheres_deformed_points)
                #     self.ct_spheres.vertices = clean_geom.vertices
                #     self.vis.update_geometry(self.ct_spheres)
                #     self.vis2.update_geometry(self.ct_spheres)
                       


                #     # change the blue ivus far pc based on deformation
                #     # test_far_pc = copy.deepcopy(self.far_pc_points)
                #     # far_pc_deformed_points = deform_points_using_knn(self.registered_ct_mesh, deformed_mesh,  self.knn_idxs_far_pc, self.knn_weights_far_pc, self.coarse_template_vertices, self.far_pc_points, self.adjacency_matrix)
                #     # self.far_pc.points =  o3d.utility.Vector3dVector(far_pc_deformed_points)
                #     # self.vis.update_geometry(self.far_pc)


                # if you want all the point cloud points (will be really slow)
                # if(self.extend == 1):
                if(self.extend == 1 and (self.dissection_mapping == 1 or self.pullback==0)):
                    # prevent memory issues by commenting this out
                    # self.volumetric_near_point_cloud.points.extend(near_vpC_points.points)
                    pass
                else:
                    self.volumetric_near_point_cloud.points = near_vpC_points.points


            

                self.volumetric_near_point_cloud.paint_uniform_color([1,0,0])


            
            
            # run this for each component -> volumetric threed points, branch pass, number of branch pixels
            for final_component in final_component_data:

                branch_pass = final_component[0]
                # branch_pass = np.asarray(branch_pass).squeeze()
                # orifice_center_three_d_points = final_component[1]
                volumetric_three_d_points_far_lumen = final_component[2]
                branch_pixels = final_component[3]

                if(volumetric_three_d_points_far_lumen is not None):
        
                    
                
                    far_vpC_points=o3d.geometry.PointCloud()
                    far_vpC_points.points=o3d.utility.Vector3dVector(volumetric_three_d_points_far_lumen)

                    #downsample volumetric point cloud
                    far_vpC_points = far_vpC_points.voxel_down_sample(voxel_size=0.0005)

                    if(self.pullback==1): # downsample again to prevent memory issues
                        far_vpC_points = far_vpC_points.voxel_down_sample(voxel_size=0.0025)

                    far_vpC_points.transform(TW_EM @ TEM_C)


                    # for results evaluation
                    max_branch_pass = 255  # Set based on your application needs
                    # normalized_pass = self.branch_pass / max_branch_pass  # Scale to [0, 1]
                    # print("branch pass for colors vpc" ,branch_pass)
                    normalized_pass = branch_pass / max_branch_pass  # Scale to [0, 1]
                    max_branch_pixels = 2000.0
                    normalized_branch_pixels = branch_pixels / max_branch_pixels
                    duplicated_pass_colors = np.repeat([[normalized_pass, normalized_branch_pixels, 0]], len(far_vpC_points.points), axis=0)

    
                    far_vpC_points.colors = o3d.utility.Vector3dVector(duplicated_pass_colors)


                    # if(self.extend == 1 and self.dissection_mapping == 1):
                    # # if(self.extend == 1):
                    # #     # prevent memory issues by commenting this out
                    #     self.volumetric_far_point_cloud.points.extend(far_vpC_points.points)
                    #     self.volumetric_far_point_cloud.colors.extend(far_vpC_points.colors)
                    # else:
                    #     self.volumetric_far_point_cloud.points = far_vpC_points.points
                    #     self.volumetric_far_point_cloud.colors = far_vpC_points.colors

                    

                    if(self.extend == 1):

                        
                        

                        self.volumetric_far_point_cloud.points.extend(far_vpC_points.points)
                        self.volumetric_far_point_cloud.colors.extend(far_vpC_points.colors)

                        

                    else:
                        self.volumetric_far_point_cloud.points = far_vpC_points.points
                        self.volumetric_far_point_cloud.colors = far_vpC_points.colors

                    if(self.figure_mapping==1):
                        # note this will mess up branch pass
                        self.simple_far_pc.points = copy.deepcopy(self.volumetric_far_point_cloud.points)
                        self.simple_far_pc.paint_uniform_color([0,0,1])
                     
                    


                    # OVERRIDE BRANCH PASS COLOURING - this is needed for clustering later
                    # self.volumetric_far_point_cloud.paint_uniform_color([0,0,1])

            if(self.endoanchor==1):
                if(volumetric_three_d_points_far_lumen is not None):
                    # note this will mess up branch pass
                    far_vpC_points=o3d.geometry.PointCloud()
                    far_vpC_points.points=o3d.utility.Vector3dVector(volumetric_three_d_points_far_lumen)

                    #downsample volumetric point cloud
                    far_vpC_points = far_vpC_points.voxel_down_sample(voxel_size=0.0005)
                    # if(self.endoanchor==1): # downsample again to prevent memory issues
                    #     far_vpC_points = far_vpC_points.voxel_down_sample(voxel_size=0.0025)

                    far_vpC_points.transform(TW_EM @ TEM_C)
                    self.simple_far_pc.points = copy.deepcopy(far_vpC_points.points)
                    self.simple_far_pc.paint_uniform_color([0,0,1])
                else:
                    self.simple_far_pc.points = o3d.utility.Vector3dVector([])

            #alternative to endoanchor for now
            if(self.registered_ct==1 and self.dest_frame == 'target1'):
                if(volumetric_three_d_points_far_lumen is not None):
                    # note this will mess up branch pass
                    far_vpC_points=o3d.geometry.PointCloud()
                    far_vpC_points.points=o3d.utility.Vector3dVector(volumetric_three_d_points_far_lumen)

                    #downsample volumetric point cloud
                    far_vpC_points = far_vpC_points.voxel_down_sample(voxel_size=0.0005)
                    # if(self.endoanchor==1): # downsample again to prevent memory issues
                    #     far_vpC_points = far_vpC_points.voxel_down_sample(voxel_size=0.0025)

                    far_vpC_points.transform(TW_EM @ TEM_C)
                    self.simple_far_pc.points = copy.deepcopy(far_vpC_points.points)
                    self.simple_far_pc.paint_uniform_color([0,0,1])
                else:
                    self.simple_far_pc.points = o3d.utility.Vector3dVector([])


        

            # self.vis.update_geometry(self.point_cloud)
            #JUST DONT VISUALIZE IT
            self.vis.update_geometry(self.volumetric_near_point_cloud)
            self.vis.update_geometry(self.volumetric_far_point_cloud)
            self.vis.update_geometry(self.simple_far_pc)
            

        # boundary point cloud mapping useful only for live deformation
        if(self.bpC_map == 1):
            if(boundary_three_d_points_near_lumen is not None):
                near_bpC_points=o3d.geometry.PointCloud()
                near_bpC_points.points=o3d.utility.Vector3dVector(boundary_three_d_points_near_lumen)

                # downsample volumetric point cloud
                near_bpC_points = near_bpC_points.voxel_down_sample(voxel_size=0.0005)
                near_bpC_points.transform(TW_EM @ TEM_C)

                if(self.extend == 1 ):
                    self.boundary_near_point_cloud.points.extend(near_bpC_points.points)
                else:
                    self.boundary_near_point_cloud.points = near_bpC_points.points
                self.boundary_near_point_cloud.paint_uniform_color([1,0,0])


       

       

        # EM BASED DEFORMATION
        # if(self.live_deformation == 1 and self.registered_ct == 1 and self.dest_frame == 'target2' and volumetric_three_d_points_near_lumen is None and self.endoanchor!=1):

            
                
            
        #     # check if EM transform is outside the mesh before deformation calculations
        #     query_points = np.asarray([TW_EM[:3,3]])
        #     query_points_tensor = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
        #     signed_distance = self.scene.compute_signed_distance(query_points_tensor)
        #     signed_distance_np = signed_distance.numpy()  # Convert to NumPy array

            
        
        #     near_vpC_points = None #this should still work with live deform there's a condition inside its

        #     if(signed_distance_np[0] > 0):
        #         # try:  
        #         deformed_mesh = live_deform(self.registered_ct_mesh, self.constraint_indices, self.centerline_pc, TW_EM, near_vpC_points, self.dest_frame, self.scene, self.free_branch_indices )
                
        #         temp_lineset = create_wireframe_lineset_from_mesh(deformed_mesh)
        #         self.registered_ct_lineset.points = temp_lineset.points
        #         self.registered_ct_lineset.lines = temp_lineset.lines
        #         self.vis.update_geometry(self.registered_ct_lineset)

        #         # mapping from coarse deformed mesh to fine deformed mesh nodes for endoscopic view
        #         fine_deformed_vertices = deform_fine_mesh_using_knn(self.registered_ct_mesh, deformed_mesh, self.registered_ct_mesh_2, self.knn_idxs, self.knn_weights, self.coarse_template_vertices, self.fine_template_vertices, self.adjacency_matrix)

        #         # self.registered_ct_mesh_2.vertices = deformed_mesh.vertices
        #         self.registered_ct_mesh_2.vertices = o3d.utility.Vector3dVector(fine_deformed_vertices)
        #         self.registered_ct_mesh_2.compute_vertex_normals()
        #         self.vis2.update_geometry(self.registered_ct_mesh_2)

        #         # deform the green spheres
        #         ct_spheres_deformed_points = deform_points_using_knn(self.registered_ct_mesh, deformed_mesh,  self.knn_idxs_spheres, self.knn_weights_spheres, self.coarse_template_vertices, self.ct_centroids, self.adjacency_matrix)
        #         ct_spheres_temp = get_sphere_cloud(ct_spheres_deformed_points , 0.00225, 20, [0,1,0])
        #         self.ct_spheres.vertices =  ct_spheres_temp.vertices
        #         self.vis.update_geometry(self.ct_spheres)
        #         self.vis2.update_geometry(self.ct_spheres)
        #         # except:
        #         #     print("EM live mesh deformation not working")
        #         #     pass

                
        #         # change the blue ivus far pc based on deformation
        #         test_far_pc = copy.deepcopy(self.far_pc_points)
        #         far_pc_deformed_points = deform_points_using_knn(self.registered_ct_mesh, deformed_mesh,  self.knn_idxs_far_pc, self.knn_weights_far_pc, self.coarse_template_vertices, test_far_pc, self.adjacency_matrix)
        #         self.far_pc.points =  o3d.utility.Vector3dVector(far_pc_deformed_points)
        #         self.vis.update_geometry(self.far_pc)

        # CARDIAC DEFORMATION

        if(self.cardiac_deformation==1):
    


            t = rospy.Time.now()
            t = t.to_sec()

            if self.ecg_state == 0:
                

                
                most_recent_peak = rospy.Time.now()
                most_recent_peak = most_recent_peak.to_sec()

                

                self.period = most_recent_peak - self.previous_peak
                self.previous_peak = most_recent_peak 
                self.ecg_state = 1
                print("RR interval:", self.period)

            

            
            
            phase_time = t-self.previous_peak    # time into current cycle (would be measured rather than calculated as remainder)
            M = phase_time / self.period


            # cardiac motion model 

            # triangle
            # if(M < 0.75):
            #     alpha = -((4/3)*M) + (1)
            # elif(M >= 0.75):
            #     alpha = 4*(M-0.75)

            # peak bin 8
            # if M <= 0.75:
            #     alpha = 0.5 * (1 + np.cos(np.pi * (M / 0.75)))
            # else:
            #     alpha = 0.5 * (1 - np.cos(np.pi * ((M - 0.75) / 0.25)))

            # swap systole and diastole if you have to!!

            # peak bin 3
            # if M<=0.25:
            #     alpha = 0.5 * (1 - np.cos(np.pi * (M / 0.25)))
            # else:
            #     alpha = 0.5 * (1 + np.cos(np.pi * ((M - 0.25) / 0.75)))

            # if M <= 0.75:
            #     alpha = 0.5 * (1 + np.cos(np.pi * (M / 0.75)))
            # else:
            #     alpha = 0.5 * (1 - np.cos(np.pi * ((M - 0.75) / 0.25)))


            # correct i think if you're going from systole to diastole
            # if M <= 0.75:
            #     alpha = 0.5 * (1 - np.cos(np.pi * (M / 0.75)))
            # else:
            #     alpha = 0.5 * (1 - np.cos(np.pi * ((1 - M) / 0.25)))

            #NEW METHOD
            current_vertices, alpha = compute_current_vertices_cached(self.d_cum,self.systole_locations,self.direction_vectors,phase_time-0.5, self.period, self.PWV)

            
            # when you fixed it we lost some vertices - old method
            # current_vertices = ((alpha*(self.diastole_locations - self.systole_locations)) + self.systole_locations)  # note M = 0 is systole


            self.deformed_mesh.vertices = o3d.utility.Vector3dVector(current_vertices)

            temp_lineset = create_wireframe_lineset_from_mesh(self.deformed_mesh)

            
            self.registered_ct_lineset.points = temp_lineset.points
            self.registered_ct_lineset.lines = temp_lineset.lines

            if(self.vis_red_vessel!=1):
                self.vis.update_geometry(self.registered_ct_lineset)

            # mapping from coarse deformed mesh to fine deformed mesh nodes for endoscopic view
            fine_deformed_vertices = deform_fine_mesh_using_knn(self.registered_ct_mesh, self.deformed_mesh, self.registered_ct_mesh_2, self.knn_idxs, self.knn_weights, self.coarse_template_vertices, self.fine_template_vertices, self.adjacency_matrix)

            # self.registered_ct_mesh_2.vertices = deformed_mesh.vertices
            self.registered_ct_mesh_2.vertices = o3d.utility.Vector3dVector(fine_deformed_vertices)
            # self.registered_ct_mesh_2.compute_vertex_normals()
            self.vis2.update_geometry(self.registered_ct_mesh_2)

            if(self.vis_red_vessel==1):
                self.vis.update_geometry(self.registered_ct_mesh_2)

            # ct_spheres_deformed_points = deform_points_using_knn(self.registered_ct_mesh, self.deformed_mesh,  self.knn_idxs_spheres, self.knn_weights_spheres, self.coarse_template_vertices, self.ct_centroids, self.adjacency_matrix)

            # sphere deformation
            # ct_spheres_deformed_vertices = deform_points_using_knn(self.registered_ct_mesh, self.deformed_mesh,  self.knn_idxs_spheres, self.knn_weights_spheres, self.coarse_template_vertices, self.vertices_before, self.adjacency_matrix)
            # self.ct_spheres.vertices =  o3d.utility.Vector3dVector(ct_spheres_deformed_vertices)
            # self.vis.update_geometry(self.ct_spheres)
            # self.vis2.update_geometry(self.ct_spheres)


            # ring deformation
            ct_spheres_deformed_points = deform_points_using_knn(self.registered_ct_mesh, self.deformed_mesh,  self.knn_idxs_spheres, self.knn_weights_spheres, self.coarse_template_vertices, self.ct_spheres_points, self.adjacency_matrix)

            self.ct_spheres.vertices =  o3d.utility.Vector3dVector(ct_spheres_deformed_points)
            self.vis_2_spheres.vertices = self.ct_spheres.vertices
            self.vis_2_spheres.triangles = self.ct_spheres.triangles
            self.vis_2_spheres.compute_vertex_normals()
            self.vis_2_spheres.vertex_colors = self.ct_spheres.vertex_colors

            self.vis.update_geometry(self.ct_spheres)
            self.vis2.update_geometry(self.vis_2_spheres)
            # self.vis2.update_geometry(self.ct_spheres)



            self.ecg_previous = self.ecg_latest



            
            
        # print("branch pass:", self.branch_pass)

 
        
        
                

        # ----- SIMULATE PROBE VIEW ------- #
        # if(self.registered_ct ==  1 or self.dissection_track == 1):
        if(self.registered_ct ==  1 or self.funsr_only == 1):
            view_control = self.vis2.get_view_control()

            # Set the camera view aligned with the x-axis
            camera_parameters = view_control.convert_to_pinhole_camera_parameters()

    
            

            T = TW_EM 

            # if(self.endoanchor==1):
            #     T = self.most_recent_extrinsic


            

            calib_view_angle = -90
            # calib_view_angle = ... endoanchor 035
            # calib_view_angle = -90 endoanchor 018

            up = -T[:3, 2]
            up = rotate_vector_around_axis( -T[:3, 0], up, calib_view_angle) # new steerable catheter - aptus tourguide

            # if(self.endoanchor==1):
            #     up = -T[:3, 2]          # Z-axis (3rd column)




            front = -T[:3, 0]      # Negative X-axis (1st column) - look forward or back
            lookat = T[:3, 3]      # Translation vector (camera position)
            # translation = np.asarray([-0.002,0,0.004])
            # translation = np.array([-0.002, 0.004, 0.0])  # aptus tourguide
            translation = np.array([-0.004, 0.004, 0.0])  # FEVAR look a little further back
            # if(self.endoanchor==1):
            #     translation = np.array([-0.0, 0.0, 0.0])  # aptus tourguide



            lookat = (T[:3,:3] @ translation) + lookat
            # lookat = lookat # for now
            

            # print("is this changing??", T)

 
            view_control.set_front(front)
            view_control.set_lookat(lookat)
            view_control.set_up(up)

            view_control.set_zoom(0.01)

            
            
            

        # ----- FOLLOW THE PROBE ------- #
        # make this a check box
        # look at centroid of all the data?

        if(self.tsdf_map ==1):
            vertices_of_interest = np.asarray(self.mesh_near_lumen.vertices)
            if(vertices_of_interest is not None):

    



                view_control_1 = self.vis.get_view_control()
                camera_parameters_1 = view_control_1.convert_to_pinhole_camera_parameters()
                if(np.size(vertices_of_interest)>0):
                    centroid = np.mean(vertices_of_interest, axis=0) 
                else:
                    centroid = np.asarray([0,0,0])
                position_tracker = TW_EM[:3,3]
                average_point = (centroid+position_tracker)/2
                lookat = average_point

                
                if(self.once == 0):
                    up = np.array([0, -1, 0])
                    self.view_control_1.set_up(up)
                    self.view_control_1.set_front([0,0,-1])
                    self.once=1
                    self.view_control_1.set_zoom(0.5)

                
                self.view_control_1.set_lookat(lookat)

                
                

            
        if(self.registered_ct ==1 or self.funsr_only == 1):


                # view_control_1 = self.vis.get_view_control()
                camera_parameters_1 = self.view_control_1.convert_to_pinhole_camera_parameters()
                
                # points = np.asarray(self.registered_ct_lineset.points)  # Convert point cloud to numpy array
                # centroid = np.mean(points, axis=0) 

                if(self.funsr_only != 1):
                    centroid = self.registered_centroid

                else:
                    centroid = self.funsr_centroid

                # position_tracker = TW_EM[:3,3]
                self.position_tracker_buffer.append(TW_EM[:3,3])
                average_position_tracker = np.mean(self.position_tracker_buffer, axis=0)
                average_point = ((centroid/4)+(3*average_position_tracker/4))
                lookat = average_point


                

        
                

                # what is the up axis of the vessel - flip this depending on side of table

               

                if(self.once == 0):
                    up = np.array([0, -1, 0])
                    self.view_control_1.set_up(up)
                    self.view_control_1.set_front([0,0,-1])
                    self.once=1
                    self.view_control_1.set_zoom(0.25)

                
                self.view_control_1.set_lookat(lookat)
                
                
    
        # ---- APPEND TO ORIFICE CENTER PC ------ #
        if(self.orifice_center_map == 1 and (self.deeplumen_on ==1 or self.deeplumen_lstm_on==1 or self.deeplumen_slim_on)):

         

            # run this for each component in current image -> orifice_center_three_d_points, branch pass 
            for final_component in final_component_data:
        
                branch_pass = final_component[0]
                orifice_center_three_d_points = final_component[1]
                volumetric_three_d_points_far_lumen = final_component[2]
                branch_pixels = final_component[3]   
                
                if(orifice_center_three_d_points is not None):
                
                    print("current branch pass", self.branch_pass)
                    

                    max_branch_pass = 255  # Set based on your application needs
                    # normalized_pass = self.branch_pass / max_branch_pass  # Scale to [0, 1]
                    normalized_pass = branch_pass / max_branch_pass  # Scale to [0, 1]

            
                    max_branch_pixels = 2000.0
                    normalized_branch_pixels = branch_pixels / max_branch_pixels


                    # Create duplicated colors
                    duplicated_pass_colors = [[normalized_pass, normalized_branch_pixels, 0]]

            
            
                    orifice_center_points = o3d.geometry.PointCloud()
                    orifice_center_points.points = o3d.utility.Vector3dVector(orifice_center_three_d_points)
                    orifice_center_points.transform(TW_EM @ TEM_C)
                
                    orifice_center_points.colors = o3d.utility.Vector3dVector(duplicated_pass_colors)

                    

                    
                    

                    if(self.extend==1):
                    
                        
                        self.orifice_center_point_cloud.points.extend(orifice_center_points.points)
                        self.orifice_center_point_cloud.colors.extend(orifice_center_points.colors)
                        

                        # this was expensive
                        # orifice_center_spheres_temp = get_sphere_cloud(np.asarray(self.orifice_center_point_cloud.points), 0.0025, 12, [0,0,1])
                        # self.orifice_center_spheres.vertices = orifice_center_spheres_temp.vertices
                        # self.orifice_center_spheres.triangles = orifice_center_spheres_temp.triangles
                        # self.orifice_center_spheres.compute_vertex_normals()
                        # self.orifice_center_spheres.paint_uniform_color([0,0,1])
                    
                    
                    else:
                        self.orifice_center_point_cloud.points = orifice_center_points.points
                        self.orifice_center_point_cloud.colors = orifice_center_points.colors

                        orifice_center_spheres_temp = get_sphere_cloud(np.asarray(orifice_center_points.points), 0.0025, 12, [0,0,1])
                        self.orifice_center_spheres.vertices = orifice_center_spheres_temp.vertices
                        self.orifice_center_spheres.triangles = orifice_center_spheres_temp.triangles
                        self.orifice_center_spheres.compute_vertex_normals()
                        # print("just one sphere", np.asarray(self.orifice_center_spheres.vertices))

        
            
            self.vis.update_geometry(self.orifice_center_point_cloud)  
            self.vis.update_geometry(self.orifice_center_spheres)


            
        # ----- SIMULATE DEVICE DEPLOYMENT (SLIDING) ------- #
        if(self.evar_slide_sim==1):

            self.evar_graft = slide_device_to_pose(self.evar_graft, self.transformed_centreline_pc, self.no_graft_points, extrinsic_matrix, 0.001)
                
            self.vis.update_geometry(self.evar_graft)


        # for updating graft every iteration
        if(self.evar_loft_sim==1):

            # self.evar_graft = slide_device_to_pose(self.evar_graft, self.transformed_centreline_pc, self.no_graft_points, extrinsic_matrix, 0.001)

            # STATIONARY GRAFT (DEPLOYED)
            # extrinsic_matrix_temp = np.eye(4)
            # extrinsic_matrix_temp[:3,3] = self.aortic_centreline[225, :]

            # print("extrinsic", extrinsic_matrix)

            # extrinsic_matrix_temp = np.asarray([[-0.29256042, -0.08216749, -0.95271029,  0.19205677],
            # [-0.94842493,  0.15210554,  0.27812596, -0.04631414],
            # [ 0.1220596,   0.98494285, -0.12242975, -0.06255438],
            # [ 0.,          0.,          0.,          1.        ]])

            # current_evar = predict_deploy(extrinsic_matrix_temp, self.aortic_centreline,self.lofted_cylinder,self.strut_geometry,self.strut_distances,self.evar_length, self.centreline_transforms, self.GD_centreline,self.evar_radius, self.fen_distances, self.fen_angles)
            

            # EVERY ITERATION
            
            current_evar = predict_deploy(extrinsic_matrix, self.aortic_centreline,self.lofted_cylinder,self.strut_geometry,self.strut_distances,self.evar_length, self.centreline_transforms, self.GD_centreline,self.evar_radius, self.fen_distances, self.fen_angles)
            

            self.evar_graft.vertices = current_evar.vertices
            self.evar_graft.triangles = current_evar.triangles
            self.evar_graft.vertex_colors = current_evar.vertex_colors

            self.evar_graft.compute_vertex_normals()
            
            self.vis.update_geometry(self.evar_graft)

            # FOR FEVAR NAVIGATION
            self.vis2.update_geometry(self.evar_graft)

            
            

        
        # ------ TRACKER FRAMES ------ #
        # catheter
        T_catheter = self.get_catheter_transform(TW_EM @ TEM_C)
        self.catheter.transform(get_transform_inverse(self.previous_catheter_transform))
        self.catheter.transform(T_catheter)
        self.previous_catheter_transform = T_catheter

        

        # tracker
        T_tracker = self.get_catheter_transform(TW_EM)
        self.tracker.transform(get_transform_inverse(self.previous_tracker_transform))
        self.tracker.transform(T_tracker)
        self.previous_tracker_transform = T_tracker
        self.vis.update_geometry(self.tracker)


        if(self.registered_ct ==1 or self.funsr_only==1):
            self.tracker_frame.transform(get_transform_inverse(self.previous_transform))
            self.tracker_frame.transform(TW_EM)
            self.previous_transform=TW_EM
            self.vis.update_geometry(self.tracker_frame)
        
        

        # if(self.registered_ct!=1):
            
            
        

        # print("appending frame!")
        self.us_frame.transform(get_transform_inverse(self.previous_transform_us))
        self.us_frame.transform(TEM_C)
        self.us_frame.transform(TW_EM)
        self.vis.update_geometry(self.us_frame)
        self.vis.update_geometry(self.catheter)

        self.previous_transform_us=TW_EM @ TEM_C

    

        if(self.registered_ct ==1 or self.funsr_only == 1):
            self.vis2.update_geometry(self.tracker)
            if(self.endoanchor==1):
                self.vis2.update_geometry(self.catheter)
            
        
            if(self.guidewire==1):

                

                T_guidewire = self.get_catheter_transform(TW_EM @ self.TEM_GW)
                self.guidewire_cylinder.transform(get_transform_inverse(self.previous_guidewire_transform))
                self.guidewire_cylinder.transform(T_guidewire)
                self.previous_guidewire_transform = T_guidewire
                self.vis.update_geometry(self.guidewire_cylinder)
                guidewire_pointcloud_temp = copy.deepcopy(self.guidewire_pointcloud_base)
                guidewire_pointcloud_temp.transform(T_guidewire)
                query_points = np.asarray(guidewire_pointcloud_temp.points)

                if(self.funsr_only != 1):
                    query_points_tensor = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
                    signed_distance = self.scene.compute_signed_distance(query_points_tensor)
                    signed_distance_np = signed_distance.numpy()  # Convert to NumPy array
                    arg_points_inside = np.argwhere(signed_distance_np < 0).flatten()
                    self.guidewire_pointcloud.points = o3d.utility.Vector3dVector(query_points[arg_points_inside, :])

                if(self.funsr_only==1):
                    self.guidewire_pointcloud.points = o3d.utility.Vector3dVector(query_points)

                # self.guidewire_pointcloud.points = o3d.utility.Vector3dVector(query_points)
                self.guidewire_pointcloud.paint_uniform_color([0,1,0])
                self.vis.update_geometry(self.guidewire_pointcloud)

            if(self.steerable == 1 and self.dest_frame=='target2'):

                ref_frame = 'ascension_origin'
                steerable_frame = 'target1'

                
                

                try:
                    # Lookup transform
                    TW_EM_3 = self.tf_buffer.lookup_transform(ref_frame, steerable_frame, self.transform_time)
                except (rospy.ROSException, tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    rospy.logwarn("Failed to lookup transform")
                    TW_EM_3 = None

                

                TW_EM_3 = transform_stamped_to_matrix(TW_EM_3)

                # - BUDGE BASE FRAME INSIDE MESH ---- #
                # query_points_tensor = o3d.core.Tensor([TW_EM_3[:3,3]], dtype=o3d.core.Dtype.Float32)  # shape (1,3)
                # signed_distance = self.scene.compute_signed_distance(query_points_tensor)
                # signed_distance_np = signed_distance.numpy()  # Convert to NumPy array
                # signed_distance_np = signed_distance_np.squeeze()

                # if(signed_distance_np > 0):

                #     result = self.scene.compute_closest_points(query_points_tensor)
                #     TW_EM_3[:3,3] = result['points'].numpy()[0]   
                    


               

                # # prerotates both frames 90 degrees clockwise
                # theta = np.pi / 2  # -90 degrees in radians
                # R_y = np.array([
                #     [np.cos(theta), 0, np.sin(theta)],
                #     [0,             1, 0],
                #     [-np.sin(theta), 0, np.cos(theta)]
                # ], dtype=float)

                # T_y = np.eye(4)
                # T_y[:3, :3] = R_y

                # # base_rotated_y = TW_EM_3 @ T_y
                # base_rotated_y = TW_EM_3 # no prerotation for new catheter
                # # tip_rotated_y = TW_EM @ T_y
                # tip_rotated_y = TW_EM


                # theta = -(np.pi / 180)*12
                # c, s = np.cos(theta), np.sin(theta)
                # R_x = np.array([
                #     [1, 0, 0],
                #     [0, c, -s],
                #     [0, s,  c]
                # ])

                # T_x = np.eye(4)
                # T_x[:3, :3] = R_x

                # for calibration in def/bending_segment_transform_test.py
                # print("base", TW_EM_3)
                # print("tip pre rotate", TW_EM)
                

                # theta = -(np.pi / 180)*12
                theta = self.bend_plane_angle
                c, s = np.cos(theta), np.sin(theta)
                R_x = np.array([
                    [1, 0, 0],
                    [0, c, -s],
                    [0, s,  c]
                ])

                T_x = np.eye(4)
                T_x[:3, :3] = R_x

                base_rotated_y = TW_EM_3 
                tip_rotated_y = TW_EM @ T_x

                # if(self.endoanchor==1):
                #     tip_rotated_y = self.most_recent_extrinsic @ T_x


                

                


                # new version in x y plane (Oct 25)
                arc_points, transform_needed = compute_arc_backwards_from_tip_and_base(tip_rotated_y, base_rotated_y, self.steerable_arc_length)

                if(self.endoanchor==1):
                    arc_points = arc_points[:-1, :] #chop off a little to expose IVUS

                # new_tube = create_tube_mesh_catheter_fast(arc_points, radius=self.catheter_radius, segments=7)

                # new_tube = create_tube_mesh_catheter_fast(arc_points, radius=self.catheter_radius, segments=15)

                new_tube = create_tube_mesh_catheter_fast(arc_points, radius=self.catheter_radius, segments=25)

                
                self.bending_segment.vertices = new_tube.vertices
                self.bending_segment.triangles = new_tube.triangles
                self.bending_segment.paint_uniform_color(self.bending_segment_color)
                self.bending_segment.compute_vertex_normals()
                self.vis.update_geometry(self.bending_segment)

                flip_mesh_orientation(self.bending_segment) 
                self.vis2.update_geometry(self.bending_segment)

                

                

                self.catheter_base_frame.transform(get_transform_inverse(self.previous_catheter_base))
                self.catheter_base_frame.transform(TW_EM_3)
                self.vis.update_geometry(self.catheter_base_frame)

                # o3d.visualization.draw_geometries([self.catheter_base_frame, self.tracker_frame])


                self.previous_catheter_base=TW_EM_3

                # find the shaft spline

                aortic_centreline = np.asarray(self.centerline_pc.points)                


                p0 = aortic_centreline[-1, :]
                p1 = arc_points[0,:]
                
                t0 = aortic_centreline[-5, :] - aortic_centreline[-1,:]
                t0 = t0 / np.linalg.norm(t0)

                t1 = arc_points[1,:] - arc_points[0,:]
                t1 = t1 / np.linalg.norm(t1)


                # arc_points = hermite_segment(p0, t0, p1, t1, n_points=25, tangents_are_directions=True, tension=1.0)

                
                arc_points = quintic_minimum_jerk_segment_fast(p0, t0, p1, t1, n_points=20, tangents_are_directions=True, scale=0.25)

                # working version
                # arc_points = deform_centerline_inside_lumen(arc_points, self.scene, self.registered_ct_mesh)

          

                shaft = create_tube_mesh_catheter_fast(arc_points, radius=self.catheter_radius, segments=8)

                self.catheter_shaft.vertices = shaft.vertices
                self.catheter_shaft.triangles = shaft.triangles
                self.catheter_shaft.paint_uniform_color([0,0,1])
                self.catheter_shaft.compute_vertex_normals()
                self.vis.update_geometry(self.catheter_shaft)


                
        self.vis2.poll_events()
        self.vis2.update_renderer()

        self.vis.poll_events()
        self.vis.update_renderer()

        # stop_entire = time.time()
        # diff_entire = stop_entire - self.start_entire
        # print("entire loop time (different rospy difference time)", diff_entire)

        

    
            
        
  




if __name__ == '__main__':
    try:
        rospy.init_node('open3d_visualizer')
        pc_updater = PointCloudUpdater()        
        # rospy.on_shutdown(pc_updater.save_image_and_transform_data)
        rospy.spin()
        
    except rospy.ROSInterruptException:
        self.vis.destroy_window()
        pass
