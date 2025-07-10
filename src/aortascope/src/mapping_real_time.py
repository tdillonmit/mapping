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
        self.vis.add_geometry(self.volumetric_near_point_cloud)
        self.vis.add_geometry(self.volumetric_far_point_cloud)
        self.vis.add_geometry(self.boundary_near_point_cloud)
        # self.vis.add_geometry(self.point_cloud)
        self.vis.add_geometry(self.orifice_center_point_cloud)



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

        self.catheter = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0015, height=0.01)
        self.catheter.compute_vertex_normals()
        self.catheter.paint_uniform_color([0,0,1])

        self.guidewire_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.000225, height=0.004)
        self.guidewire_cylinder.compute_vertex_normals()
        self.guidewire_cylinder.paint_uniform_color([0,1,0])
        self.guidewire_pointcloud_base = o3d.geometry.PointCloud()
        n = 30
        z = np.linspace(0, 0.05, n)
        x = np.zeros_like(z)
        y = np.zeros_like(z)
        base_points = np.stack((x, y, z), axis=1)
        self.guidewire_pointcloud_base.points = o3d.utility.Vector3dVector(base_points)
        self.guidewire_pointcloud_base.paint_uniform_color([0,1,0])
        self.guidewire_pointcloud = o3d.geometry.PointCloud()
        
        self.previous_transform=np.eye(4)
        self.previous_transform_1=np.eye(4)
        self.previous_transform_us=np.eye(4)
        self.previous_catheter_transform=np.eye(4)
        self.previous_tracker_transform=np.eye(4)
        self.previous_guidewire_transform=np.eye(4)

        self.vis.add_geometry(self.catheter)
        self.vis.add_geometry(self.guidewire_cylinder)
        self.vis.add_geometry(self.tracker)
        self.vis.add_geometry(self.us_frame)
        self.vis.add_geometry(self.tracker_frame)
        self.vis.add_geometry(self.baseframe)
        self.vis.add_geometry(self.guidewire_pointcloud)

    
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
        self.lstm_length = 5

        self.centroid_buffer = deque(maxlen=self.buffer_size)
        self.delta_buffer_x = deque(maxlen=self.buffer_size)
        self.delta_buffer_y = deque(maxlen=self.buffer_size)
        self.delta_buffer_z = deque(maxlen=self.buffer_size)
        self.position_buffer = deque(maxlen=self.buffer_size)
        self.grayscale_buffer = deque(maxlen=self.lstm_length)
        self.mask_1_buffer = deque(maxlen=self.buffer_size)
        self.mask_2_buffer = deque(maxlen=self.buffer_size)
        self.orifice_angles = deque(maxlen=5)

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
        self.voxel_size = 0.002
        self.sdf_trunc = 3 * self.voxel_size
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
            self.vis.add_geometry(self.mesh_near_lumen_lineset)

        

        

    
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
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        
       

        if(self.gating ==1):
            # self.lock = threading.Lock()  # A lock to ensure thread safety
            self.ecg_sub = rospy.Subscriber('ecg', Int32, self.ecg_callback)
            
        self.rgb_image_pub = rospy.Publisher('/rgb_image', Image, queue_size=1)

        # self.pullback_pub = rospy.Publisher('/pullback', Int32, queue_size=1)
        # pullback = rospy.get_param('pullback', 0)
        # self.pullback_pub.publish(pullback)
        # print("pullback check", pullback)

        self.dest_frame = 'target1'


        # initialize view above the phantom
        view_control_1 = self.vis.get_view_control()

        view_control_1.set_up([0,-1,0])

        view_control_1.set_front([0,0,-1])


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
        rospy.Subscriber('/switch_probe', Bool, self.switch_probe_cb)
        rospy.Subscriber('/refine_started', Bool, self.refine_started_cb)
        rospy.Subscriber('/refine_done', Bool, self.refine_done_cb)
        rospy.Subscriber('/sim_device', Bool, self.sim_device_cb)
        rospy.Subscriber('/shutdown', Bool, self.shutdown_cb)
        rospy.Subscriber('/pullback', Int32, self.pullback_cb)
        rospy.Subscriber('/replay', Bool, self.replay_cb)

        self.start_record = False
        self.save_data = False
        self.gate = False
        self.funsr_start = False
        self.funsr_complete = False
        self.registration_start = False
        self.reg_complete = False
        self.pause = False
        self.switch_probe = False
        self.refine_start = False
        self.refine_complete = False
        self.sim_device = False
        self.shutdown = False
        self.pullback = 0  # Int32
        self.replay = False

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
    def switch_probe_cb(self, msg):        self.switch_probe = msg.data
    def refine_started_cb(self, msg):      self.refine_start = msg.data
    def refine_done_cb(self, msg):         self.refine_complete = msg.data
    def sim_device_cb(self, msg):          self.sim_device = msg.data
    def shutdown_cb(self, msg):            self.shutdown = msg.data
    def pullback_cb(self, msg):            self.pullback = msg.data
    def replay_cb(self, msg):              self.replay = msg.data
    

    def initialize_deeplumen_model(self):
        
        if(self.deeplumen_on == 1):

            

            DRN_inputs_3,DRN_outputs_3 = get_DRN_network()

            
            model = tf.keras.Model(inputs=DRN_inputs_3, outputs=DRN_outputs_3)

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

        self.record_poses =0

        # healthy first return mapping for reference
        self.tsdf_map = config_yaml['tsdf_map']

        self.deeplumen_on = config_yaml['deeplumen_on']

        self.deeplumen_lstm_on = config_yaml['deeplumen_lstm_on']

        self.deeplumen_valve_on = config_yaml['deeplumen_valve_on']

        self.model_path  = config_yaml['model_path']

        self.model_path_cusps  = config_yaml['model_path_cusps']

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

        self.guidewire = config_yaml['guidewire']

        self.double_display = config_yaml['double_display']

        self.test_image = config_yaml['test_image']

        self.test_transform = config_yaml['test_transform']

        self.machine = config_yaml['machine']

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

        # change directory to gated folder
        self.write_folder = self.write_folder + '/gated'
        create_folder(self.write_folder )
        
        print("new folder after gating is:", self.write_folder )
        


    def replay(self):

        self.test_image = 0
        self.test_transform = 0

        # NEED TO LOAD RELEVANT CALIBRATION FILE!!!!! AND SAVE RELEVANT CALIBRATION FILE OTHERWISE YOULL LOSE DATASETS


        # rospy.set_param('replay', 0)

        # clear view of any geometries
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

        print("pulling from folder",self.write_folder  )
        print("number of loaded images:", len(grayscale_images))


        starting_index=0
        ending_index=len(em_transforms)-1

        # if(self.centre_data == 1):
        #     average_transform = get_transform_data_center(em_transforms)

        for i in np.arange(starting_index,ending_index):

            print(f"image index: {i}")

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
                self.append_image_transform_pair(TW_EM, grayscale_image)

            except KeyboardInterrupt:
                print("Ctrl+C detected, stopping visualizer...")
                self.stop()  #fake function that throws an exception

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

    

    def funsr_started(self):

        rospy.set_param('funsr_started', 0)
        self.write_folder = rospy.get_param('dataset', 0)
        self.deeplumen_on = 0
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
        # rospy.set_param('pullback', 0)
        # print("pullback device stopped")

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
        

    
    def funsr_done(self):

        
        self.write_folder = rospy.get_param('dataset', 0)
        if(self.write_folder ==0):
            self.write_folder = self.prompt_for_folder()
            
        while(self.write_folder ==None):
            self.write_folder = self.prompt_for_folder()
            
        # superimpose the completed surface geometry
        # ivus_funsr_mesh = o3d.io.read_triangle_mesh(self.write_folder + '/full_lumen_mesh.ply')
        # ivus_funsr_lineset = create_wireframe_lineset_from_mesh(ivus_funsr_mesh)
        # ivus_funsr_lineset.paint_uniform_color([0,0,0])
        # self.vis.add_geometry(ivus_funsr_lineset)
        # self.vis2.add_geometry(ivus_funsr_mesh)

        self.deeplumen_on = 0
        self.deeplumen_lstm_on = 0

        if not hasattr(self, 'model'):
            self.initialize_deeplumen_model()

      

    def load_registetered_ct(self):

        no_reg=0

        print("loading registration lineset")
        self.registered_ct_lineset = o3d.io.read_line_set(self.write_folder+ '' + '/final_registration.ply')
        print("loading registration mesh")
        self.registered_ct_mesh = o3d.io.read_triangle_mesh(self.write_folder + '/final_registration_mesh.ply')

        if(self.refine==1):
            self.registered_ct_lineset = o3d.io.read_line_set(self.write_folder+ '' + '/final_registration_refine.ply')
            print("loading registration mesh")
            self.registered_ct_mesh = o3d.io.read_triangle_mesh(self.write_folder + '/final_registration_mesh_refine.ply')

        # TSDF LOAD MODULES
        if self.registered_ct_mesh is None or len(self.registered_ct_mesh.vertices) == 0 or len(self.registered_ct_mesh.triangles) == 0:

            self.ct_centroids = np.load(self.write_folder + '/ivus_centroids.npy')
            self.ct_spheres = get_sphere_cloud(self.ct_centroids, 0.004, 20, [0,1,0])

            self.registered_ct_lineset.paint_uniform_color([0,0,0])

            self.registered_ct_mesh = o3d.io.read_triangle_mesh(self.write_folder + '/tsdf_mesh_near_lumen.ply')
            self.registered_ct_mesh_2 = copy.deepcopy(self.registered_ct_mesh)
            self.registered_ct_mesh_2.compute_vertex_normals()

            # DELETED FOR FEVAR
            self.vis2.add_geometry(self.ct_spheres)
            self.registered_ct_mesh_2.paint_uniform_color([1,0,0])
            self.vis2.add_geometry(self.registered_ct_mesh_2)
            self.vis2.add_geometry(self.tracker)

            
            # self.vis.remove_geometry(self.catheter)

            self.registered_ct_lineset = create_wireframe_lineset_from_mesh(self.registered_ct_mesh)
            self.vis.add_geometry(self.registered_ct_lineset)

            # DELETED FOR FEVAR
            # self.vis.add_geometry(self.ct_spheres)

            print("LOADING TSDF MESH ONLY!!")

    
            # Get current camera parameters
            view_control_1 = self.vis.get_view_control()

            view_control_1.set_up([0,-1,0])

            view_control_1.set_front([0,0,-1])
            

            view_control_1.set_zoom(0.25)


            
            return

        ct_centroid_pc = o3d.io.read_point_cloud(self.write_folder + '/side_branch_centrelines.ply')
        if(self.refine==1):
            ct_centroid_pc = o3d.io.read_point_cloud(self.write_folder + '/side_branch_centrelines_refine.ply')

        self.centerline_pc = o3d.io.read_point_cloud(self.write_folder + '/centerline_pc.ply')
        if(self.refine==1):
            self.centerline_pc = o3d.io.read_point_cloud(self.write_folder + '/centerline_pc_refine.ply')

        


        # self.ct_centroids = np.load(self.write_folder + '/ct_centroids.npy')
        # NOte that ivus centroids have been used here!!!
        self.ct_centroids = np.load(self.write_folder + '/ivus_centroids.npy')


        


        self.registered_ct_mesh.remove_unreferenced_vertices()

        self.registered_ct_mesh_2 = copy.deepcopy(self.registered_ct_mesh)

        avg_edge_length = 999
        desired_edge_length = 0.002

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
        self.knn_idxs, self.knn_weights = precompute_knn_mapping(self.registered_ct_mesh, self.registered_ct_mesh_2, k=5)
        self.coarse_template_vertices = copy.deepcopy(np.asarray(self.registered_ct_mesh.vertices))
        self.fine_template_vertices = copy.deepcopy(np.asarray(self.registered_ct_mesh_2.vertices))
        self.adjacency_matrix = build_adjacency_matrix(self.registered_ct_mesh_2)
        # visualize_knn_mapping(self.registered_ct_mesh_2, self.registered_ct_mesh, self.knn_idxs, sample_indices=[0, 50, 100])
        print("computed coarse to fine mesh node mapping")


        self.constraint_locations = np.asarray(ct_centroid_pc.points)
        
        
        # self.ct_spheres = get_sphere_cloud(self.ct_centroids, 0.00225, 20, [0,1,0])
        self.ct_spheres = get_sphere_cloud(self.ct_centroids, 0.004, 20, [0,1,0])
        self.knn_idxs_spheres, self.knn_weights_spheres = precompute_knn_mapping(self.registered_ct_mesh, self.ct_centroids, k=5)

    

        # # SMOOTHING MESH AS IMPORTED!! - not anymore
        # self.registered_ct_mesh = self.registered_ct_mesh.filter_smooth_taubin(number_of_iterations=10)
        # self.registered_ct_lineset = create_wireframe_lineset_from_mesh(self.registered_ct_mesh)
        # #END

        self.registered_ct_lineset.paint_uniform_color([0,0,0])

        # self.registered_ct_mesh_2 = copy.deepcopy(self.registered_ct_mesh)
        
        self.registered_ct_mesh_2.compute_vertex_normals()

        # DELETED FOR FEVAR
        self.vis2.add_geometry(self.ct_spheres)

        self.vis2.add_geometry(self.registered_ct_mesh_2)
        self.vis2.add_geometry(self.tracker)

        
        # self.vis.remove_geometry(self.catheter)

        self.vis.add_geometry(self.registered_ct_lineset)

        # DELETED FOR FEVAR
        # self.vis.add_geometry(self.ct_spheres)


        
        print("registered the ct from non rigid icp!!")

        self.scene = o3d.t.geometry.RaycastingScene()
        self.registered_ct_mesh_copy = copy.deepcopy(self.registered_ct_mesh)
        self.registered_ct_mesh_copy = o3d.t.geometry.TriangleMesh.from_legacy(self.registered_ct_mesh_copy)
        _ = self.scene.add_triangles(self.registered_ct_mesh_copy)  # we do not need the geometry ID for mesh

        # Get current camera parameters
        view_control_1 = self.vis.get_view_control()

        view_control_1.set_up([0,-1,0])

        view_control_1.set_front([0,0,-1])
        

        view_control_1.set_zoom(0.25)


        points = np.asarray(self.registered_ct_lineset.points)  # Convert point cloud to numpy array
        self.registered_centroid = np.mean(points, axis=0) 

        

        
        
    
      
    
    def tracking(self):

        self.write_folder = rospy.get_param('dataset', 0) 

        self.deeplumen_on = 0
        self.deeplumen_lstm_on = 0

        if(self.write_folder ==0):
            self.write_folder = self.prompt_for_folder()
            
        while(self.write_folder ==None):
            self.write_folder = self.prompt_for_folder()

        self.extend = 0
        self.record = 0

        if(self.dest_frame == 'target1'):
            print("assuming integrated catheter")
            self.vpC_map = 1
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
        # try:
        #     self.vis.remove_geometry(self.us_frame)
        # except:
        #     print("no us frame present")
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
            self.pose_batch=[]
            self.image_tags=[]
           

        # GUI SPECIFIC
        if hasattr(self, 'registered_ct') and self.registered_ct is not None:
            print("registered_ct is", self.registered_ct)
        else:
            print("registered_ct is not defined or is None")
        if(self.registered_ct ==1):

            self.load_registetered_ct()
            

        if(self.live_deformation == 1):
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
            
            # visualize the constraints if desired
            # test_pc= o3d.geometry.PointCloud() 
            # test_pc.points = o3d.utility.Vector3dVector(np.asarray(self.registered_ct_mesh.vertices)[self.constraint_indices,:])
            # constraint_seeds = get_sphere_cloud(self.constraint_locations, 0.0025, 12, [0,0,1])
            # o3d.visualization.draw_geometries([test_pc, self.registered_ct_lineset, constraint_seeds])


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

        if(self.deeplumen_lstm_on ==1):
    

            if not hasattr(self, 'model'):
                self.initialize_deeplumen_model()

  

            



        

    


    def switch_probe_view(self):
        print("switched_frames")
        # just switch the transform you fetch
        if(self.dest_frame == 'target1'):

            print("switching")
            self.dest_frame = 'target2'

            # turn off ML components
            self.deeplumen_on = 0
            self.deeplumen_lstm_on = 0

            self.vis.remove_geometry(self.catheter)
            self.vis.remove_geometry(self.tracker_frame)
            self.vis.remove_geometry(self.volumetric_near_point_cloud)
         

        elif (self.dest_frame == 'target2'):
            self.dest_frame = 'target1'

       


  

    def refine_record(self):


        self.refine=1
        self.extend=1
        self.tsdf_map =1
        self.vpC_map = 1
        self.orifice_center_map = 1

        

        

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
        self.vis.add_geometry(self.volumetric_near_point_cloud)
        self.vis.add_geometry(self.volumetric_far_point_cloud)

        

        # self.vis.add_geometry(self.orifice_center_point_cloud)

        self.orifice_center_spheres = o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.orifice_center_spheres)

        
        

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

        print("saved refine data!")

    

        while(self.refine_done ==0):

            # refine_done = rospy.get_param('refinement_computed', 0)
            print("waiting for refinement to compute...")
            time.sleep(1)

        print("refinement complete (check if result is reasonable)")

        # initialize tracking type code
        self.refine=1

        # get rid of old geometry
        self.vis2.remove_geometry(self.registered_ct_mesh_2)
        self.vis.remove_geometry(self.registered_ct_lineset)
        self.tracking()

        self.tsdf_map = 0
        # self.refine = 0 # want to call the refined centerline for device simulation
        self.extend = 0

    def simulate_device(self):

        # do all initialization here
        self.evar_slide_sim = 0
        self.evar_loft_sim = 1
        self.tavr_sim = 0

        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        self.deeplumen_on = 0
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
            self.vis.remove_geometry(self.volumetric_near_point_cloud)
            self.vis.remove_geometry(self.volumetric_far_point_cloud)

            

        if(self.tavr_sim == 1):
            self.evar_graft = o3d.io.read_triangle_mesh('/home/tdillon/Downloads/sapient_stent_frame.stl')
            # fill in rest here

        if(self.evar_loft_sim):

            # precomputed
            # self.evar_radius = 0.014
            self.evar_radius = 0.012

            # deployed inside FEVAR
            self.evar_radius = 0.009

            self.evar_length = 0.14
            amplitude = 0.005
            num_struts = 5
            axial_spacing = 0.015
            self.no_graft_points = 12

            self.load_registetered_ct()
            
            self.aortic_centreline = np.asarray(self.centerline_pc.points)
            
            x_points, y_points, z_points = fit_3D_bspline(self.aortic_centreline, 0.0001)
            self.aortic_centreline = np.column_stack((x_points,y_points,z_points))
            self.lofted_cylinder, self.strut_geometry, self.strut_distances, self.aortic_centreline, self.centreline_transforms, self.GD_centreline = get_evar_template_geometries(self.aortic_centreline, self.evar_radius, self.evar_length, amplitude, num_struts, axial_spacing, self.no_graft_points)

            # for fenestrated evar (FEVAR)
            # self.fen_distances =np.asarray([0.1,0.05,0.1])
            self.fen_distances =np.asarray([0.044, 0.056, 0.1155, 0.07])
            self.fen_angles = np.asarray([2*3.24, 2*3.24, 3.12/2, 2*2.7])

            self.lofted_cylinder = o3d.t.geometry.TriangleMesh.from_legacy(self.lofted_cylinder)
            self.evar_graft = o3d.geometry.TriangleMesh()
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

        if(self.record==1):
            self.ecg_times.append( ecg_timestamp_in_seconds)
            self.ecg_signal.append(ecg_value)

        
    def image_callback(self, msg):

        now = rospy.Time.now()
        delta = now - self.last_image_time  
        # print("rospy difference time", delta.to_sec())
        if delta < rospy.Duration(0.02):  # max ~50 Hz
            print("throttling to cool CPU!")
            return
        self.last_image_time = now

        # self.start_entire = time.time()

        

        # self.replay_data = rospy.get_param('replay', 0 )
        if(self.replay_data ==1):
            self.replay()
            self.replay_data = 0
            # rospy.set_param('replay', 0)

        # print("self.replay data", self.replay_data)
        # if(self.replay_data==1):
        #     # print("not called")
        #     return
        

        start_total_time = time.time()
        # this is the shortest number of lines it takes to grab the transform once image arrives
        # transform_time = rospy.Time(0) #get the most recent transform
        transform_time = msg.header.stamp #get the most recent transform

        if(self.test_transform ==1): 
            transform_time = rospy.Time(0)
        
        # Assuming you have the frame IDs for your transform
        ref_frame = 'ascension_origin'
        dest_frame = self.dest_frame



        try:
            # Lookup transform
            TW_EM = self.tf_buffer.lookup_transform(ref_frame, dest_frame, transform_time)
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

       
        TW_EM=transform_stamped_to_matrix(TW_EM)

        # prune fast probe moves
        current_time_in_sec = transform_time.to_sec()
        self.smoothed_linear_speed = update_linear_speed_ema(TW_EM, current_time_in_sec, self.previous_transform_ema, self.previous_time_in_sec, self.smoothed_linear_speed)
        self.previous_time_in_sec = current_time_in_sec
        self.previous_transform_ema = TW_EM

        print("smoothed linear speed:",self.smoothed_linear_speed)
        if self.smoothed_linear_speed > 0.15:  # max ~50 Hz
            print("probe speed too fast! omit image")
            return


        # now get the original image's timestamp
        timestamp_secs = msg.header.stamp.secs
        timestamp_nsecs = msg.header.stamp.nsecs
        image_timestamp_in_seconds = timestamp_secs + (timestamp_nsecs * 1e-9)

        # Create a rospy.Time object using the truncated timestamp information
        timestamp = rospy.Time(secs=timestamp_secs, nsecs=timestamp_nsecs)

        
        # Assuming RGB format
        rgb_image_data = np.frombuffer(msg.data, dtype=np.uint8)

        

        rgb_image = rgb_image_data.reshape((self.image_height, self.image_width, 3))


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
            if(len(self.image_batch)>500):
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

        # first return segmentation
        if(self.deeplumen_on == 0 and self.deeplumen_lstm_on == 0 and self.dest_frame== 'target1'):

            relevant_pixels=first_return_segmentation(grayscale_image,threshold, crop_index,self.gridlines)
            relevant_pixels=np.asarray(relevant_pixels).squeeze()

            # EM TRACKER ONLY MODE
            # relevant_pixels = np.asarray([[int(centre_x)-10, int(centre_y)], [int(centre_x)+10, int(centre_y)], [int(centre_x), int(centre_y)], [int(centre_x), int(centre_y)+2], [int(centre_x)+1, int(centre_y)]])
            
            
            # ellipse fitting to first return
            ellipse_model = cv2.fitEllipse(relevant_pixels[:,[1,0]].astype(np.float32))  
            ellipse_contour= cv2.ellipse2Poly((int(ellipse_model[0][0]), int(ellipse_model[0][1])),
                                            (int(ellipse_model[1][0]/2), int(ellipse_model[1][1]/2)),
                                            int(ellipse_model[2]), 0, 360, 5)

            # insert IVUSProcSegmentation instead

            # get_gpu_temp()
            # get_cpu_temp()
            
            

            color_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
            
            cv2.drawContours(color_image, [ellipse_contour], -1, (0, 0, 255), thickness = 1)

            
            
       
            # For compatibility with subsequent deeplumen assumed functions
            mask_1 = np.zeros_like(grayscale_image, dtype=np.uint8)  # Create a black mask
            mask_2 = np.zeros_like(grayscale_image, dtype=np.uint8)  # Create a black mask
            cv2.fillPoly(mask_1, [ellipse_contour], 255)  # Filled white ellipse on black mask

            

            mask_1 = cv2.resize(mask_1, (224, 224))
            mask_2 = cv2.resize(mask_2, (224, 224))

            mask_1_contour,hier = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_2_contour,hier = cv2.findContours(mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

           
         

            # original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            # cv2.drawContours(original_image, [ellipse_contour], -1, (0, 0, 255), thickness = 1)
            # header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
            # rgb_image_msg = Image(
            #     header=header,
            #     height=np.shape(original_image)[0],
            #     width=np.shape(original_image)[1],
            #     encoding='rgb8',
            #     is_bigendian=False,
            #     step=np.shape(original_image)[1] * 3,
            #     data=original_image.tobytes()
            # )
            # self.rgb_image_pub.publish(rgb_image_msg)

            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(original_image, [ellipse_contour], -1, (0, 0, 255), thickness = 1)
       
            rgb_image_msg = Image(
              
                height=np.shape(original_image)[0],
                width=np.shape(original_image)[1],
                encoding='rgb8',
                is_bigendian=False,
                step=np.shape(original_image)[1] * 3,
                data=original_image.tobytes()
            )
            self.rgb_image_pub.publish(rgb_image_msg)

            


            # get 3D points from ellipse contour
            # relevant_pixels=ellipse_contour
            # centred_pixels=relevant_pixels - [centre_x,centre_y]
            # two_d_points=centred_pixels*scaling
            # three_d_points=np.hstack((np.zeros((two_d_points.shape[0], 1)),two_d_points)) 

        if((self.deeplumen_on == 1 or self.deeplumen_lstm_on == 1) and self.dest_frame=='target1'):

            if(self.deeplumen_on == 1):
            
                # FOR DISSECTION IMPLEMENTATION LATER

            

                # # note 224,224 image for compatibility with network is hardcoded
                grayscale_image = cv2.resize(grayscale_image, (224, 224))
                image = cv2.cvtColor(grayscale_image,cv2.COLOR_GRAY2RGB)


                # #---------- SEGMENTATION --------------#


                start_time = time.time()

                

                # mask_1, mask_2 = deeplumen_segmentation(image,self.model)
                pred= deeplumen_segmentation(image,self.model)
                raw_data = pred[0].numpy()
                mask_1, mask_2 = post_process_deeplumen(raw_data)

                # get_gpu_temp()
                # get_cpu_temp()

                end_time=time.time()
                diff_time=end_time-start_time
                print("segmentation time:", diff_time)

                if(self.deeplumen_valve_on == 1):
                    pred= deeplumen_segmentation(image,self.model_cusp)
                    raw_data = pred[0].numpy()
                    mask_1_opening, mask_2_cusp = post_process_deeplumen(raw_data)


                    # get mask_2 as subtraction of mask_1_opening from mask_1
                    # overwriting output from ML model!!
                    
                    # mask_2 = np.logical_and(mask_1 == 1, mask_1_opening == 0).astype(np.uint8)
                    mask_2 = np.clip(mask_1 - mask_1_opening, 0, 1)
                    mask_1 = mask_1_opening

                    # cv2.imshow("mask 2 example", mask_2)
                    # cv2.waitKey(0)

                    


            

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
            
                

         

            # ENSURES NEAR LUMEN IS ALWAYS THE ONE WITH THE PROBE IN IT
            # if(mask_2[112,112] == 255):
                
            #     mask_1, mask_2 = mask_2, mask_1

            # elif(mask_1[112,112] !=255 and mask_2[112,112] != 255):
                
            #     ret, mask = cv2.threshold(grayscale_image, threshold, 255, cv2.THRESH_BINARY)
            #     num_labels, labels = cv2.connectedComponents(mask)
            #     component_label = labels[112,112]

            #     component_mask = np.zeros_like(mask)
            #     component_mask[labels == component_label] = 255

            #     overlap_mask = cv2.bitwise_and(component_mask, mask_1)
            #     overlapping_pixels_1 = cv2.countNonZero(overlap_mask)

            #     overlap_mask = cv2.bitwise_and(component_mask, mask_2)
            #     overlapping_pixels_2 = cv2.countNonZero(overlap_mask)

            #     if(overlapping_pixels_1 > overlapping_pixels_2):
            #         mask_1 = cv2.bitwise_or(component_mask + mask_1)

            #     if(overlapping_pixels_2 > overlapping_pixels_1): 

                   
            #         mask_1, mask_2 = mask_2, mask_1

            #         mask_1 = cv2.bitwise_or(component_mask + mask_1)
            

            # self.mask_1_buffer.append(mask_1)
            # self.mask_2_buffer.append(mask_2)

            branch_pixels = np.count_nonzero(mask_2)


            if(self.previous_mask is None):
                self.previous_mask = np.zeros_like(mask_2,dtype=np.uint8)

            # every point for orifice detection
            mask_1_contour_every_point,hier = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            
        
            if(self.orifice_center_map == 1 ):


                # ------ FIND ORIFICE PIXELS ------- #
                
                # ensure that masks are well separated
                kernel = np.ones((self.minimum_thickness, self.minimum_thickness), np.uint8)  # You can adjust the kernel size for the dilation effect
                dilated_mask_1 = cv2.dilate(mask_1, kernel, iterations=1)

                # Find the overlap between the dilated mask
                touching_pixels = cv2.bitwise_and(dilated_mask_1, mask_2)

                # # for all non zero pixels in the mask "touching_pixels", find the nearest points on mask_1_contour
                non_zero_pixels = np.column_stack(np.where(touching_pixels > 0))  # Shape (M, 2)

                # Step 3: Compute nearest contour point for each non-zero pixel
                # This creates a matrix of distances between each pixel in `non_zero_pixels` and `contour_points`

                contour_points = np.vstack(mask_1_contour_every_point).squeeze()
                non_zero_pixels_xy = non_zero_pixels[:, [1, 0]]
                distances = cdist(non_zero_pixels_xy, contour_points, metric='euclidean')

                # For each non-zero pixel, find the index of the nearest contour point
                nearest_indices = distances.argmin(axis=1)  # Shape (M,)
            

                if nearest_indices.size > 0:

                    start_raycast = time.time()

                    contiguous_indices, contiguous_block_points = get_contiguous_block_from_contour(contour_points, nearest_indices)

                    orifice_mask = mask_2

                    contour_points = np.asarray(contiguous_block_points)
                    
                    normals = visualize_contour_normals(orifice_mask, contour_points)
                
                    raycast_hits, ray_lengths = compute_branch_raycast_hits(mask_2, contour_points, normals)

                    end_raycast = time.time()
                    diff_raycast = end_raycast - start_raycast

                    print("raycasting operations time:", end_raycast)

                    mid_index = len(raycast_hits) // 2

                    # check if orifice center detection is correct
                    # rgb_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
                    # cv2.circle(rgb_image, contiguous_block_points[mid_index], radius=5, color=(0, 0, 255), thickness=-1) 
                    # cv2.imshow("rgb_image", rgb_image)
                    # cv2.waitKey(0)
                    

                    orifice_center_three_d_points = get_single_point_cloud_from_pixels([contiguous_block_points[mid_index]],scaling)
                    

                
                else:
                    orifice_three_d_points = None
                    orifice_center_three_d_points = None

                # ----- DETEMINE IF OVERLAP EXISTS ------ #

                branch_visible = np.any(mask_2 > 0)

                # if a branch is visible and branch mask borders red mask
                if(branch_visible and nearest_indices.size > 0):
                    
                    # number of buffer images to compare to
                    n = 10
                    
                    if self.mask_2_buffer:
                    
                        combined_previous_mask = np.logical_or.reduce(list(self.mask_2_buffer)[-n-1:-1])
                        combined_previous_mask = (combined_previous_mask * 255).astype(np.uint8)
                        overlap = np.logical_and(combined_previous_mask == 255, mask_2 == 255)

                    else:
                        overlap = np.logical_and(self.previous_mask == 255, mask_2 == 255)
                        

                    num_overlap_pixels = np.sum(overlap)
                    threshold = 10


                    # NOT NEEDED FOR NOW?
                    # centroid_mask_1 = get_centroid_from_contour(np.asarray(mask_1_contour_every_point[0]),scaling)

                    # only calculated if blue mask is close to the border of red
                    orifice_two_d = np.asarray(contiguous_block_points[mid_index]).squeeze()

                    
                    if(np.shape(self.orifice_angles)[0] >=5):
                        average_orifice_angle = np.mean(self.orifice_angles, axis=0)
                    else:
                        average_orifice_angle = orifice_two_d 

                    angle_threshold =25


                    self.orifice_angles.append(orifice_two_d)

                    if(num_overlap_pixels > threshold and np.linalg.norm((orifice_two_d-average_orifice_angle),axis=0) < angle_threshold):
                        pass
                        
                    else:
                        self.branch_pass = self.branch_pass + 1
            
            

            # ---- PUBLISH IMAGE ----- #

            start_time = time.time()

            mask_1_contour,hier = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_2_contour,hier = cv2.findContours(mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # visualize segmentations
            grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(grayscale_image, mask_1_contour, -1, (0, 0, 255), thickness=1)
            cv2.drawContours(grayscale_image, mask_2_contour, -1, (255, 0, 0), thickness=1)

            
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

            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            mask_1_send = cv2.resize(mask_1, (np.shape(original_image)[0], np.shape(original_image)[1]))
            mask_2_send = cv2.resize(mask_2, (np.shape(original_image)[0], np.shape(original_image)[1]))
            # rather than finding contours, just colour mask to save time
            mask_1_contour_send,hier = cv2.findContours(mask_1_send, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_2_contour_send,hier = cv2.findContours(mask_2_send, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(original_image, mask_1_contour_send, -1, (0, 0, 255), thickness=2)
            cv2.drawContours(original_image, mask_2_contour_send, -1, (255, 0, 0), thickness=2)
            # header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
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

            end_time = time.time()
            diff_time = end_time - start_time
            # print("image publish time", diff_time)


        # FOT INTEGRATED IVUS AND STEERING GET RID OF THIS!
        if(self.dest_frame=='target2'):
            mask_1 = np.zeros_like(grayscale_image)
            mask_2 = np.zeros_like(grayscale_image)

            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
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

        angle=self.default_values['/angle'] 
        translation=self.default_values['/translation'] 
        radial_offset=self.default_values['/radial_offset'] 
        oclock=self.default_values['/oclock'] 

       
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


        if(self.dest_frame == 'target1'):
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
                    update_tsdf_mesh(self.vis, self.tsdf_volume_near_lumen,self.mesh_near_lumen,three_d_points_near_lumen, extrinsic_matrix,[1,0,0], keep_largest=False)

                

                    if(self.dissection_mapping!=1 and np.shape(np.array(self.mesh_near_lumen.vertices))[0]>0):
                        temp_lineset = create_wireframe_lineset_from_mesh(self.mesh_near_lumen) 
                        self.mesh_near_lumen_lineset.points = temp_lineset.points
                        self.mesh_near_lumen_lineset.lines = temp_lineset.lines
                        # if(self.refine==1):
                        #     # self.mesh_near_lumen_lineset.paint_uniform_color([0,1,0])
                        #     self.mesh_n
                        # else:

                        if(self.refine ==0):
                            self.vis.update_geometry(self.mesh_near_lumen_lineset)

                if(three_d_points_far_lumen is not None):
                    update_tsdf_mesh(self.vis,self.tsdf_volume_far_lumen,self.mesh_far_lumen,three_d_points_far_lumen, extrinsic_matrix,[0,0,1], keep_largest =False)

            
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

                # IVUS BASED DEFORMATION
                if(self.live_deformation == 1 and self.registered_ct == 1 and self.dest_frame == 'target1'):
                     
                    #  coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])
                    #  coordinate_frame.transform(TW_EM @ TEM_C)
                    #  o3d.visualization.draw_geometries([self.registered_ct_mesh, self.centerline_pc, coordinate_frame, near_pC_points])

                    #  try:
                        
                    deformed_mesh = live_deform(self.registered_ct_mesh, self.constraint_indices, self.centerline_pc, TW_EM @ TEM_C, near_vpC_points )
                    temp_lineset = create_wireframe_lineset_from_mesh(deformed_mesh)
                    self.registered_ct_lineset.points = temp_lineset.points
                    self.registered_ct_lineset.lines = temp_lineset.lines
                    self.vis.update_geometry(self.registered_ct_lineset)

                    # mapping from coarse deformed mesh to fine deformed mesh nodes for endoscopic view
                    fine_deformed_vertices = deform_fine_mesh_using_knn(self.registered_ct_mesh, deformed_mesh, self.registered_ct_mesh_2, self.knn_idxs, self.knn_weights, self.coarse_template_vertices, self.fine_template_vertices, self.adjacency_matrix)

                    # self.registered_ct_mesh_2.vertices = deformed_mesh.vertices
                    self.registered_ct_mesh_2.vertices = o3d.utility.Vector3dVector(fine_deformed_vertices)
                    self.registered_ct_mesh_2.compute_vertex_normals()
                    self.vis2.update_geometry(self.registered_ct_mesh_2)


                    # deform the green spheres
                    ct_spheres_deformed_points = deform_points_using_knn(self.registered_ct_mesh, deformed_mesh,  self.knn_idxs_spheres, self.knn_weights_spheres, self.coarse_template_vertices, self.ct_centroids, self.adjacency_matrix)
                    ct_spheres_temp = get_sphere_cloud(ct_spheres_deformed_points , 0.00225, 20, [0,1,0])
                    self.ct_spheres.vertices =  ct_spheres_temp.vertices
                    self.vis.update_geometry(self.ct_spheres)
                    self.vis2.update_geometry(self.ct_spheres)
                       
                    #  except:
                    #     print("IVUS live mesh deformation not working")
                    #     pass

                if(self.extend == 1 and (self.dissection_mapping == 1 or self.refine==1)):
                    # prevent memory issues by commenting this out
                    self.volumetric_near_point_cloud.points.extend(near_vpC_points.points)
                else:
                    self.volumetric_near_point_cloud.points = near_vpC_points.points


                self.volumetric_near_point_cloud.paint_uniform_color([1,0,0])

            # EM BASED DEFORMATION


            if(self.live_deformation == 1 and self.registered_ct == 1 and self.dest_frame == 'target2' and volumetric_three_d_points_near_lumen is None):

                
                    
                
                # check if EM transform is outside the mesh before deformation calculations
                query_points = np.asarray([TW_EM[:3,3]])
                query_points_tensor = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
                signed_distance = self.scene.compute_signed_distance(query_points_tensor)
                signed_distance_np = signed_distance.numpy()  # Convert to NumPy array

               
            
                near_vpC_points = None #this should still work with live deform there's a condition inside its

                if(signed_distance_np[0] > 0):
                    # try:  
                    deformed_mesh = live_deform(self.registered_ct_mesh, self.constraint_indices, self.centerline_pc, TW_EM, near_vpC_points, self.dest_frame, self.scene, self.free_branch_indices )
                    
                    temp_lineset = create_wireframe_lineset_from_mesh(deformed_mesh)
                    self.registered_ct_lineset.points = temp_lineset.points
                    self.registered_ct_lineset.lines = temp_lineset.lines
                    self.vis.update_geometry(self.registered_ct_lineset)

                    # mapping from coarse deformed mesh to fine deformed mesh nodes for endoscopic view
                    fine_deformed_vertices = deform_fine_mesh_using_knn(self.registered_ct_mesh, deformed_mesh, self.registered_ct_mesh_2, self.knn_idxs, self.knn_weights, self.coarse_template_vertices, self.fine_template_vertices, self.adjacency_matrix)

                    # self.registered_ct_mesh_2.vertices = deformed_mesh.vertices
                    self.registered_ct_mesh_2.vertices = o3d.utility.Vector3dVector(fine_deformed_vertices)
                    self.registered_ct_mesh_2.compute_vertex_normals()
                    self.vis2.update_geometry(self.registered_ct_mesh_2)

                    # deform the green spheres
                    ct_spheres_deformed_points = deform_points_using_knn(self.registered_ct_mesh, deformed_mesh,  self.knn_idxs_spheres, self.knn_weights_spheres, self.coarse_template_vertices, self.ct_centroids, self.adjacency_matrix)
                    ct_spheres_temp = get_sphere_cloud(ct_spheres_deformed_points , 0.00225, 20, [0,1,0])
                    self.ct_spheres.vertices =  ct_spheres_temp.vertices
                    self.vis.update_geometry(self.ct_spheres)
                    self.vis2.update_geometry(self.ct_spheres)
                    # except:
                    #     print("EM live mesh deformation not working")
                    #     pass


                

                

            if(volumetric_three_d_points_far_lumen is not None):
                far_vpC_points=o3d.geometry.PointCloud()
                far_vpC_points.points=o3d.utility.Vector3dVector(volumetric_three_d_points_far_lumen)

                #downsample volumetric point cloud
                far_vpC_points = far_vpC_points.voxel_down_sample(voxel_size=0.0005)
                far_vpC_points.transform(TW_EM @ TEM_C)


                # for results evaluation
                max_branch_pass = 255  # Set based on your application needs
                normalized_pass = self.branch_pass / max_branch_pass  # Scale to [0, 1]
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

                    


                # OVERRIDE BRANCH PASS COLOURING
                self.volumetric_far_point_cloud.paint_uniform_color([0,0,1])


            # self.vis.update_geometry(self.point_cloud)
            #JUST DONT VISUALIZE IT
   
            self.vis.update_geometry(self.volumetric_near_point_cloud)
            self.vis.update_geometry(self.volumetric_far_point_cloud)

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

 
        
        
                

        # ----- SIMULATE PROBE VIEW ------- #
        # if(self.registered_ct ==  1 or self.dissection_track == 1):
        if(self.registered_ct ==  1 ):
            view_control = self.vis2.get_view_control()

            # Set the camera view aligned with the x-axis
            camera_parameters = view_control.convert_to_pinhole_camera_parameters()

    

            T = TW_EM 

            up = T[:3, 2]          # Z-axis (3rd column)
            # front = -T[:3, 0]      # Negative X-axis (1st column) - look forward or back
            front = -T[:3, 0]      # Negative X-axis (1st column) - look forward or back

            lookat = T[:3, 3]      # Translation vector (camera position)
            translation = np.asarray([-0.002,0,0.004])
            lookat = (TW_EM[:3,:3] @ translation) + lookat
            

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
                view_control_1.set_lookat(lookat)

            
        if(self.registered_ct ==1):


                view_control_1 = self.vis.get_view_control()
                camera_parameters_1 = view_control_1.convert_to_pinhole_camera_parameters()
                
                # points = np.asarray(self.registered_ct_lineset.points)  # Convert point cloud to numpy array
                # centroid = np.mean(points, axis=0) 

                centroid = self.registered_centroid

                position_tracker = TW_EM[:3,3]
                average_point = ((centroid/4)+(3*position_tracker/4))
                lookat = average_point


                view_control_1.set_lookat(lookat)

                
                # Define the up direction explicitly (same as before)
                up = np.array([0, -1, 0])

                # what is the up axis of the vessel - flip this depending on side of table
                
                

        # ---- APPEND TO ORIFICE CENTER PC ------ #
        if(self.orifice_center_map == 1 and (self.deeplumen_on ==1 or self.deeplumen_lstm_on==1)):

            if(orifice_center_three_d_points is not None):           
                
                print("current branch pass", self.branch_pass)
                

                max_branch_pass = 255  # Set based on your application needs
                normalized_pass = self.branch_pass / max_branch_pass  # Scale to [0, 1]

        
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

                    
                    orifice_center_spheres_temp = get_sphere_cloud(np.asarray(self.orifice_center_point_cloud.points), 0.0025, 12, [0,0,1])
                    self.orifice_center_spheres.vertices = orifice_center_spheres_temp.vertices
                    self.orifice_center_spheres.triangles = orifice_center_spheres_temp.triangles
                    self.orifice_center_spheres.compute_vertex_normals()
                    self.orifice_center_spheres.paint_uniform_color([0,0,1])
                
                   
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


        if(self.registered_ct ==1):
            self.tracker_frame.transform(get_transform_inverse(self.previous_transform))
            self.tracker_frame.transform(TW_EM)
            self.previous_transform=TW_EM
            self.vis.update_geometry(self.tracker_frame)
        
        

        # if(self.registered_ct!=1):
            
            
        self.previous_transform_us=TW_EM @ TEM_C
        self.us_frame.transform(get_transform_inverse(self.previous_transform_us))
        self.us_frame.transform(TEM_C)
        self.us_frame.transform(TW_EM)
        self.vis.update_geometry(self.us_frame)
        self.vis.update_geometry(self.catheter)

    

        if(self.registered_ct ==1):
            self.vis2.update_geometry(self.tracker)
            self.vis2.poll_events()
            self.vis2.update_renderer()
        
            if(self.guidewire==1):

                

                T_guidewire = self.get_catheter_transform(TW_EM @ self.TEM_GW)
                self.guidewire_cylinder.transform(get_transform_inverse(self.previous_guidewire_transform))
                self.guidewire_cylinder.transform(T_guidewire)
                self.previous_guidewire_transform = T_guidewire
                self.vis.update_geometry(self.guidewire_cylinder)
                guidewire_pointcloud_temp = copy.deepcopy(self.guidewire_pointcloud_base)
                guidewire_pointcloud_temp.transform(T_guidewire)
                query_points = np.asarray(guidewire_pointcloud_temp.points)

                query_points_tensor = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
                signed_distance = self.scene.compute_signed_distance(query_points_tensor)
                signed_distance_np = signed_distance.numpy()  # Convert to NumPy array
                arg_points_inside = np.argwhere(signed_distance_np < 0).flatten()
                self.guidewire_pointcloud.points = o3d.utility.Vector3dVector(query_points[arg_points_inside, :])

                # self.guidewire_pointcloud.points = o3d.utility.Vector3dVector(query_points)
                self.guidewire_pointcloud.paint_uniform_color([0,1,0])
                self.vis.update_geometry(self.guidewire_pointcloud)


        

        self.vis.poll_events()
        self.vis.update_renderer()

        # stop_entire = time.time()
        # diff_entire = stop_entire - self.start_entire
        # print("entire loop time", diff_entire)

    
            

  




if __name__ == '__main__':
    try:
        rospy.init_node('open3d_visualizer')
        pc_updater = PointCloudUpdater()        
        # rospy.on_shutdown(pc_updater.save_image_and_transform_data)
        rospy.spin()
        
    except rospy.ROSInterruptException:
        self.vis.destroy_window()
        pass
