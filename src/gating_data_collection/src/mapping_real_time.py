#!/usr/bin/env python3.9

import rospy
import sys
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
import math
import yaml
import os
import time
import random
import threading
from collections import deque
# import snake as sn

from voxblox import (
    BaseTsdfIntegrator,
    FastTsdfIntegrator,
    MergedTsdfIntegrator,
    SimpleTsdfIntegrator,
)

from segmentation_helpers import *
from reconstruction_helpers import *

class PointCloudUpdater:



    def __init__(self):
        
        # load the yaml file - specific to post processing
        # with open('/home/tdillon/mapping/src/mapping_parameters_real_time.yaml', 'r') as file:
        with open('/home/tdillon/mapping/src/mapping_parameters_real_time.yaml', 'r') as file:
            config_yaml = yaml.safe_load(file)

        self.savepath = config_yaml['savepath']
        # is there an ECG signal?
        self.gating = config_yaml['gating']

        if(self.gating == 1):
            self.savepath = self.savepath + '/ungated'

        create_folder(self.savepath)

        # save data?
        self.record = config_yaml['record']

        # healthy first return mapping for reference
        self.tsdf_map = config_yaml['tsdf_map']

        self.deeplumen_on = config_yaml['deeplumen_on']

        self.model_path  = config_yaml['model_path']

        self.vpC_map = config_yaml['vpC_map']

        self.extend = config_yaml['extend']

        self.tsdf_map = config_yaml['tsdf_map']

        self.orifice_center_map = config_yaml['orifice_center_map']

        self.registered_ct = config_yaml['registered_ct']

        self.registered_ct_dataset = config_yaml['registered_ct_dataset']

        self.live_deformation = config_yaml['live_deformation']

        

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


        # Set default parameter values if they don't exist
        for param_name in param_names:
            if not rospy.has_param(param_name):
                rospy.set_param(param_name, self.default_values.get(param_name, None))
                rospy.loginfo(f"Initialized {param_name} with default value: {self.default_values.get(param_name, None)}")


        #setup subscribers
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        print("frame grabber subscriber:", self.image_sub)
        print("tf listener:", self.tf_listener)

        if(self.gating ==1):
            self.lock = threading.Lock()  # A lock to ensure thread safety
            self.ecg_sub = rospy.Subscriber('ecg', Int32, self.ecg_callback)
            print("ecg signal subscriber:", self.ecg_sub)



        #setup publishers
        self.binary_image_pub = rospy.Publisher('/binary_image', Image, queue_size=1)
        self.rgb_image_pub = rospy.Publisher('/rgb_image', Image, queue_size=1)

        self.pullback_pub = rospy.Publisher('/pullback', Int32, queue_size=1)

        self.image_width = rospy.get_param('/usb_cam/image_width', default=1280)
        self.image_height = rospy.get_param('/usb_cam/image_height', default=1024)

        self.crop_radius=10
        no_points=rospy.get_param('no_points')
        self.previous_no_points=no_points
        

        # what number image callback are we on?
        self.image_call=1
        
        if(self.record==1):
            self.write_folder = self.savepath
            self.image_batch=[]
            self.tw_em_batch=[]
            
            self.ecg_times=[]
            self.ecg_signal=[]
            self.image_times=[]



        # ---- INITIALIZE VISUALIZER ----- #
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.get_render_option().mesh_show_back_face = True
        
        # ----- INITIALIZE RECONSTRUCTIONS ------ #
        self.near_point_cloud = o3d.geometry.PointCloud()
        self.far_point_cloud = o3d.geometry.PointCloud()
        self.dissection_flap_point_cloud = o3d.geometry.PointCloud()
        self.volumetric_far_point_cloud = o3d.geometry.PointCloud() 
        self.volumetric_near_point_cloud = o3d.geometry.PointCloud() 
        self.point_cloud = o3d.geometry.PointCloud()
        self.orifice_center_point_cloud = o3d.geometry.PointCloud()


        self.vis.add_geometry(self.near_point_cloud)
        self.vis.add_geometry(self.far_point_cloud)
        self.vis.add_geometry(self.dissection_flap_point_cloud)
        self.vis.add_geometry(self.volumetric_near_point_cloud)
        self.vis.add_geometry(self.volumetric_far_point_cloud)
        self.vis.add_geometry(self.point_cloud)
        self.vis.add_geometry(self.orifice_center_point_cloud)



        # ----- INITIALIZE TRACKER FRAMES ------ #
        self.frame_scaling=0.025
        self.tracker_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.tracker_frame.scale(self.frame_scaling,center=[0,0,0])

        self.baseframe=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.baseframe.scale(self.frame_scaling,center=[0,0,0])

        self.us_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.us_frame.scale(self.frame_scaling,center=[0,0,0])
        
        self.previous_transform=np.eye(4)
        self.previous_transform_1=np.eye(4)
        self.previous_transform_us=np.eye(4)

        self.vis.add_geometry(self.us_frame)
        self.vis.add_geometry(self.tracker_frame)
        self.vis.add_geometry(self.baseframe)

    
        # ----- INITIALIZE BOUNDING BOX ----- #
        min_bounds=np.array([0.05,-0.1,-0.1]) 
        max_bounds=np.array([0.3,0.1,0.1]) 
        self.box=get_box(min_bounds,max_bounds)
        self.vis.add_geometry(self.box)

  
        # ------- INITIALIZE IMAGING PARAMETERS ------- #

        self.minimum_thickness = 15


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

        
        # BUFFER INITIALIZATION
        self.buffer_size = 30
        self.centroid_buffer = deque(maxlen=self.buffer_size)
        self.delta_buffer_x = deque(maxlen=self.buffer_size)
        self.delta_buffer_y = deque(maxlen=self.buffer_size)
        self.delta_buffer_z = deque(maxlen=self.buffer_size)
        self.position_buffer = deque(maxlen=self.buffer_size)
        self.mask_1_buffer = deque(maxlen=self.buffer_size)
        self.mask_2_buffer = deque(maxlen=self.buffer_size)
        self.orifice_angles = deque(maxlen=5)

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
        sdf_trunc = 2 * self.voxel_size
        self.tsdf_volume = SimpleTsdfIntegrator(self.voxel_size, sdf_trunc)
        self.mesh=o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.mesh)

        # ------- INITIALIZE DEEPLUMEN ML MODEL ------- #

        if(self.deeplumen_on == 1):
            # for dissection mapping, load the GPU
            print(f"Python version: {sys.version}")
            print(f"Version info: {sys.version_info}")
            print(f"TensorFlow version: {tf.__version__}")
            

            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            DRN_inputs_3,DRN_outputs_3 = get_DRN_network()

            
            model = tf.keras.Model(inputs=DRN_inputs_3, outputs=DRN_outputs_3)


            # sub branch segmentation
            model.load_weights( self.model_path)  


            model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

            model.summary()

            self.model = model

        if(self.registered_ct ==1):
            # needs to be scaled down?
            # registered_ct_lineset = o3d.io.read_line_set('/home/tdillon/datasets/' + dataset_name + '' + '/registered_ct.ply')
            self.registered_ct_lineset = o3d.io.read_line_set('/home/tdillon/datasets/' + self.registered_ct_dataset + '' + '/final_registration.ply')
            self.registered_ct_mesh = o3d.io.read_triangle_mesh('/home/tdillon/datasets/' + self.registered_ct_dataset + '' + '/final_registration_mesh.ply')
            self.registered_ct_mesh.remove_unreferenced_vertices()


            ct_centroid_pc = o3d.io.read_point_cloud('/home/tdillon/datasets/' + self.registered_ct_dataset + '' + '/side_branch_centrelines.ply')
     
            self.constraint_locations = np.asarray(ct_centroid_pc.points)
           

            self.centerline_pc = o3d.io.read_point_cloud('/home/tdillon/datasets/' + self.registered_ct_dataset + '' + '/centerline_pc.ply')

            
            volumetric_full_lumen_point_cloud = o3d.io.read_point_cloud('/home/tdillon/datasets/' + self.registered_ct_dataset + '' + '/volumetric_full_lumen_point_cloud.ply')
            # volumetric_full_lumen_point_cloud_check = o3d.io.read_point_cloud('/home/tdillon/datasets/' + dataset_name + '' + '/downsampled_pC.ply')

            ivus_funsr_mesh = o3d.io.read_triangle_mesh('/home/tdillon/datasets/' + self.registered_ct_dataset + '' + '/full_lumen_mesh.ply')

   

            self.registered_ct_lineset.paint_uniform_color([0,0,0])



            self.registered_ct_mesh_2 = copy.deepcopy(self.registered_ct_mesh)
            self.registered_ct_mesh_2.compute_vertex_normals()


      
      
            som_centreline = o3d.io.read_point_cloud('/home/tdillon/datasets/' + self.registered_ct_dataset + '' + '/som_centreline.ply')
            som_centreline_points = np.asarray(som_centreline.points) 
   

            


            ivus_funsr_lineset = create_wireframe_lineset_from_mesh(ivus_funsr_mesh)
            ivus_funsr_lineset.paint_uniform_color([0,0,0])
            self.vis.add_geometry(self.registered_ct_lineset)
            # self.vis.add_geometry(ivus_funsr_lineset)
            # self.vis.add_geometry(som_centreline)
            # self.vis.add_geometry(volumetric_full_lumen_point_cloud)
            print("registered the ct from non rigid icp!!")
            
            self.vis2 = o3d.visualization.Visualizer()
            self.vis2.create_window()
            self.vis2.get_render_option().mesh_show_back_face = True
            self.vis2.add_geometry(self.registered_ct_mesh_2)
            

            
            self.z_motion = 0
            self.z_position_previous = 0
            self.branch_pixel_array = []
            self.z_positions = []
            self.z_motions = []

            self.geodesic_distances = compute_geodesic_distances_simple(som_centreline_points)
            self.som_centreline = som_centreline_points
            self.som_centreline = np.asarray(som_centreline.points)

        if(self.live_deformation == 1):
            self.constraint_radius=self.default_values['/constraint_radius'] 
                       
            self.constraint_locations = np.vstack((self.constraint_locations,np.asarray(self.centerline_pc.points)[0,:],np.asarray(self.centerline_pc.points)[-1,:]))
            # o3d.visualization.draw_geometries([self.registered_ct_mesh, self.baseframe])
            self.constraint_indices = get_all_nodes_inside_radius(self.constraint_locations, self.constraint_radius, self.registered_ct_mesh)
            test_pc= o3d.geometry.PointCloud() 
            test_pc.points = o3d.utility.Vector3dVector(np.asarray(self.registered_ct_mesh.vertices)[self.constraint_indices,:])
            # o3d.visualization.draw_geometries([self.registered_ct_mesh, self.baseframe, test_pc])
            
         

        pullback = rospy.get_param('pullback', 0)
        self.pullback_pub.publish(pullback)
        print("pullback check", pullback)

    
        self.ray_lengths_global = []
        self.branch_pass = 0
        self.branch_visible_previous = 0
        self.previous_mask = None



    def ecg_callback(self,ecg_msg):

        current_time = rospy.get_rostime()
        ecg_timestamp_secs = current_time.secs
        ecg_timestamp_nsecs = current_time.nsecs
        
        ecg_timestamp_in_seconds = ecg_timestamp_secs + (ecg_timestamp_nsecs * 1e-9)

        ecg_value = ecg_msg.data

        if(self.record==1):
            self.ecg_times.append( ecg_timestamp_in_seconds)
            self.ecg_signal.append(ecg_value)

        
    def image_callback(self, msg):
        

        start_total_time = time.time()
        # this is the shortest number of lines it takes to grab the transform once image arrives
        # transform_time = rospy.Time(0) #get the most recent transform
        transform_time = msg.header.stamp #get the most recent transform
        
        # Assuming you have the frame IDs for your transform
        ref_frame = 'ascension_origin'
        dest_frame = 'target1'

        try:
            # Lookup transform
            TW_EM = self.tf_buffer.lookup_transform(ref_frame, dest_frame, transform_time)
        except (rospy.ROSException, tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Failed to lookup transform")
            TW_EM = None

        TW_EM=transform_stamped_to_matrix(TW_EM)

        # now get the original image's timestamp
        timestamp_secs = msg.header.stamp.secs
        timestamp_nsecs = msg.header.stamp.nsecs
        image_timestamp_in_seconds = timestamp_secs + (timestamp_nsecs * 1e-9)

        # Create a rospy.Time object using the truncated timestamp information
        timestamp = rospy.Time(secs=timestamp_secs, nsecs=timestamp_nsecs)

        
        # Assuming RGB format
        rgb_image_data = np.frombuffer(msg.data, dtype=np.uint8)
        rgb_image = rgb_image_data.reshape((self.image_height, self.image_width, 3))

        
        start_time = time.time()
        grayscale_image=preprocess_ivus_image(rgb_image,self.box_crop,self.circle_crop,self.text_crop,self.crosshairs_crop)
        end_time = time.time()
        diff_time = end_time - start_time 
        print("preprocessing time", diff_time)
        
        # how to make this faster
        if(self.record==1):
            # code for saving images

            self.image_batch.append(grayscale_image)
            self.tw_em_batch.append(TW_EM)
            self.image_times.append(image_timestamp_in_seconds)

        # for computational efficiency, stop here and do all processing / reconstruction later
        if(self.tsdf_map != 1 and self.vpC_map != 1):
            pass

        # fetch rospy parameters for real time mapping (could be placed in initialization)
        threshold = rospy.get_param('threshold')
        no_points = rospy.get_param('no_points')
        crop_index = rospy.get_param('crop_index')
        scaling = rospy.get_param('scaling')
        angle = rospy.get_param('angle')
        translation = rospy.get_param('translation')
        radial_offset = rospy.get_param('radial_offset')
        oclock = rospy.get_param('oclock')

        pullback = rospy.get_param('pullback', 0)
        
        self.pullback_pub.publish(pullback)

        centre_x=self.centre_x
        centre_y=self.centre_y

        

        # ------ FIRST RETURN SEGMENTATION -------- #

        # first return segmentation
        if(self.deeplumen_on == 0 ):

            # check if rospy parameter has changed:
            if no_points != self.previous_no_points:
                self.gridlines=get_gridlines(centre_x,centre_y,no_points, self.crop_radius)
                self.previous_no_points=no_points

            relevant_pixels=first_return_segmentation(grayscale_image,threshold, crop_index,self.gridlines)
            relevant_pixels=np.asarray(relevant_pixels).squeeze()
            binary_image=np.zeros_like(grayscale_image)
            binary_image[relevant_pixels[:, 0], relevant_pixels[:, 1]] = 255

            # first return image publishing
            header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
            binary_image_msg = Image(
                header=header,
                height=self.new_height,
                width=self.new_width,
                encoding='mono8',
                is_bigendian=False,
                step=self.new_width,
                data=binary_image.tobytes()
            )
            self.binary_image_pub.publish(binary_image_msg)


            # ellipse fitting to first return
            ellipse_model = cv2.fitEllipse(relevant_pixels[:,[1,0]].astype(np.float32))  
            ellipse_contour= cv2.ellipse2Poly((int(ellipse_model[0][0]), int(ellipse_model[0][1])),
                                            (int(ellipse_model[1][0]/2), int(ellipse_model[1][1]/2)),
                                            int(ellipse_model[2]), 0, 360, 5)
            color_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(color_image, [ellipse_contour], -1, (0, 0, 255), 1)

            # ellipse image publishing
            header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
            rgb_image_msg = Image(
                header=header,
                height=self.new_height,
                width=self.new_width,
                encoding='rgb8',
                is_bigendian=False,
                step=self.new_width * 3,
                data=color_image.tobytes()
            )
            self.rgb_image_pub.publish(rgb_image_msg)

            # get 3D points from ellipse contour
            relevant_pixels=ellipse_contour
            centred_pixels=relevant_pixels - [centre_x,centre_y]
            two_d_points=centred_pixels*scaling
            three_d_points=np.hstack((np.zeros((two_d_points.shape[0], 1)),two_d_points)) 


        if(self.deeplumen_on == 1):
        
            # FOR DISSECTION IMPLEMENTATION LATER

            # # note 224,224 image for compatibility with network is hardcoded
            grayscale_image = cv2.resize(grayscale_image, (224, 224))
            image = cv2.cvtColor(grayscale_image,cv2.COLOR_GRAY2RGB)


            # #---------- SEGMENTATION --------------#

            mask_1, mask_2 = deeplumen_segmentation(image,self.model)

            cv2.imshow("mask image", mask_1)
            cv2.waitKey(1)

            

            self.mask_1_buffer.append(mask_1)
            self.mask_2_buffer.append(mask_2)

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

                    contiguous_indices, contiguous_block_points = get_contiguous_block_from_contour(contour_points, nearest_indices)

                    orifice_mask = mask_2

                    contour_points = np.asarray(contiguous_block_points)
                    
                    normals = visualize_contour_normals(orifice_mask, contour_points)
                
                    raycast_hits, ray_lengths = compute_branch_raycast_hits(mask_2, contour_points, normals)

                    mid_index = len(raycast_hits) // 2

                    orifice_center_three_d_points = get_single_point_cloud_from_pixels([contiguous_block_points[mid_index]],scaling)
                    print("got non zero points")

                
                else:
                    orifice_three_d_points = None
                    orifice_center_three_d_points = None

                # ----- DETEMINE IF OVERLAP EXISTS ------ #

                branch_visible = np.any(mask_2 > 0)

                # if a branch is visible
                if(branch_visible):
                    
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



                    centroid_mask_1 = get_centroid_from_contour(np.asarray(mask_1_contour_every_point[0]),scaling)
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
            cv2.drawContours(grayscale_image, mask_1_contour, -1, (0, 0, 255), thickness=2)
            cv2.drawContours(grayscale_image, mask_2_contour, -1, (255, 0, 0), thickness=2)
           

            # ellipse image publishing
            header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
            rgb_image_msg = Image(
                header=header,
                height=224,
                width=224,
                encoding='rgb8',
                is_bigendian=False,
                step=self.new_width * 3,
                data=grayscale_image.tobytes()
            )
            self.rgb_image_pub.publish(rgb_image_msg)

            end_time = time.time()
            diff_time = end_time - start_time
            print("diff time", diff_time)


        scaling=self.default_values['/scaling'] 
        # if dissection_parameterize == 0:
        #     three_d_points, three_d_points_near_lumen,three_d_points_far_lumen,three_d_points_dissection_flap = get_point_cloud_from_masks(combined_mask, scaling, mask_1_contour,mask_2_contour)

        # if dissection_parameterize == 1:
        #     three_d_points, three_d_points_near_lumen,three_d_points_far_lumen,three_d_points_dissection_flap = get_point_cloud_from_masks(combined_mask, scaling, mask_1_contour,mask_2_contour, dissection_flap_skeleton)

        if self.vpC_map == 1:
            volumetric_three_d_points_near_lumen = get_single_point_cloud_from_mask(mask_1, scaling)
            volumetric_three_d_points_far_lumen = get_single_point_cloud_from_mask(mask_2, scaling) 

        # ---- KINEMATICS ---- #

        angle=self.default_values['/angle'] 
        translation=self.default_values['/translation'] 
        radial_offset=self.default_values['/radial_offset'] 
        oclock=self.default_values['/oclock'] 


        TEM_C = [[1,0,0,translation],[0,np.cos(angle),-np.sin(angle),radial_offset*np.cos(oclock)],[0,np.sin(angle),np.cos(angle),radial_offset*np.sin(oclock)],[0, 0, 0, 1]]




        # ------ NON SMOOTHED ESDF MESHING ------ #
        # colors_certainty_TL = update_esdf_mesh(self.voxelCarver, self.esdf_mesh,TW_EM @ TEM_C, mask_1, [1,0,0])
        # colors_certainty_FL = update_esdf_mesh(self.voxelCarver_2, self.esdf_mesh_2,TW_EM @ TEM_C, mask_2, [0,0,1])
        # if(certainty_coloring  == 1):
        #     self.esdf_mesh.color = colors_certainty_TL
        #     self.esdf_mesh_2.color = colors_certainty_FL
        # self.vis.update_geometry(self.esdf_mesh)
        # self.vis.update_geometry(self.esdf_mesh_2)


        extrinsic_matrix=TW_EM @ TEM_C

        # ------- VOXBLOX TSDF MESHING -------- #

        # disssection
        if(self.tsdf_map ==1 and self.deeplumen_on == 1):

            if(three_d_points_near_lumen is not None):
                self.update_tsdf_mesh(self.vis, self.tsdf_volume_near_lumen,self.mesh_near_lumen,three_d_points_near_lumen, extrinsic_matrix,[1,0,0])
            if(three_d_points_far_lumen is not None):
                self.update_tsdf_mesh(self.vis,self.tsdf_volume_far_lumen,self.mesh_far_lumen,three_d_points_far_lumen, extrinsic_matrix,[0,0,1])
            if(tsdf_map == 1):
                if(three_d_points_dissection_flap is not None):
                    self.update_tsdf_mesh(self.vis,self.tsdf_volume_dissection_flap,self.mesh_dissection_flap,three_d_points_dissection_flap, extrinsic_matrix,[0,1,0])

        # healthy
        if(self.tsdf_map ==1 and self.deeplumen_on == 1):
            self.tsdf_volume.integrate(points=three_d_points, extrinsic=extrinsic_matrix)
            vertices, triangles = self.tsdf_volume.extract_triangle_mesh()
            self.mesh.triangles= o3d.utility.Vector3iVector(triangles)
            self.mesh.vertices= o3d.utility.Vector3dVector(vertices)
            self.mesh.merge_close_vertices(0.00001)

            # keep the largest cluster only
            if(np.shape(triangles)[0]>0):
                triangle_clusters, cluster_n_triangles, _ = (self.mesh.cluster_connected_triangles())
                triangle_clusters = np.asarray(triangle_clusters)
                cluster_n_triangles = np.asarray(cluster_n_triangles)
                largest_cluster_idx = cluster_n_triangles.argmax()
                triangles_to_remove = triangle_clusters != largest_cluster_idx
                self.mesh.remove_triangles_by_mask(triangles_to_remove)

            self.mesh.compute_vertex_normals()
            self.vis.update_geometry(self.mesh)

        # volumetric point cloud
        if(self.vpC_map == 1):
            if(volumetric_three_d_points_near_lumen is not None):
                near_vpC_points=o3d.geometry.PointCloud()
                near_vpC_points.points=o3d.utility.Vector3dVector(volumetric_three_d_points_near_lumen)

                # downsample volumetric point cloud
                near_vpC_points = near_vpC_points.voxel_down_sample(voxel_size=0.0005)

                near_vpC_points.transform(TW_EM @ TEM_C)

                if(self.live_deformation == 1 and self.registered_ct == 1):
                     
                    #  coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])
                    #  coordinate_frame.transform(TW_EM @ TEM_C)
                    #  o3d.visualization.draw_geometries([self.registered_ct_mesh, self.centerline_pc, coordinate_frame, near_pC_points])

                     try:
                        deformed_mesh = live_deform(self.registered_ct_mesh, self.constraint_indices, self.centerline_pc, TW_EM @ TEM_C, near_vpC_points )
                        temp_lineset = create_wireframe_lineset_from_mesh(deformed_mesh)
                        self.registered_ct_lineset.points = temp_lineset.points
                        self.registered_ct_lineset.lines = temp_lineset.lines
                        self.vis.update_geometry(self.registered_ct_lineset)

                        self.registered_ct_mesh_2.vertices = deformed_mesh.vertices
                        self.registered_ct_mesh_2.compute_vertex_normals()
                        self.vis2.update_geometry(self.registered_ct_mesh_2)
                     except:
                        pass

                if(self.extend == 1 ):
                    self.volumetric_near_point_cloud.points.extend(near_vpC_points.points)
                else:
                    self.volumetric_near_point_cloud.points = near_vpC_points.points
                self.volumetric_near_point_cloud.paint_uniform_color([1,0,0])

            if(volumetric_three_d_points_far_lumen is not None):
                far_vpC_points=o3d.geometry.PointCloud()
                far_vpC_points.points=o3d.utility.Vector3dVector(volumetric_three_d_points_far_lumen)

                #downsample volumetric point cloud
                far_vpC_points = far_vpC_points.voxel_down_sample(voxel_size=0.0005)
      

                far_vpC_points.transform(TW_EM @ TEM_C)
                if(self.extend == 1):
                    self.volumetric_far_point_cloud.points.extend(far_vpC_points.points)
                else:
                    self.volumetric_far_point_cloud.points = far_vpC_points.points
                self.volumetric_far_point_cloud.paint_uniform_color([0,0,1])

            # self.vis.update_geometry(self.point_cloud)
            self.vis.update_geometry(self.volumetric_near_point_cloud)
            self.vis.update_geometry(self.volumetric_far_point_cloud)
                

        # ----- SIMULATE PROBE VIEW ------- #
        if(self.registered_ct ==  1):
            view_control = self.vis2.get_view_control()

            # Set the camera view aligned with the x-axis
            camera_parameters = view_control.convert_to_pinhole_camera_parameters()

            # intrinsic = camera_parameters.intrinsic

            T = TW_EM @ TEM_C

            up = T[:3, 2]          # Z-axis (3rd column)
            front = -T[:3, 0]      # Negative X-axis (1st column) - look forward or back
            lookat = T[:3, 3]      # Translation vector (camera position)

            print("is this changing??", T)

            # view_control.set_up(up) # lock rotation
            view_control.set_front(front)
            view_control.set_lookat(lookat)

            view_control.set_zoom(0.01)
            # view_control.set_zoom(0.05)

            # Define a transformation to align the camera along the x-axis
            # camera_parameters.extrinsic = TW_EM @ TEM_C

            # Synchronize the pinhole parameters with the current window size
            # view_control.convert_from_pinhole_camera_parameters(camera_parameters)

            self.vis2.poll_events()
            self.vis2.update_renderer()

        if(self.orifice_center_map == 1):

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
                self.orifice_center_point_cloud.points.extend(orifice_center_points.points)
                self.orifice_center_point_cloud.colors.extend(orifice_center_points.colors)
                
            
            self.vis.update_geometry(self.orifice_center_point_cloud)  

        
        # ------ TRACKER FRAMES ------ #
        self.tracker_frame.transform(get_transform_inverse(self.previous_transform))
        self.tracker_frame.transform(TW_EM)
        self.previous_transform=TW_EM
        self.us_frame.transform(get_transform_inverse(self.previous_transform_us))
        self.us_frame.transform(TEM_C)
        self.us_frame.transform(TW_EM)
        self.previous_transform_us=TW_EM @ TEM_C
        self.vis.update_geometry(self.us_frame)
        self.vis.update_geometry(self.tracker_frame)

        self.vis.poll_events()
        self.vis.update_renderer()

        # time.sleep(0.1)

        total_end_time = time.time()
        diff_total_time = total_end_time - start_total_time
        print("callback total time",diff_total_time)








    def save_image_and_transform_data(self):

        save_frequency = 1

        time.sleep(1)

        # o3d.io.write_point_cloud(dataset_directory + pc_updater.dataset_name + bin_name +  "/volumetric_near_lumen_point_cloud.ply", pc_updater.volumetric_near_point_cloud)

        # o3d.io.write_point_cloud(dataset_directory + pc_updater.dataset_name + bin_name +  "/volumetric_far_lumen_point_cloud.ply", pc_updater.volumetric_far_point_cloud)

        

        # o3d.io.write_point_cloud(dataset_directory + pc_updater.dataset_name + bin_name +  "/orifice_center_pc.ply", pc_updater.orifice_center_point_cloud)

        

        o3d.io.write_point_cloud(self.write_folder +  "/volumetric_near_lumen_point_cloud.ply", pc_updater.volumetric_near_point_cloud)

        o3d.io.write_point_cloud(self.write_folder + "/volumetric_far_lumen_point_cloud.ply", pc_updater.volumetric_far_point_cloud)

        

        o3d.io.write_point_cloud(self.write_folder +  "/orifice_center_pc.ply", pc_updater.orifice_center_point_cloud)

        if(self.record==1):

            folder_path = self.write_folder + '/grayscale_images'
            create_folder(folder_path)

            folder_path = self.write_folder + '/transform_data'
            create_folder(folder_path)

            folder_path = self.write_folder + '/ecg_signal'
            create_folder(folder_path)

            print("started saving")
            # Iterate through the image and TW_EM batches simultaneously
            for i, (grayscale_image, TW_EM) in enumerate(zip(self.image_batch, self.tw_em_batch)):
                if i % save_frequency == 0:
                    # Save the image
                    image_filename = f'{self.write_folder}/grayscale_images/grayscale_image_{self.image_call + i}.npy'
                    with open(image_filename, 'wb') as f:
                        np.save(f, grayscale_image)
                    

                    # Save the TW_EM data
                    tw_em_filename = f'{self.write_folder}/transform_data/TW_EM_{self.image_call + i}.npy'
                    with open(tw_em_filename, 'wb') as f:
                        np.save(f, TW_EM)

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

        return

            

if __name__ == '__main__':
    try:
        rospy.init_node('open3d_visualizer')
        pc_updater = PointCloudUpdater()        
        rospy.on_shutdown(pc_updater.save_image_and_transform_data)
        rospy.spin()
        
    except rospy.ROSInterruptException:
        self.vis.destroy_window()
        pass
