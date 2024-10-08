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
import random
import threading
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
        with open('mapping_parameters_real_time.yaml', 'r') as file:
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

        # Set default parameter values if they don't exist
        for param_name in param_names:
            if not rospy.has_param(param_name):
                rospy.set_param(param_name, default_values.get(param_name, None))
                rospy.loginfo(f"Initialized {param_name} with default value: {default_values.get(param_name, None)}")


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

        # ---- LOAD ROS PARAMETERS ------ #
        param_names = ['/angle', '/translation','/scaling','/threshold','/no_points','/crop_index','/radial_offset','/oclock']

        # fetch these from yaml instead..
        self.default_values=load_default_values()

        # mapping specific parameters
        self.default_values['/no_points'] = 1000

        no_points=self.default_values['/no_points']
        # override previous threshold
        self.default_values['/threshold'] = 50

        self.default_values['/crop_index'] = 60


        # ---- INITIALIZE VISUALIZER ----- #
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.get_render_option().mesh_show_back_face = True
        
        # ----- INITIALIZE RECONSTRUCTIONS ------ #
        self.near_point_cloud = o3d.geometry.PointCloud()
        self.far_point_cloud = o3d.geometry.PointCloud()
        self.dissection_flap_point_cloud = o3d.geometry.PointCloud()
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.near_point_cloud)
        self.vis.add_geometry(self.far_point_cloud)
        self.vis.add_geometry(self.dissection_flap_point_cloud)
        self.vis.add_geometry(self.point_cloud)



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
        self.image_width=1280
        self.image_height=1024

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
        

        # this is the shortest number of lines it takes to grab the transform once image arrives
        transform_time = rospy.Time(0) #get the most recent transform
        
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
        

        grayscale_image=preprocess_ivus_image(rgb_image,self.box_crop,self.circle_crop,self.text_crop,self.crosshairs_crop)
        
        # how to make this faster
        if(self.record==1):
            # code for saving images

            self.image_batch.append(grayscale_image)
            self.tw_em_batch.append(TW_EM)
            self.image_times.append(image_timestamp_in_seconds)

        # can stop here and do all processing / reconstruction later
        if(tsdf_map != 1):
            continue

        # fetch rospy parameters for real time mapping (could be placed in initialization)
        threshold = rospy.get_param('threshold')
        no_points = rospy.get_param('no_points')
        crop_index = rospy.get_param('crop_index')
        scaling = rospy.get_param('scaling')
        angle = rospy.get_param('angle')
        translation = rospy.get_param('translation')
        radial_offset = rospy.get_param('radial_offset')
        oclock = rospy.get_param('oclock')

        centre_x=self.centre_x
        centre_y=self.centre_y

        # check if rospy parameter has changed:
        if no_points != self.previous_no_points:
            self.gridlines=get_gridlines(centre_x,centre_y,no_points, self.crop_radius)
            self.previous_no_points=no_points

        # first return segmentation
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


        # FOR DISSECTION IMPLEMENTATION LATER

        # # note 224,224 image for compatibility with network is hardcoded
        # grayscale_image = cv2.resize(grayscale_image, (224, 224))
        # image = cv2.cvtColor(grayscale_image,cv2.COLOR_GRAY2RGB)


        # #---------- SEGMENTATION --------------#

        # mask_1, mask_2 = deeplumen_segmentation(image,model)


        # # ----- GET CONVEX HULL
        # combined_mask = cv2.bitwise_or(mask_1, mask_2)
        # combined_mask = np.uint8(combined_mask)
        # contours,hier = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # length = len(contours)
        # cont = [contours[i] for i in range(length)]
        # cont = np.vstack(cont)
        # hull = cv2.convexHull(cont)
        


        # # ----- DISSECTION PARAMETERIZATION -------- #
        # if(dissection_parameterize == 1):
        #     try:
        #         near_lumen_edge, far_lumen_edge, dissection_flap_skeleton, convex_hull_flap = parameterize_dissection(mask_1,mask_2)
        #         clean_image = grayscale_image

        #         inv_hull_mask = get_inv_convex_hull_mask(clean_image,near_lumen_edge,far_lumen_edge,dissection_flap_skeleton)


        #         flap_esdf_image, modified_flap_esdf_image=flap_image_for_killingfusion_2(clean_image, dissection_flap_skeleton, mask_1)

        #         for point in near_lumen_edge :
        #             cv2.circle(clean_image, tuple(point), 1, (0,0,255), -1)  # Draw points

        #         for point in far_lumen_edge :
        #             cv2.circle(clean_image, tuple(point), 1, (255,0,0), -1)  # Draw points

        #         for point in dissection_flap_skeleton :
        #             cv2.circle(clean_image, tuple(point), 1, (0,255,0), -1)  # Draw points

                

        #         # cv2.imshow('Skeleton extracted', clean_image)
        #         # cv2.waitKey(5)
        #     except:
        #         print("parameterization failed!")
        #         return


        # ---- SCALING AND PADDING ----- #

        # mask_1_contour,hier = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # mask_2_contour,hier = cv2.findContours(mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # visualize segmentations
        # grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(grayscale_image, mask_1_contour, -1, (0, 0, 255), thickness=2)
        # cv2.drawContours(grayscale_image, mask_2_contour, -1, (255, 0, 0), thickness=2)
        # cv2.imshow("final_image", cv2.resize(grayscale_image, (224 * 2, 224 * 2)))

        scaling=self.default_values['/scaling'] 
        # if dissection_parameterize == 0:
        #     three_d_points, three_d_points_near_lumen,three_d_points_far_lumen,three_d_points_dissection_flap = get_point_cloud_from_masks(combined_mask, scaling, mask_1_contour,mask_2_contour)

        # if dissection_parameterize == 1:
        #     three_d_points, three_d_points_near_lumen,three_d_points_far_lumen,three_d_points_dissection_flap = get_point_cloud_from_masks(combined_mask, scaling, mask_1_contour,mask_2_contour, dissection_flap_skeleton)


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
        # if(three_d_points_near_lumen is not None):
        #     self.update_tsdf_mesh(self.vis, self.tsdf_volume_near_lumen,self.mesh_near_lumen,three_d_points_near_lumen, extrinsic_matrix,[1,0,0])
        # if(three_d_points_far_lumen is not None):
        #     self.update_tsdf_mesh(self.vis,self.tsdf_volume_far_lumen,self.mesh_far_lumen,three_d_points_far_lumen, extrinsic_matrix,[0,0,1])
        # if(tsdf_map == 1):
        #     if(three_d_points_dissection_flap is not None):
        #         self.update_tsdf_mesh(self.vis,self.tsdf_volume_dissection_flap,self.mesh_dissection_flap,three_d_points_dissection_flap, extrinsic_matrix,[0,1,0])

  
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








    def save_image_and_transform_data(self):

        save_frequency = 1

    
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

            if(gating == 1):
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
