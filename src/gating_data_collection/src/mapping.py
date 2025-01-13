#!/usr/bin/env python3.9

import rospy
import sys
print(f"Python version: {sys.version}")
print(f"Version info: {sys.version_info}")
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped
import cv2
import tf.transformations
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





def make_crosshairs_crop(new_height,new_width,crosshair_width,crosshair_height,crosshair_vert_coordinates,crosshair_horiz_coordinates):

    # don't want to do this every iteration
    crosshairs_crop=np.ones((new_height, new_width), dtype=np.uint8)*255

    for coordinate in crosshair_vert_coordinates:
        rect_start=coordinate
        rect_end=[coordinate[0]+crosshair_width,coordinate[1]+crosshair_height]
        cv2.rectangle(crosshairs_crop, tuple(rect_start), tuple(rect_end), 0, thickness=cv2.FILLED)

    for coordinate in crosshair_horiz_coordinates:
        rect_start=coordinate
        rect_end=[coordinate[0]+crosshair_height,coordinate[1]+crosshair_width]
        cv2.rectangle(crosshairs_crop, tuple(rect_start), tuple(rect_end), 0, thickness=cv2.FILLED)

    return crosshairs_crop

def make_wire_crop(new_height,new_width,centre_x,centre_y, radius_wire):
    mask = np.ones((new_height, new_width), dtype=np.uint8)*255
    cv2.circle(mask, (centre_x,centre_y), radius_wire, 0, thickness=cv2.FILLED)
    return mask

def make_circle_crop(new_height,new_width,centre_x,centre_y):
    mask = np.zeros((new_height, new_width), dtype=np.uint8)
    radius=min(centre_x,centre_y) 
    cv2.circle(mask, (centre_x,centre_y), radius, 255, thickness=cv2.FILLED)
    return mask

def preprocess_ivus_image(original_image,box_crop,circle_crop,text_crop,crosshairs_crop):

    # initial box crop
    cropped_image=original_image[box_crop[0]:box_crop[1],box_crop[2]:box_crop[3],:]

    # grayscale the image
    grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

    # radius cropping
    result = cv2.bitwise_and(grayscale_image, grayscale_image, mask=circle_crop)

    # crosshairs crop
    result = cv2.bitwise_and(result, result, mask=crosshairs_crop)

    # final text cropping - is this consuming?
    cv2.rectangle(result, (text_crop[0],text_crop[1]), (text_crop[2],text_crop[3]), 0, thickness=cv2.FILLED)
    

    return result

def load_default_values(file_path='/home/tdillon/mapping/calibration_parameters_ivus.yaml'):
    with open(file_path, 'r') as stream:
        
        default_values={}

        try:
            all_parameters = yaml.safe_load(stream)
            # Convert the loaded angle to radians if needed
            if '/angle' in all_parameters:
                default_values['/angle'] = all_parameters['/angle']

            if '/threshold' in all_parameters:
                default_values['/threshold'] = all_parameters['/threshold']

            if '/translation' in all_parameters:
                default_values['/translation'] = all_parameters['/translation']

            if '/scaling' in all_parameters:
                default_values['/scaling'] = all_parameters['/scaling']

            if '/radial_offset' in all_parameters:
                default_values['/radial_offset'] = all_parameters['/radial_offset']

            if '/oclock' in all_parameters:
                default_values['/oclock'] = all_parameters['/oclock']
            
            return default_values
        except yaml.YAMLError as exc:
            print(exc)

def get_gridlines(centre_x,centre_y, no_points, crop_radius):

    # should centre_x and centre_y be swapped?
    radius_ivus=centre_x-crop_radius
    gridlines=[]

    #pull out all the voxels in each angular direction
    for theta in np.arange(0,2*np.pi,2*np.pi/no_points):
        
            #2D voxel traversal
            strtPt=np.asarray([centre_x,centre_y])
            endPt_x=centre_x+((radius_ivus)*np.cos(theta))
            endPt_y=centre_y+((radius_ivus)*np.sin(theta))
            endPt=[endPt_x,endPt_y]
            #not yielding an ordered list
            this_gridline=getIntersectPts(strtPt, endPt, geom=[0,1,0,0,0,1])
            this_gridline=np.asarray(list(this_gridline))
            this_gridline=np.round(this_gridline.astype(int))
            
            delta=this_gridline-[centre_x,centre_y]
            distances=np.linalg.norm(delta,axis=1)
            sort_indices=np.argsort(distances)
                    
            ordered_gridline=this_gridline[sort_indices]
            gridlines.append(ordered_gridline)

    return gridlines


def getIntersectPts(strPt, endPt, geom=[0,1,0,0,0,1]):

    x0 = geom[0]
    y0 = geom[3]

    (sX, sY) = (strPt[0], strPt[1])
    (eX, eY) = (endPt[0], endPt[1])
    xSpace = geom[1]
    ySpace = geom[5]

    sXIndex = ((sX - x0) / xSpace)
    sYIndex = ((sY - y0) / ySpace)
    eXIndex = ((eX - sXIndex) / xSpace) + sXIndex
    eYIndex = ((eY - sYIndex) / ySpace) + sYIndex


    dx = (eXIndex - sXIndex)
    dy = (eYIndex - sYIndex)
    xHeading = 1.0 if dx > 0 else -1.0 if dx < 0 else 0.0
    yHeading = 1.0 if dy > 0 else -1.0 if dy < 0 else 0.0

    xOffset = (1 - (math.modf(sXIndex)[0]))
    yOffset = (1 - (math.modf(sYIndex)[0]))

    ptsIndexes = []
    x = sXIndex
    y = sYIndex
    pt = (x, y) #1st pt

    if dx != 0:
        m = (float(dy) / float(dx))
        b = float(sY - sX * m )

    dx = abs(int(dx))
    dy = abs(int(dy))

    if dx == 0:
        for h in range(0, dy + 1):
            pt = (x, y + (yHeading *h))
            ptsIndexes.append(pt)

        return ptsIndexes


    #snap to half a cell size so we can find intersections on cell boundaries
    sXIdxSp = round(2.0 * sXIndex) / 2.0
    sYIdxSp = round(2.0 * sYIndex) / 2.0
    eXIdxSp = round(2.0 * eXIndex) / 2.0
    eYIdxSp = round(2.0 * eYIndex) / 2.0
    # ptsIndexes.append(pt)
    prevPt = False
    #advance half grid size
    for w in range(0, dx * 4):
        x = xHeading * (w / 2.0) + sXIdxSp
        y = (x * m + b)
        if xHeading < 0:
            if x < eXIdxSp:
                break
        else:
            if x > eXIdxSp:
                break

        pt = (round(x), round(y)) #snapToGrid
        # print(w, x, y)

        if prevPt != pt:
            ptsIndexes.append(pt)
            prevPt = pt
    #advance half grid size
    for h in range(0, dy * 4):
        y = yHeading * (h / 2.0) + sYIdxSp
        x = ((y - b) / m)
        if yHeading < 0:
            if y < eYIdxSp:
                break
        else:
            if y > eYIdxSp:
                break
        pt = (round(x), round(y)) # snapToGrid
        # print(h, x, y)

        if prevPt != pt:
            ptsIndexes.append(pt)
            prevPt = pt

    return set(ptsIndexes) #elminate duplicates

def get_transform_inverse(transform):
    disp=transform[:3, 3]
    T_inv=np.eye(4)
    T_inv[:3, 3]=(-np.transpose(transform[:3,:3])) @ (disp)
    T_inv[:3,:3]=np.transpose(transform[:3,:3])
    return T_inv

def transform_stamped_to_matrix(transform_stamped):
    translation = np.array([transform_stamped.transform.translation.x,
                            transform_stamped.transform.translation.y,
                            transform_stamped.transform.translation.z])

    #make sure you check the convention on x,y,z,w components and order - scalar vs vector part
    quaternion = np.array([transform_stamped.transform.rotation.x,
                           transform_stamped.transform.rotation.y,
                           transform_stamped.transform.rotation.z,
                           transform_stamped.transform.rotation.w])

    # Create a 4x4 transformation matrix
    transformation_matrix = tf.transformations.quaternion_matrix(quaternion)
    transformation_matrix[:3, 3] = translation

    return transformation_matrix

def first_return_segmentation(gray, threshold, ignore_up_to_index, gridlines):

    closest_pixel=[]

    

    for ordered_gridline in gridlines:

        try:
            
            ordered_intensities=gray[ordered_gridline[:,0],ordered_gridline[:,1]]
          
            cropped_intensities=ordered_intensities[ignore_up_to_index:]
            
            thresholded=cropped_intensities>threshold
            hyp_indices=np.squeeze(np.argwhere(thresholded))
            
            if(len(hyp_indices)!=0):
                min_d=np.min(hyp_indices)+ignore_up_to_index #maybe more efficient to pull out first element
                closest_pixel.append([ordered_gridline[min_d]])
                
        except:
            
            pass

    return closest_pixel

def get_box(min_bounds,max_bounds):
    
    box=o3d.geometry.AxisAlignedBoundingBox(min_bounds,max_bounds)
    box_lineset=o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
    colors = [[0, 0, 0] for i in range(np.shape(box_lineset.lines)[0])]
    box_lineset.colors = o3d.utility.Vector3dVector(colors)

    return box

def morphological_processing(input_image,median_kernel, closing_kernel, min_component_size, threshold):

    # start_time=time.time()
    
    # apply a median filter
    blurred_image = cv2.medianBlur(input_image, ksize=median_kernel)  # ksize is the kernel size, should be an odd number

    # smoothed = gaussian_filter( (img-img.min()) / (img.max()-img.min()), sigma )
    
    # apply adaptive thresholding
    _,thresholded_image = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY)
    # _,thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # closing operations
    closed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, closing_kernel)

    # remove small components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_image, connectivity=8)

    # Create an empty mask to store the result
    # Iterate through the components and remove small components directly from the binary mask
    # largest_labels = np.argsort(stats[1:, cv2.CC_STAT_AREA])[-2:] + 1  # Add 1 to skip the background label (0)

    # # Create a mask for the two largest components
    # filtered_mask = np.zeros_like(closed_image)
    # for label in largest_labels:
    #     filtered_mask[labels == label] = 255

    # closed_image=filtered_mask
    
    
    for label in range(1, num_labels):  # Skip the background label (0)
        if stats[label, cv2.CC_STAT_AREA] < min_component_size:
            closed_image[labels == label] = 0

    # end_time=time.time()
    # elapsed_time = end_time - start_time
    # print(f"morphological processing took: {elapsed_time:.5f} seconds")

    # cv2.imshow('input Image', input_image)
    # cv2.imshow('blurred image', blurred_image)
    # cv2.imshow('thresholded Image', thresholded_image)
    # # add check for component removal
    # cv2.imshow('closed Image', closed_image)
    # key = cv2.waitKey(0)

    return closed_image, labels, stats


class PointCloudUpdater:



    def __init__(self):
        
        print("booting visualizer")

        # Define parameter names and default values
        param_names = ['/angle', '/translation','/scaling','/threshold','/no_points','/crop_index','/radial_offset','/oclock']

        # fetch these from yaml instead..
        # default_values = {'/angle': np.pi/2, '/translation': 0.01, '/scaling': 0.000075, '/threshold':100}
        default_values=load_default_values()

        # mapping specific parameters
        default_values['/no_points'] = 1000
        # override previous threshold
        default_values['/threshold'] = 50

        default_values['/crop_index'] = 60

        # Set default parameter values if they don't exist
        for param_name in param_names:
            if not rospy.has_param(param_name):
                rospy.set_param(param_name, default_values.get(param_name, None))
                rospy.loginfo(f"Initialized {param_name} with default value: {default_values.get(param_name, None)}")

        self.point_cloud = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.get_render_option().mesh_show_back_face = True

        self.frame_scaling=0.025
        self.tracker_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.tracker_frame.scale(self.frame_scaling,center=[0,0,0])

        self.second_tracker=0
        if(self.second_tracker==1):
            self.em_1_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
            self.em_1_frame.scale(self.frame_scaling,center=[0,0,0])
            self.vis.add_geometry(self.em_1_frame)

        self.baseframe=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.baseframe.scale(self.frame_scaling,center=[0,0,0])

        self.us_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.us_frame.scale(self.frame_scaling,center=[0,0,0])
        
        self.previous_transform=np.eye(4)
        self.previous_transform_1=np.eye(4)
        self.previous_transform_us=np.eye(4)


        #setup subscribers
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        # self.transform_sub = rospy.Subscriber('/ascension_node/target_poses', TransformStamped, self.transform_callback)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.lock = threading.Lock()  # A lock to ensure thread safety
        self.ecg_sub = rospy.Subscriber('ecg', Int32, self.ecg_callback)
        self.tf_sub = rospy.Subscriber('tf', TFMessage, self.tf_callback)
        
        print("ecg signal subscriber:", self.ecg_sub)
        print("frame grabber subscriber:", self.image_sub)
        print("tf subscriber:", self.tf_sub)
        print("tf listener:", self.tf_listener)
        print("successfully initialized visualizer!")

        #setup publishers
        self.binary_image_pub = rospy.Publisher('/binary_image', Image, queue_size=1)
        self.rgb_image_pub = rospy.Publisher('/rgb_image', Image, queue_size=1)


        self.image_width = rospy.get_param('/usb_cam/image_width', default=1280)
        self.image_height = rospy.get_param('/usb_cam/image_height', default=1024)

        self.crop_radius=10
        no_points=rospy.get_param('no_points')
        self.previous_no_points=no_points
        
      

        # assume these are ideal bounds
        min_bounds=np.array([0.05,-0.1,-0.1]) 
        max_bounds=np.array([0.3,0.1,0.1]) 
        self.box=get_box(min_bounds,max_bounds)

        self.vis.add_geometry(self.point_cloud)
        self.vis.add_geometry(self.us_frame)
        self.vis.add_geometry(self.tracker_frame)

        self.vis.add_geometry(self.baseframe)
        self.vis.add_geometry(self.box)

        # self.transform_sub = rospy.Subscriber('/tf', TransformStamped, self.transform_callback)

        # image preprocessing parameters
        start_x=59
        end_x=840
        start_y=10
        end_y=790
        self.box_crop=[start_x,end_x,start_y,end_y]

        text_start_x=393
        text_end_x=440
        text_start_y=0
        text_end_y=15
        self.text_crop=[text_start_x,text_start_y,text_end_x,text_end_y]
        
        self.new_height=end_x-start_x
        self.new_width=end_y-start_y
        self.centre_x=int(self.new_height/2)
        self.centre_y=int(self.new_width/2)
        # THIS NEEDS TO BE CHANGED IN ALL VERSIONS OF CODE!
        self.gridlines=get_gridlines(self.centre_x,self.centre_y,no_points, self.crop_radius)
        
        crosshair_width=5
        crosshair_height=2
        crosshair_vert_coordinates=[[380,62],[380,128],[380,193],[380,259],[380,324],[394,455],[394,520],[394,585],[394,651],[394,717]]
        crosshair_horiz_coordinates=[[61,395],[127,395],[192,395],[258,395],[323,395],[454,381],[519,381],[585,381],[650,381],[716,381]]
        self.crosshairs_crop=make_crosshairs_crop(self.new_height,self.new_width,crosshair_width,crosshair_height,crosshair_vert_coordinates,crosshair_horiz_coordinates)
        self.circle_crop=make_circle_crop(self.new_height,self.new_width,self.centre_x,self.centre_y)

        # what number image callback are we on?
        self.image_call=1
        
        # make this a rospy parameter later to turn recording on and off and also clean the field of view
        self.record=1

        if(self.record==1):
            self.write_folder = rospy.get_param('~image_path')
            self.image_batch=[]
            self.tw_em_batch=[]
            
            self.ecg_times=[]
            self.ecg_signal=[]
            # self.tf_times=[]
            # self.tf_signal=[]
            self.image_times=[]


        self.voxel_size = 0.002
        sdf_trunc = 2 * self.voxel_size
        self.tsdf_volume = SimpleTsdfIntegrator(self.voxel_size, sdf_trunc)
        self.mesh=o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.mesh)

        # filtering parameters
        self.median_kernel=3
        closing_kernel_size=40
        self.closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
        self.min_component_size = 50

        # how certain do you need to be?
        self.saturation_value = 0.12

        # how much thickness for robustness?
        self.thickness = 30
        self.area_threshold = 10000

    # inefficient because there are so many transforms
    def tf_callback(self,tf_msg):

        # tf_timestamp_secs = tf_msg.header.stamp.secs
        # tf_timestamp_nsecs = tf_msg.header.stamp.nsecs
        
        # tf_timestamp_in_seconds = tf_timestamp_secs + (tf_timestamp_nsecs * 1e-9)

        # tf_value = tf_msg.data

        # if(self.record==1):
        #     self.tf_times.append(tf_timestamp_in_seconds)
        #     self.tf_signal.append(tf_value)

        # print(f"Received TF: {tf_value} @ time: {tf_timestamp_in_seconds}")

        # this may also be inefficient

        pass

    def ecg_callback(self,ecg_msg):

        current_time = rospy.get_rostime()
        ecg_timestamp_secs = current_time.secs
        ecg_timestamp_nsecs = current_time.nsecs
        
        ecg_timestamp_in_seconds = ecg_timestamp_secs + (ecg_timestamp_nsecs * 1e-9)

        ecg_value = ecg_msg.data

        if(self.record==1):
            self.ecg_times.append( ecg_timestamp_in_seconds)
            self.ecg_signal.append(ecg_value)

        print(f"Received ECG value: {ecg_value} @ time: {ecg_timestamp_in_seconds}")
        





    def image_callback(self, msg):
        

        # first and foremost fetch the transform
        timestamp_secs = msg.header.stamp.secs
        timestamp_nsecs = msg.header.stamp.nsecs
        image_timestamp_in_seconds = timestamp_secs + (timestamp_nsecs * 1e-9)

        # # Truncate nanoseconds to the first 4 decimal places
        # truncated_nsecs = timestamp_nsecs // 1000  # Keep only the first 4 decimal places

        # Create a rospy.Time object using the truncated timestamp information
        timestamp = rospy.Time(secs=timestamp_secs, nsecs=timestamp_nsecs)

        if(self.record==1):
            self.image_times.append(image_timestamp_in_seconds)



        # Calculate transform_time by subtracting a duration
        # transform_time = timestamp - rospy.Duration(0.02)
        transform_time = rospy.Time(0) #i think this is better??

        # this code should be in a function
        
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

        if(self.second_tracker==1):
            ref_frame = 'ascension_origin'
            dest_frame = 'target2'

            try:
                # Lookup transform
                TW_EM_1 = self.tf_buffer.lookup_transform(ref_frame, dest_frame, transform_time)
            except (rospy.ROSException, tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Failed to lookup transform")
                TW_EM_1 = None
            
            TW_EM_1=transform_stamped_to_matrix(TW_EM_1)


        # Assuming RGB format
        rgb_image_data = np.frombuffer(msg.data, dtype=np.uint8)

        # Reshape the RGB data
        rgb_image = rgb_image_data.reshape((self.image_height, self.image_width, 3))

        # rgb_image = np.load('/media/tdillon/4D71-BDA7/frame_grabber_images/evar_graft/rgb_image_150.npy')

        # original_image=grayscale_image
        
        grayscale_image=preprocess_ivus_image(rgb_image,self.box_crop,self.circle_crop,self.text_crop,self.crosshairs_crop)
        
 
        
        # how to make this faster
        if(self.record==1):
            # code for saving images

            self.image_batch.append(grayscale_image)
            self.tw_em_batch.append(TW_EM)




        centre_x=self.centre_x
        centre_y=self.centre_y

        

        # #Crop the image - assuming 1280 x 1024 image
        # rgb_image=rgb_image[55:840,5:792,:]
    
        # #Extract luminance (Y) component
        # grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        # new_height, new_width = grayscale_image.shape
        # centre_x=new_height/2
        # centre_y=new_width/2

        

        
        
        # threshold segmentation
        threshold = rospy.get_param('threshold')
        no_points = rospy.get_param('no_points')
        crop_index = rospy.get_param('crop_index')

        # filtered_image, labels, stats=morphological_processing(grayscale_image, self.median_kernel,self.closing_kernel, self.min_component_size, threshold)
        # grayscale_image=filtered_image


        # check if rospy parameter has changed:
        if no_points != self.previous_no_points:
            # if resolution changes there may be a little lag...
            self.gridlines=get_gridlines(centre_x,centre_y,no_points, self.crop_radius)
            self.previous_no_points=no_points

        # do segmentation
        relevant_pixels=first_return_segmentation(grayscale_image,threshold, crop_index,self.gridlines)
        
        relevant_pixels=np.asarray(relevant_pixels).squeeze()
        # print("relevant pixels", relevant_pixels)

        binary_image=np.zeros_like(grayscale_image)
        binary_image[relevant_pixels[:, 0], relevant_pixels[:, 1]] = 255

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

        # header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        # binary_image_msg = Image(
        #     header=header,
        #     height=self.new_height,
        #     width=self.new_width,
        #     encoding='mono8',
        #     is_bigendian=False,
        #     step=self.new_width,
        #     data=grayscale_image.tobytes()
        # )
        # self.binary_image_pub.publish(binary_image_msg)
       

        # insert ellipse fitting!!
        ellipse_model = cv2.fitEllipse(relevant_pixels[:,[1,0]].astype(np.float32))  
        ellipse_contour= cv2.ellipse2Poly((int(ellipse_model[0][0]), int(ellipse_model[0][1])),
                                        (int(ellipse_model[1][0]/2), int(ellipse_model[1][1]/2)),
                                        int(ellipse_model[2]), 0, 360, 5)

        color_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(color_image, [ellipse_contour], -1, (0, 0, 255), 1)

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


        # only accept ransac fit if it improves over fitting to all points

        # ransac_contour=ellipse_RANSAC(relevant_pixels,num_iterations,fraction_random_points, inlier_threshold)

        # switch this as you need to
        relevant_pixels=ellipse_contour
        

        #you would convert the relevant coordinates to position in x and y - points expressed in camera frame
        centred_pixels=relevant_pixels - [centre_x,centre_y]


        scaling = rospy.get_param('scaling')
        two_d_points=centred_pixels*scaling


        #maybe try putting in shape[1] instead and set the other to -1
        three_d_points=np.hstack((np.zeros((two_d_points.shape[0], 1)),two_d_points)) #see function for this


        # read in calibration parameters - could change image to image
        angle = rospy.get_param('angle')
        # print("angle", angle)
        translation = rospy.get_param('translation')
        radial_offset = rospy.get_param('radial_offset')
        oclock = rospy.get_param('oclock')

        # Create a rotation matrix using the parameter value
        # TEM_C = [[1,0,0,translation],[0,np.cos(angle),-np.sin(angle),0],[0,np.sin(angle),np.cos(angle),0],[0, 0, 0, 1]]

        TEM_C = [[1,0,0,translation],[0,np.cos(angle),-np.sin(angle),radial_offset*np.cos(oclock)],[0,np.sin(angle),np.cos(angle),radial_offset*np.sin(oclock)],[0, 0, 0, 1]]

        #construct the point cloud object and transform it to WCS
        scan_points=o3d.geometry.PointCloud()
        scan_points.points=o3d.utility.Vector3dVector(three_d_points)
        scan_points.transform(TW_EM @ TEM_C)

        #modify the tracker frame (target1)
        # check what the arguments are for .transform - may be able to put in a quaternion directly...
        self.tracker_frame.transform(get_transform_inverse(self.previous_transform))
        self.tracker_frame.transform(TW_EM)
        self.previous_transform=TW_EM

        if(self.second_tracker==1):
            self.em_1_frame.transform(get_transform_inverse(self.previous_transform_1))
            self.em_1_frame.transform(TW_EM_1)
            self.previous_transform_1=TW_EM_1
            self.vis.update_geometry(self.em_1_frame)

        self.us_frame.transform(get_transform_inverse(self.previous_transform_us))
        self.us_frame.transform(TEM_C)
        self.us_frame.transform(TW_EM)
        self.previous_transform_us=TW_EM @ TEM_C
       
        # self.point_cloud.points.extend(scan_points.points)
        self.point_cloud.points = scan_points.points

        self.vis.update_geometry(self.point_cloud)
        self.vis.update_geometry(self.us_frame)
        self.vis.update_geometry(self.tracker_frame)
        

        self.tsdf_volume.integrate(points=three_d_points, extrinsic=TW_EM @ TEM_C)
        vertices, triangles = self.tsdf_volume.extract_triangle_mesh()
        self.mesh.triangles= o3d.utility.Vector3iVector(triangles)
        self.mesh.vertices= o3d.utility.Vector3dVector(vertices)
        self.mesh.merge_close_vertices(0.00001)
        if(np.shape(triangles)[0]>0):
            triangle_clusters, cluster_n_triangles, _ = (self.mesh.cluster_connected_triangles())
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            self.mesh.remove_triangles_by_mask(triangles_to_remove)
        self.mesh.compute_vertex_normals()
        self.vis.update_geometry(self.mesh)

        self.vis.poll_events()
        self.vis.update_renderer()


    #should this be removed??
    # def transform_callback(self, msg):
    #     # Process the robot transform data here
    #     pass

    # def run(self):
    #     while not rospy.is_shutdown():
            # Update the Open3D visualization with the point cloud
            # self.vis.clear_geometries()
            # self.vis.update_geometry(self.point_cloud)
            # self.vis.update_geometry(self.tracker_frame)
            # self.vis.update_geometry(self.baseframe)

            # self.vis.poll_events()
            # self.vis.update_renderer()

    def save_image_and_transform_data(self):

        save_frequency = 1

        

        if(self.record==1):
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


            # np.asarray(self.) 

            ecg_times = np.asarray(self.ecg_times)
            ecg_times_filename = f'{self.write_folder}/ecg_signal/ecg_times.npy'
            with open(ecg_times_filename, 'wb') as f:
                np.save(f, ecg_times)

            ecg_signal = np.asarray(self.ecg_signal)

            ecg_signal_filename = f'{self.write_folder}/ecg_signal/ecg_signal.npy'
            with open(ecg_signal_filename, 'wb') as f:
                np.save(f, ecg_signal)

            # tf_times = np.asarray(self.tf_times)

            # tf_times_filename = f'{self.write_folder}/transform_data/tf_times.npy'
            # with open(tf_times_filename, 'wb') as f:
            #     np.save(f, tf_times)

            # tf_signal = np.asarray(self.tf_signal)

            # tf_signal_filename = f'{self.write_folder}/transform_data/tf_signal.npy'
            # with open(tf_signal_filename, 'wb') as f:
            #     np.save(f, tf_signal)

            image_times = np.asarray(self.image_times) 

            image_times_filename = f'{self.write_folder}/ecg_signal/image_times.npy'
            with open(image_times_filename, 'wb') as f:
                np.save(f, image_times)

            print("finished saving images and transform data!")

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
