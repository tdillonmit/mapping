import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped
import cv2
import tf.transformations
import tf2_ros
import open3d as o3d
import numpy as np
import copy
from cv_bridge import CvBridge
from threading import Thread
from std_msgs.msg import Header  # Import Header
import yaml
import glob
import os
import math
import time


def save_parameters_on_shutdown():
    # Get the parameter values
    params = rospy.get_param_names()
    param_values = {param: rospy.get_param(param) for param in params}

    print("Saving rospy calibration parameters")
    # Save the parameters to a YAML file
    with open('/home/tdillon/mapping/registration_parameters_ice.yaml', 'w') as file:
        yaml.dump(param_values, file)

def align_vectors(a, b):
    b = b / np.linalg.norm(b) # normalize a
    a = a / np.linalg.norm(a) # normalize b
    v = np.cross(a, b)
    # s = np.linalg.norm(v)
    c = np.dot(a, b)

    v1, v2, v3 = v
    h = 1 / (1 + c)

    Vmat = np.array([[0, -v3, v2],
                  [v3, 0, -v1],
                  [-v2, v1, 0]])

    R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
    return R


def angle_between_vectors(vector_1,vector_2):
    

    angle=np.arccos((np.dot(vector_1,vector_2))/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2)))
    angle_radians=angle
    
    return angle*(180/np.pi), angle_radians

def define_2D_CS(origin,x_point,y_point):
    
    basis_x=x_point-origin
    norm=np.linalg.norm(basis_x)
    basis_x=basis_x/norm
    
    basis_y=y_point-origin
    norm=np.linalg.norm(basis_y)
    basis_y=basis_y/norm
    
    basis_z=np.cross(basis_x,basis_y)
    rotation_2D=np.asarray([basis_x,basis_y,basis_z])
    
    rotation_2D=np.transpose(rotation_2D)
    
    return rotation_2D

def create_line(point_1,point_2):
    
    linepoints=o3d.cpu.pybind.utility.Vector3dVector(np.array([point_1,point_2]))
    indices=np.array([[0,1]])
    connections=o3d.cpu.pybind.utility.Vector2iVector(indices)
    lineset=o3d.geometry.LineSet(linepoints,connections)
    return lineset

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

def verify_scaling(across_model,across_em):
   
    distance=np.linalg.norm(across_em)
    measured_reference=np.linalg.norm(across_model)
    EM_scaling=distance/measured_reference

    print("EM to real scaling is", EM_scaling)
    print("this number should be close to 1.0!")

def get_normalized_rejection(A,B):

    # Calculate the dot product of A and B
    dot_product = np.dot(A, B)

    # Calculate the magnitude (squared) of B
    magnitude_squared = np.dot(B, B)

    # Calculate the vector rejection
    rejection = A - (dot_product / magnitude_squared) * B

    rejection_magnitude = np.linalg.norm(rejection)

    # Normalize the rejection vector
    normalized_rejection = rejection / rejection_magnitude

    return normalized_rejection


def revised_fiducial_model_registration(model, fiducial_1,reference_1):

    # simplify string phantom
    model  = model .subdivide_midpoint(number_of_iterations=3)
    point_cloud_1=o3d.geometry.PointCloud()
    point_cloud_1.points=model.vertices
    point_cloud_1.paint_uniform_color([0, 0, 1])
    string_phantom=point_cloud_1

    # # prerotation=np.array([[1, 0, 0],[0, np.cos(-0.05),-np.sin(-0.05)],[0, np.sin(-0.05), np.cos(-0.05)]])
    # prerotation=np.array([[np.cos(0.02), 0, np.sin(0.02)],[0, 1,0],[-np.sin(0.02), 0, np.cos(0.02)]])
    # prerotate_roll=np.eye(4)
    # prerotate_roll[0:3,0:3]=prerotation
    # string_phantom.transform(prerotate_roll)
  
    # # prerotation=np.array([[np.cos(-0.02),-np.sin(-0.02), 0],[np.sin(-0.02), np.cos(-0.02),0],[0, 0, 1]])
    # # prerotation=np.array([[np.cos(0.035),-np.sin(0.035), 0],[np.sin(0.035), np.cos(0.035),0],[0, 0, 1]])
    # prerotation=np.array([[np.cos(-0.065),-np.sin(-0.065), 0],[np.sin(-0.065), np.cos(-0.065),0],[0, 0, 1]])
    # prerotate_roll=np.eye(4)
    # prerotate_roll[0:3,0:3]=prerotation
    # string_phantom.transform(prerotate_roll)

    # get the fiducial transform
    fiducial_1=transform_stamped_to_matrix(fiducial_1)
    print("fiducial_1 is", fiducial_1)
    fiducial_1_position=fiducial_1[0:3,3]
    print("fiducial_1_position is", fiducial_1_position)

    # make copies for registration - needed?
    rotated_model=copy.deepcopy(string_phantom)

    # grab normal of 2000th fiducial1 and 2
    x_axis=fiducial_1[:3,0]
    print("x axis of fiducial is", x_axis)
 
    # rotate model y to fiducial x using align vectors
    # alignment_rotation=align_vectors([0,-1,0], x_axis)
    
    angle_degrees,angle_radians=angle_between_vectors([0,1],x_axis[:2])

    # small optional prerotation - this is specifc to a particular sensor mounting
    # prerotation=(-2.5)*(np.pi/180) 

    # note no prerotation when saving for later...
    # prerotation because the hole wasn't drilled properly
    prerotation=(-3.25)*(np.pi/180) 
    # prerotation=(0)*(np.pi/180) 
    angle_radians=angle_radians+prerotation
    alignment_rotation=[[np.cos(angle_radians),-np.sin(angle_radians),0],[np.sin(angle_radians),np.cos(angle_radians),0],[0,0,1]]

    # A= alignment_rotation[:, 0]
    # B = np.array([0, 0, 1])

    # alignment_rotation[:, 0]=get_normalized_rejection(A,B)

    # A= alignment_rotation[:, 1]
    # B = np.array([0, 0, 1])

    # alignment_rotation[:, 1]=get_normalized_rejection(A,B)

    # alignment_rotation[2,0]=0
    # alignment_rotation[2,1]=0
    # alignment_rotation[0,2]=0
    # alignment_rotation[0,1]=0
    # alignment_rotation[2,2]=1

    alignment_transform=np.eye(4)
    alignment_transform[0:3,0:3]=alignment_rotation
    # alignment_transform=get_transform_inverse(alignment_transform)
    
    rotated_model.transform(alignment_transform)

    # get required translation based on prior sum of rotations
    reference_pC=o3d.geometry.PointCloud()
    reference_pC.points=o3d.utility.Vector3dVector([reference_1,[0,0,0]])
    reference_pC.transform(alignment_transform)
    new_point=np.asarray(reference_pC.points)

    # apply model reference point to EM tracked fiducial
    tracker_2_displacement=fiducial_1_position-new_point[0,:]
    rotated_model.translate(tracker_2_displacement)

    # maybe include some angle verification

    return rotated_model


def one_fiducial_model_registration(model, fiducial_1,reference_1):

    # simplify string phantom
    model  = model .subdivide_midpoint(number_of_iterations=3)
    point_cloud_1=o3d.geometry.PointCloud()
    point_cloud_1.points=model.vertices
    point_cloud_1.paint_uniform_color([0, 0, 1])
    string_phantom=point_cloud_1

    # # prerotation=np.array([[1, 0, 0],[0, np.cos(-0.05),-np.sin(-0.05)],[0, np.sin(-0.05), np.cos(-0.05)]])
    # prerotation=np.array([[np.cos(0.02), 0, np.sin(0.02)],[0, 1,0],[-np.sin(0.02), 0, np.cos(0.02)]])
    # prerotate_roll=np.eye(4)
    # prerotate_roll[0:3,0:3]=prerotation
    # string_phantom.transform(prerotate_roll)
  
    # # prerotation=np.array([[np.cos(-0.02),-np.sin(-0.02), 0],[np.sin(-0.02), np.cos(-0.02),0],[0, 0, 1]])
    # # prerotation=np.array([[np.cos(0.035),-np.sin(0.035), 0],[np.sin(0.035), np.cos(0.035),0],[0, 0, 1]])
    # prerotation=np.array([[np.cos(-0.065),-np.sin(-0.065), 0],[np.sin(-0.065), np.cos(-0.065),0],[0, 0, 1]])
    # prerotate_roll=np.eye(4)
    # prerotate_roll[0:3,0:3]=prerotation
    # string_phantom.transform(prerotate_roll)

    # get the fiducial transform
    fiducial_1=transform_stamped_to_matrix(fiducial_1)
    print("fiducial_1 is", fiducial_1)
    fiducial_1_position=fiducial_1[0:3,3]
    print("fiducial_1_position is", fiducial_1_position)

    temp_em_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
    temp_em_frame.scale(0.025,center=[0,0,0])
    temp_em_frame.transform(fiducial_1)

    baseframe=o3d.geometry.TriangleMesh.create_coordinate_frame()
    baseframe.scale(0.025,center=[0,0,0])


    # make copies for registration - needed?
    rotated_model=copy.deepcopy(string_phantom)

    # grab normal of 2000th fiducial1 and 2
    x_axis=fiducial_1[:3,0]
    print("x axis of fiducial is", x_axis)

    # rotate model y to fiducial x using align vectors
    alignment_rotation=align_vectors([0,-1,0], x_axis)
    alignment_transform=np.eye(4)
    alignment_transform[0:3,0:3]=alignment_rotation
    # alignment_transform=get_transform_inverse(alignment_transform)
    rotated_model.transform(alignment_transform)

    reference_pC=o3d.geometry.PointCloud()
    reference_pC.points=o3d.utility.Vector3dVector([reference_1,[0,0,0]])
    reference_pC.transform(alignment_transform)
    new_point=np.asarray(reference_pC.points)

    # rotate model z to model
    alignment_rotation=align_vectors([0,-1,0], x_axis)
    alignment_transform=np.eye(4)
    alignment_transform[0:3,0:3]=alignment_rotation
    # alignment_transform=get_transform_inverse(alignment_transform)
    rotated_model.transform(alignment_transform)

    reference_pC=o3d.geometry.PointCloud()
    reference_pC.points=o3d.utility.Vector3dVector([reference_1,[0,0,0]])
    reference_pC.transform(alignment_transform)
    new_point=np.asarray(reference_pC.points)

    # apply model reference point to EM tracked fiducial
    tracker_2_displacement=fiducial_1_position-new_point[0,:]
    rotated_model.translate(tracker_2_displacement)

    # maybe include some angle verification

    return rotated_model

class Registerer:
    def __init__(self):
        
        print("booting visualizer")

        

        # self.frame_scaling=0.025


        # self.transform_sub = rospy.Subscriber('/ascension_node/target_poses', TransformStamped, self.transform_callback)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # be careful getting the transformations at boot - better to take an average?
        # CHANGE THIS PATH
        model_mesh = o3d.io.read_triangle_mesh('/home/tdillon/mapping/src/calibration_ros/src/open3d_phantom_with_strings.stl')
        model_mseh = model_mesh.scale(0.001,center=[0,0,0])

     

        while not rospy.is_shutdown():
            try:
                if self.tf_buffer.can_transform('ascension_origin', 'target1', rospy.Time(0)):
                    fiducial_1=self.tf_buffer.lookup_transform('ascension_origin', 'target1', rospy.Time(0))
                    break  # Exit the loop if the transform is available
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Waiting for the transform 'ascension_origin' to 'target1'")
    

        # fiducial_1 = self.tf_buffer.lookup_transform('ascension_origin','target2', rospy.Duration(0))
        print("fiducial 1 is", fiducial_1)
        # fiducial_2 = self.tf_buffer.lookup_transform('ascension_origin','target3',  rospy.Time.now()- rospy.Duration(0.02))

        # chatgpt look up many transforms?
        # fiducial_1 = np.asarray([np.mean(tx1[2000:]),np.mean(ty1[2000:]),np.mean(tz1[2000:])])
        # fiducial_2 = np.asarray([np.mean(tx2[2000:]),np.mean(ty2[2000:]),np.mean(tz2[2000:])]) 

        # model_1_position=np.asarray([0.0295,0.1045,-0.0115]) 
        # add 0.0005 in x direction, 0.01 in y direction for new hole
        # model_1_position=np.asarray([0.03,0.1045,-0.0215]) 
        # artifically add small displacement
        model_1_position=np.asarray([0.032,0.005,-0.0215])  
 
        # model_2_position=np.asarray([0.665,0.55,-0.9])*10
        # model=one_fiducial_model_registration(model_mesh,fiducial_1, model_1_position, self.vis.add_geometry(self.em_1_frame))
        print("about to register..")
        rospy.sleep(1.0)

       
        model=revised_fiducial_model_registration(model_mesh,fiducial_1, model_1_position)
        
        # self.vis.add_geometry(self.tracker_frame)
        # self.vis.add_geometry(self.em_1_frame)
        # self.vis.add_geometry(self.baseframe)
        # self.vis.add_geometry(model)
        
        # print("tf listener:", self.tf_listener)
        # print("successfully initialized registerer")

        # just save the point cloud to a npy file
        
        o3d.io.write_point_cloud("/home/tdillon/mapping/registration.ply", model)
        print("point cloud registration saved")
                

if __name__ == '__main__':
    try:
        print("started main function")
        rospy.sleep(2.0)
        rospy.init_node('open3d_visualizer')
        # fiducial_ = self.tf_buffer.lookup_transform('ascension_origin','target2', rospy.Duration(0))
        print("initialization successful, back in main function")
        pc_updater = Registerer()
        # rospy.on_shutdown(save_parameters_on_shutdown)
        # rospy.spin()
        
        
    except rospy.ROSInterruptException:
        self.vis.destroy_window()
        pass
