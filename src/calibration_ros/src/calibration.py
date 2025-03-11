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

def preprocess_ivus_image(original_image,box_crop,circle_crop,text_crop,crosshairs_crop,wire_crop):

    # initial box crop
    cropped_image=original_image[box_crop[0]:box_crop[1],box_crop[2]:box_crop[3],:]

    # grayscale the image
    grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

    # radius cropping
    result = cv2.bitwise_and(grayscale_image, grayscale_image, mask=circle_crop)

    # crosshairs crop
    result = cv2.bitwise_and(result, result, mask=crosshairs_crop)

    # final text cropping
    cv2.rectangle(result, (text_crop[0],text_crop[1]), (text_crop[2],text_crop[3]), 0, thickness=cv2.FILLED)

    # wire crop
    result = cv2.bitwise_and(result, result, mask=wire_crop)

    return result

def segment_centroids(thresh1,centre_x,centre_y):
    
    #detect connected components
    analysis = cv2.connectedComponentsWithStats(thresh1, 4)
    (totalLabels, label_ids, values, centroid) = analysis
    centroid_indices=np.round(centroid).astype(int)

    

    # Initialize a new image to
    # store all the output components
    output = np.zeros(thresh1.shape, dtype="uint8")


    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA]  
        # print("areas:", area)

        # this should be replaced with crop radius...
        # centroid_distance=np.linalg.norm(centroid_indices[i,:]-[centre_x,centre_y])
        if (area > 100):

            # Labels stores all the IDs of the components on the each pixel
            # It has the same dimension as the threshold
            # So we'll check the component
            # then convert it to 255 value to mark it white
            componentMask = (label_ids == i).astype("uint8") * 255

            # Creating the Final output mask
            output = cv2.bitwise_or(output, componentMask)

    analysis = cv2.connectedComponentsWithStats(output,
                                                4,
                                                cv2.CV_32S)

    (totalLabels, label_ids, values, centroid) = analysis

    # centroid_indices=np.round(centroid).astype(int)

    local_image = cv2.cvtColor(output,cv2.COLOR_GRAY2RGB)

    #find the top rod - lowest value of y
    centroid_indices=np.round(centroid).astype(int)

#     #final image
    for i in range(1, totalLabels):

        row=centroid_indices[i,:]
        average_x=np.mean(np.argwhere(label_ids==1)[:,0])
        average_y=np.mean(np.argwhere(label_ids==1)[:,1])
        # if(abs(row[0]-average_x) > abs(row[0]-average_y)):  
        local_image = cv2.circle(local_image, (row[0],row[1]), radius=3, color=(0, 0, 255), thickness=-1)
        # else:
        #     local_image = cv2.circle(local_image, (row[1],row[0]), radius=3, color=(0, 0, 255), thickness=-1)

    print("centroids are:", centroid)
    return centroid,local_image

def save_parameters_on_shutdown():
    # Get the parameter values
    params = rospy.get_param_names()
    param_values = {param: rospy.get_param(param) for param in params}

    print("Saving rospy calibration parameters")
    # Save the parameters to a YAML file
    with open('/home/tdillon/mapping/src/calibration_parameters_ivus.yaml', 'w') as file:
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

    # print("transform_stamped", transform_stamped.transform)

    translation = np.array([transform_stamped.transform.translation.x,
                            transform_stamped.transform.translation.y,
                            transform_stamped.transform.translation.z])

    # print("transform_stamped rotation", transform_stamped.transform.rotation)

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


def revised_fiducial_model_registration(model, fiducial_1,reference_1, em_frame):

    # simplify string phantom
    model  = model .subdivide_midpoint(number_of_iterations=3)
    point_cloud_1=o3d.geometry.PointCloud()
    point_cloud_1.points=model.vertices
    point_cloud_1.paint_uniform_color([0, 0, 1])
    string_phantom=point_cloud_1

    # baseframe=o3d.geometry.TriangleMesh.create_coordinate_frame()
    # baseframe.scale(0.025,center=[0,0,0])

    # o3d.visualization.draw_geometries([string_phantom, baseframe], mesh_show_back_face=True)

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

    # get the 2D vector?
    
    angle_degrees,angle_radians=angle_between_vectors([0,1],x_axis[:2])

    # small optional prerotation - this is specifc to a particular sensor mounting
    prerotation=(-3.25)*(np.pi/180) 
    # prerotation=(0)*(np.pi/180) 

    angle_radians=angle_radians+prerotation
    alignment_rotation=[[np.cos(angle_radians),-np.sin(angle_radians),0],[np.sin(angle_radians),np.cos(angle_radians),0],[0,0,1]]

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


def one_fiducial_model_registration(model, fiducial_1,reference_1, em_frame):

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

class PointCloudUpdater:
    def __init__(self):
        
        print("booting visualizer")

        radial_offset = rospy.get_param('radial_offset')
        oclock = rospy.get_param('oclock')

        # Define parameter names and default values
        param_names = ['/angle', '/translation','/scaling','/threshold', '/radial_offset', '/oclock']
        default_values = {'/angle': 3.24, '/translation': 0.009, '/scaling': 0.0000765, '/threshold':112, '/radial_offset':0.00225, '/oclock':2.16}

        # Set default parameter values if they don't exist
        for param_name in param_names:
            if not rospy.has_param(param_name):
                rospy.set_param(param_name, default_values.get(param_name, None))
                rospy.loginfo(f"Initialized {param_name} with default value: {default_values.get(param_name, None)}")

        self.spheres=o3d.geometry.TriangleMesh()
        self.sphere=o3d.geometry.TriangleMesh()
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.frame_scaling=0.025
        self.tracker_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.tracker_frame.scale(self.frame_scaling,center=[0,0,0])

        self.em_1_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.em_1_frame.scale(self.frame_scaling,center=[0,0,0])

        self.baseframe=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.baseframe.scale(self.frame_scaling,center=[0,0,0])

        self.us_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.us_frame.scale(self.frame_scaling,center=[0,0,0])

        self.previous_transform=np.eye(4)
        self.previous_transform_1=np.eye(4)
        self.previous_transform_us=np.eye(4)

    
        

        print("initializing frame grabber subscriber")
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        print("frame grabber initialization complete (note may not be valid still)")
        # self.transform_sub = rospy.Subscriber('/ascension_node/target_poses', TransformStamped, self.transform_callback)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # be careful getting the transformations at boot - better to take an average?
        # CHANGE THIS PATH
        model_mesh = o3d.io.read_triangle_mesh('/home/tdillon/mapping/src/calibration_ros/src/open3d_phantom_with_strings.stl')
        model_mseh = model_mesh.scale(0.001,center=[0,0,0])

     
        # COMPUTE REGISTRATION
        while not rospy.is_shutdown():
            try:
                if self.tf_buffer.can_transform('ascension_origin', 'target2', rospy.Time(0)):
                    fiducial_1=self.tf_buffer.lookup_transform('ascension_origin', 'target2', rospy.Time(0))
                    break  # Exit the loop if the transform is available
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Waiting for the transform 'ascension_origin' to 'target2'")
    

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

        # new hole
        # model_1_position=np.asarray([0.032,0.102,-0.0215])  

        model_1_position=np.asarray([0.032,0.005,-0.0215]) 
 
        # model_2_position=np.asarray([0.665,0.55,-0.9])*10
        # model=one_fiducial_model_registration(model_mesh,fiducial_1, model_1_position, self.vis.add_geometry(self.em_1_frame))
        model=revised_fiducial_model_registration(model_mesh,fiducial_1, model_1_position, self.vis.add_geometry(self.em_1_frame))

        # LOAD PRIOR REGISTRATION
        # model = load_previous_registration
        
        self.vis.add_geometry(self.spheres)
        self.vis.add_geometry(self.point_cloud)
        self.vis.add_geometry(self.tracker_frame)
        self.vis.add_geometry(self.em_1_frame)
        self.vis.add_geometry(self.baseframe)
        self.vis.add_geometry(self.us_frame)
        self.vis.add_geometry(model)
        
        print("frame grabber subscriber:", self.image_sub)
        print("tf listener:", self.tf_listener)
        print("successfully initialized visualizer!")

        
        # self.transform_sub = rospy.Subscriber('/tf', TransformStamped, self.transform_callback)

        # Start a separate thread for visualization
        # self.visualization_running = True
        # self.visualization_thread = Thread(target=self.image_processing_and_visualization)
        # self.visualization_thread.start()

        self.binary_image_pub = rospy.Publisher('/binary_image', Image, queue_size=1)
        self.rgb_image_pub = rospy.Publisher('/rgb_image', Image, queue_size=1)

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
        
        crosshair_width=5
        crosshair_height=2
        crosshair_vert_coordinates=[[380,62],[380,128],[380,193],[380,259],[380,324],[394,455],[394,520],[394,585],[394,651],[394,717]]
        crosshair_horiz_coordinates=[[61,395],[127,395],[192,395],[258,395],[323,395],[454,381],[519,381],[585,381],[650,381],[716,381]]
        self.crosshairs_crop=make_crosshairs_crop(self.new_height,self.new_width,crosshair_width,crosshair_height,crosshair_vert_coordinates,crosshair_horiz_coordinates)

        radius_wire=49
        self.wire_crop=make_wire_crop(self.new_height,self.new_width,self.centre_x,self.centre_y, radius_wire)
        self.circle_crop=make_circle_crop(self.new_height,self.new_width,self.centre_x,self.centre_y)

        # cv2.namedWindow('ImageWindow', cv2.WINDOW_NORMAL)




        # for simulation
        self.j=1
        folder_path = '/media/tdillon/4D71-BDA7/frame_grabber_images/string_phantom'

        # Use glob to find all .npy files in the folder
        npy_files = glob.glob(os.path.join(folder_path, '*.npy'))
        self.file_count = len(npy_files)

        print("finished init")
        print("self box crop is", self.box_crop)



    def image_callback(self, msg):
        
        image_width = rospy.get_param('/usb_cam/image_width', default=1280)
        image_height = rospy.get_param('/usb_cam/image_height', default=1024)

        # Assuming RGB format
        rgb_image_data = np.frombuffer(msg.data, dtype=np.uint8)

        # Reshape the RGB data
        rgb_image = rgb_image_data.reshape((image_height, image_width, 3))

        
        #  for simulation, import images instead.. (comment this line if necessary)
        # rgb_image = np.load('/media/tdillon/4D71-BDA7/frame_grabber_images/string_phantom/rgb_image_'+ str(self.j)+'.npy')

        # self.j=self.j+1
        # if(self.j==self.file_count):
        #     self.j=1

        # simulate a single static ultrasound scan
        # rgb_image = np.load('/media/tdillon/4D71-BDA7/frame_grabber_images/string_phantom/rgb_image_150.npy')

        grayscale_image=preprocess_ivus_image(rgb_image,self.box_crop,self.circle_crop,self.text_crop,self.crosshairs_crop,self.wire_crop)
        centre_x=self.centre_x
        centre_y=self.centre_y

        # #Crop the image - assuming 1280 x 1024 image
        # rgb_image=rgb_image[55:840,5:792,:]
    
        # #Extract luminance (Y) component
        # grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        # new_height, new_width = grayscale_image.shape
        # centre_x=new_height/2
        # centre_y=new_width/2
        

        # # threshold segmentation
        threshold = rospy.get_param('threshold')

        _, binary_image = cv2.threshold(grayscale_image, threshold, 255, cv2.THRESH_BINARY)

        # # maybe just update an opencv window instead?? similar to vnav code...
        # header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        # binary_image_msg = Image(
        #     header=header,
        #     height=self.new_height,
        #     width=self.new_width,
        #     encoding='mono8',
        #     is_bigendian=False,
        #     step=self.new_width,
        #     data=binary_image.tobytes()
        # )
        # self.binary_image_pub.publish(binary_image_msg)

        # Find coordinates of non-zero pixels (pixels above the threshold)

        # relevant_pixels = np.column_stack(np.where(binary_image > 0))

        # get the relevan234t pixels from segment_centroids instead
        relevant_pixels,local_image=segment_centroids(binary_image,centre_x,centre_y)

        print("size of relevant_pixels:", np.shape(relevant_pixels))

        # cv2.imshow('ImageWindow', local_image)

        # maybe just update an opencv window instead?? similar to vnav code...
        header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        rgb_image_msg = Image(
            header=header,
            height=self.new_height,
            width=self.new_width,
            encoding='rgb8',
            is_bigendian=False,
            step=self.new_width * 3,
            data=local_image.tobytes()
        )
        self.rgb_image_pub.publish(rgb_image_msg)

        #you would convert the relevant coordinates to position in x and y - points expressed in camera frame
        # relevant_pixels = relevant_pixels[1:]
        centred_pixels=relevant_pixels - [centre_x,centre_y]


        scaling = rospy.get_param('scaling')
        two_d_points=centred_pixels*scaling


        #maybe try putting in shape[1] instead and set the other to -1
        # note this is for IVUS where x axis points in direction of probe
        # three_d_points=np.hstack((two_d_points,np.zeros((two_d_points.shape[0], 1)))) #see function for this
        three_d_points=np.hstack((np.zeros((two_d_points.shape[0], 1)),two_d_points)) #see function for this
        
        timestamp_secs = msg.header.stamp.secs
        timestamp_nsecs = msg.header.stamp.nsecs

        # # Truncate nanoseconds to the first 4 decimal places
        # truncated_nsecs = timestamp_nsecs // 1000  # Keep only the first 4 decimal places

        # Create a rospy.Time object using the truncated timestamp information
        timestamp = rospy.Time(secs=timestamp_secs, nsecs=timestamp_nsecs)

        # Calculate transform_time by subtracting a duration
        transform_time = timestamp - rospy.Duration(0.02)


        # Assuming you have the fscan_points.pointsrame IDs for your transform
        ref_frame = 'ascension_origin'
        dest_frame = 'target1'


        try:
            # Lookup transform
            TW_EM = self.tf_buffer.lookup_transform(ref_frame, dest_frame, transform_time)
        except (rospy.ROSException, tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Failed to lookup transform")
            TW_EM = None

        ref_frame = 'ascension_origin'
        dest_frame = 'target2'

        try:
            # Lookup transform
            TW_EM_1 = self.tf_buffer.lookup_transform(ref_frame, dest_frame, transform_time)
        except (rospy.ROSException, tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Failed to lookup transform")
            TW_EM_1 = None
    
 
        TW_EM=transform_stamped_to_matrix(TW_EM)
        TW_EM_1=transform_stamped_to_matrix(TW_EM_1)

        # read in calibration parameters - could change image to image
        angle = rospy.get_param('angle')
        translation = rospy.get_param('translation')
        radial_offset = rospy.get_param('radial_offset')
        oclock = rospy.get_param('oclock')

        # Create a rotation matrix using the parameter value
        TEM_C = [[1,0,0,translation],[0,np.cos(angle),-np.sin(angle),radial_offset*np.cos(oclock)],[0,np.sin(angle),np.cos(angle),radial_offset*np.sin(oclock)],[0, 0, 0, 1]]

        # TEM_C = [[np.cos(angle),-np.sin(angle),0,translation],[np.sin(angle),np.cos(angle),0,0],[0,0,1,0],[0, 0, 0, 1]]

        #construct the point cloud object and transform it to WCS
        scan_points=o3d.geometry.PointCloud()
        scan_points.points=o3d.utility.Vector3dVector(three_d_points)
        scan_points.transform(TW_EM @ TEM_C)

        #modify the tracker frame (target1)
        # check what the arguments are for .transform - may be able to put in a quaternion directly...
        self.tracker_frame.transform(get_transform_inverse(self.previous_transform))
        self.tracker_frame.transform(TW_EM)
        self.previous_transform=TW_EM

        self.us_frame.transform(get_transform_inverse(self.previous_transform_us))
        self.us_frame.transform(TEM_C)
        self.us_frame.transform(TW_EM)
        self.previous_transform_us=TW_EM @ TEM_C

        self.em_1_frame.transform(get_transform_inverse(self.previous_transform_1))
        self.em_1_frame.transform(TW_EM_1)
        self.previous_transform_1=TW_EM_1

        #see jupyter notebook for faster functions

        self.spheres.vertices = o3d.utility.Vector3dVector([])
        self.spheres.triangles = o3d.utility.Vector3iVector([])
        
        self.sphere.vertices = o3d.utility.Vector3dVector([])
        self.sphere.triangles = o3d.utility.Vector3iVector([])
        
       
        # first_run=1
        # counter=0
        # print("size of scan_points.points:", np.shape(np.asarray(scan_points.points)))
        # for image_point in scan_points.points:
        #     # reduce resolution of balls again...
            
        #     self.sphere=o3d.geometry.TriangleMesh.create_sphere(radius=0.0025,resolution=5)
        #     # sphere=o3d.geometry.TriangleMesh.create_sphere(radius=5000000,resolution=5)
        #     self.sphere.translate(image_point) 

        #     if(first_run==0):
        #         num_target_vertices = len(self.spheres.vertices)
        #         triangles_offset_np = np.asarray(self.spheres.triangles) + num_target_vertices
        #         triangles_offset = o3d.utility.Vector3iVector(triangles_offset_np)
        #         self.spheres.vertices.extend(self.sphere.vertices)
        #         self.spheres.triangles.extend(triangles_offset)
            
        #     else:
        #         self.spheres.vertices.extend(self.sphere.vertices)
        #         self.spheres.triangles.extend(self.sphere.triangles)

        #     first_run=0


        vertices_list=[]
        triangles_list=[]
        first_run=1
        counter=0


        sphere_radius=0.0025
        sphere_resolution=5
        # no_sphere_vertices=np.shape(np.asarray(o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius,resolution=sphere_resolution).vertices))[0]

        # for image_point in scan_points.points:
            
        #     # reduce resolution of balls again...
            
        #     self.sphere=o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius,resolution=sphere_resolution)

        #     self.sphere.translate(image_point) 

        #     if(first_run==0):
            
        #         total_vertices=total_vertices+no_sphere_vertices
        #         triangles_offset_np = np.asarray(self.sphere.triangles) + total_vertices
 
        #         vertices_list.append(np.asarray(self.sphere.vertices))
        #         triangles_list.append(triangles_offset_np)

         
            
        #     else:
          
        #         total_vertices=no_sphere_vertices
        #         vertices_list.append(np.asarray(self.sphere.vertices))
        #         triangles_list.append(np.asarray(self.sphere.triangles))
        #         first_run=0
            
            
        # vertices_list=np.vstack(vertices_list)
        # triangles_list=np.vstack(triangles_list)

        # self.spheres.vertices.extend(o3d.utility.Vector3dVector(vertices_list))
        # self.spheres.triangles.extend(o3d.utility.Vector3iVector(triangles_list))

        centroid_sphere = get_sphere_cloud(np.asarray(scan_points.points), 0.0025, 5)

        self.spheres.vertices = centroid_sphere.vertices
        self.spheres.triangles = centroid_sphere.triangles



           
        self.spheres.paint_uniform_color([0,1,0])
        self.vis.update_geometry(self.spheres)
        self.vis.update_geometry(self.tracker_frame)
        self.vis.update_geometry(self.em_1_frame)
        self.vis.update_geometry(self.us_frame)

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

def get_sphere_cloud(points,radius,resolution, color=[0,1,0]):

    spheres = o3d.geometry.TriangleMesh()
    spheres.vertices = o3d.utility.Vector3dVector([])
    spheres.triangles = o3d.utility.Vector3iVector([])

    for image_point in points:

        
        
        # Create a new sphere for each point
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        translated=sphere.translate(image_point)

        num_target_vertices = len(spheres.vertices)
        triangles_offset_np = np.asarray(sphere.triangles) + num_target_vertices
        triangles_offset = o3d.utility.Vector3iVector(triangles_offset_np)

        # Extend the vertices and triangles of the target mesh with those of the source mesh
        spheres.vertices.extend(sphere.vertices)
        spheres.triangles.extend(triangles_offset )

        # print("sphere triangles", np.shape(np.asarray(spheres.triangles))[0])

        


    spheres.paint_uniform_color(color)
    spheres.compute_vertex_normals()

    return spheres 

if __name__ == '__main__':
    # try:

    rospy.init_node('open3d_visualizer')
    pc_updater = PointCloudUpdater()        
    # rospy.on_shutdown(pc_updater.save_image_and_transform_data)
    rospy.on_shutdown(save_parameters_on_shutdown)
    rospy.spin()

        # print("started main function")
        # rospy.sleep(2.0)
        # rospy.init_node('open3d_visualizer')
        # # fiducial_ = self.tf_buffer.lookup_transform('ascension_origin','target2', rospy.Duration(0))
        # print("initialization successful, back in main function")
        # pc_updater = PointCloudUpdater()
        
        # rospy.spin()
        
    # except rospy.ROSInterruptException:
    #     self.vis.destroy_window()
    #     pass
