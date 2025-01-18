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
from std_msgs.msg import Header  
import math
import yaml

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

# def load_default_values(file_path='/home/tdillon/mapping/calibration_parameters.yaml'):
#     with open(file_path, 'r') as stream:
        
#         default_values={}

#         try:
#             all_parameters = yaml.safe_load(stream)
#             # Convert the loaded angle to radians if needed
#             if '/angle' in all_parameters:
#                 default_values['/angle'] = all_parameters['/angle']

#             if '/threshold' in all_parameters:
#                 default_values['/threshold'] = all_parameters['/threshold']

#             if '/translation' in all_parameters:
#                 default_values['/translation'] = all_parameters['/translation']

#             if '/scaling' in all_parameters:
#                 default_values['/scaling'] = all_parameters['/scaling']

#             if '/radial_offset' in all_parameters:
#                 default_values['/radial_offset'] = all_parameters['/radial_offset']

#             if '/oclock' in all_parameters:
#                 default_values['/oclock'] = all_parameters['/oclock']
            
#             return default_values
#         except yaml.YAMLError as exc:
#             print(exc)

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
        default_values['/threshold'] = 100

        default_values['/crop_index'] = 49

        # Set default parameter values if they don't exist
        for param_name in param_names:
            if not rospy.has_param(param_name):
                rospy.set_param(param_name, default_values.get(param_name, None))
                rospy.loginfo(f"Initialized {param_name} with default value: {default_values.get(param_name, None)}")

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



        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        # self.transform_sub = rospy.Subscriber('/ascension_node/target_poses', TransformStamped, self.transform_callback)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        print("frame grabber subscriber:", self.image_sub)
        print("tf listener:", self.tf_listener)
        print("successfully initialized visualizer!")

        self.binary_image_pub = rospy.Publisher('/binary_image', Image, queue_size=1)

        self.image_width = rospy.get_param('/usb_cam/image_width', default=1280)
        self.image_height = rospy.get_param('/usb_cam/image_height', default=1024)

        self.crop_radius=10
        no_points=rospy.get_param('no_points')
        self.previous_no_points=no_points
        self.gridlines=get_gridlines(self.image_height/2,self.image_width/2,no_points, self.crop_radius)
      

        # assume these are ideal bounds
        min_bounds=np.array([0.05,-0.1,-0.1]) 
        max_bounds=np.array([0.3,0.1,0.1]) 
        self.box=get_box(min_bounds,max_bounds)

        self.vis.add_geometry(self.point_cloud)
        self.vis.add_geometry(self.us_frame)
        self.vis.add_geometry(self.tracker_frame)
        self.vis.add_geometry(self.em_1_frame)
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
        
        crosshair_width=5
        crosshair_height=2
        crosshair_vert_coordinates=[[380,62],[380,128],[380,193],[380,259],[380,324],[394,455],[394,520],[394,585],[394,651],[394,717]]
        crosshair_horiz_coordinates=[[61,395],[127,395],[192,395],[258,395],[323,395],[454,381],[519,381],[585,381],[650,381],[716,381]]
        self.crosshairs_crop=make_crosshairs_crop(self.new_height,self.new_width,crosshair_width,crosshair_height,crosshair_vert_coordinates,crosshair_horiz_coordinates)
        self.circle_crop=make_circle_crop(self.new_height,self.new_width,self.centre_x,self.centre_y)

        # what number image callback are we on?
        self.image_call=1
        self.write_folder = rospy.get_param('~image_path')
        # make this a rospy parameter later to turn recording on and off and also clean the field of view
        self.record=1
        

    def image_callback(self, msg):
        

        # Assuming RGB format
        rgb_image_data = np.frombuffer(msg.data, dtype=np.uint8)

        # Reshape the RGB data
        rgb_image = rgb_image_data.reshape((self.image_height, self.image_width, 3))


        # rgb_image = np.load('/media/tdillon/4D71-BDA7/frame_grabber_images/evar_graft/rgb_image_150.npy')

        
        grayscale_image=preprocess_ivus_image(rgb_image,self.box_crop,self.circle_crop,self.text_crop,self.crosshairs_crop)
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
        

        #you would convert the relevant coordinates to position in x and y - points expressed in camera frame
        centred_pixels=relevant_pixels - [centre_x,centre_y]


        scaling = rospy.get_param('scaling')
        two_d_points=centred_pixels*scaling


        #maybe try putting in shape[1] instead and set the other to -1
        three_d_points=np.hstack((np.zeros((two_d_points.shape[0], 1)),two_d_points)) #see function for this
        
        timestamp_secs = msg.header.stamp.secs
        timestamp_nsecs = msg.header.stamp.nsecs

        # # Truncate nanoseconds to the first 4 decimal places
        # truncated_nsecs = timestamp_nsecs // 1000  # Keep only the first 4 decimal places

        # Create a rospy.Time object using the truncated timestamp information
        timestamp = rospy.Time(secs=timestamp_secs, nsecs=timestamp_nsecs)

        # Calculate transform_time by subtracting a duration
        transform_time = timestamp - rospy.Duration(0.02)


        # Assuming you have the frame IDs for your transform
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

        # this code is specific to mapping_record!
        if(self.record==1):
            # code for saving images
            full_str=self.write_folder + '/rgb_images/rgb_image_'+ str(self.image_call) + '.npy'
            
            # save rgb_image as is
            with open(full_str, 'wb') as f:
                np.save(f,rgb_image)

            print("Saved", self.i, "images")
            

            full_str=self.write_folder + '/transform_data/TW_EM_'+ str(self.image_call) + '.npy'

            # code for saving transform data
            with open(full_str, 'wb') as f:
                np.save(f,TW_EM)

            self.image_call=self.i_image_call+1


        # read in calibration parameters - could change image to image
        angle = rospy.get_param('angle')
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

        self.em_1_frame.transform(get_transform_inverse(self.previous_transform_1))
        self.em_1_frame.transform(TW_EM_1)
        self.previous_transform_1=TW_EM_1

        self.us_frame.transform(get_transform_inverse(self.previous_transform_us))
        self.us_frame.transform(TEM_C)
        self.us_frame.transform(TW_EM)
        self.previous_transform_us=TW_EM @ TEM_C
       
        self.point_cloud.points.extend(scan_points.points)

        self.vis.update_geometry(self.point_cloud)
        self.vis.update_geometry(self.us_frame)
        self.vis.update_geometry(self.tracker_frame)
        self.vis.update_geometry(self.em_1_frame)

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

            

if __name__ == '__main__':
    try:
        rospy.init_node('open3d_visualizer')
        pc_updater = PointCloudUpdater()
        rospy.spin()
        
    except rospy.ROSInterruptException:
        self.vis.destroy_window()
        pass
