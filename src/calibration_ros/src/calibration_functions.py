def model_registration(model, fiducial_1,fiducial_2, reference_1,reference_2):

    simplified  = simplified .subdivide_midpoint(number_of_iterations=3)
    point_cloud_1=o3d.geometry.PointCloud()
    point_cloud_1.points=simplified.vertices
    point_cloud_1.paint_uniform_color([0, 0, 1])
    string_phantom=point_cloud_1

    # define lines for registration
    model_line=create_line(reference_1,reference_2)
    across_model=reference_1-reference_2
    
    fiducial_1_position=fiducial_1[3,0:3]
    fiducial_2_position=fiducial_2[3,0:3]
    em_line=create_line(fiducial_2_position,fiducial_1_position) #this is correct
    em_line_points=np.asarray(em_line.points)
    across_em=reference_1-reference_2

    # just for verification
    verify_scaling(across_model,across_em)

    # make copies for registration - needed?
    shrunk=copy.deepcopy(string_phantom).scale(EM_scaling,center=[0,0,0])
    model_line=model_line.scale(EM_scaling,center=[0,0,0])
    z_line=create_line([0,0,0],[0,0,1])

    rotated_model=copy.deepcopy(shrunk)
    rotated_line=copy.deepcopy(model_line)

    # small optional prerotation
    prerotation=np.array([[1, 0, 0],[0, np.cos(-0.05),-np.sin(-0.05)],[0, np.sin(-0.05), np.cos(-0.05)]])
    prerotate_roll=np.eye(4)
    prerotate_roll[0:3,0:3]=prerotation
    rotated_model.transform(prerotate_roll)
    rotated_line.transform(prerotate_roll)
    z_line.transform(prerotate_roll)

    prerotation=np.array([[np.cos(-0.02),-np.sin(-0.02), 0],[np.sin(-0.02), np.cos(-0.02),0],[0, 0, 1]])
    prerotate_roll=np.eye(4)
    prerotate_roll[0:3,0:3]=prerotation
    rotated_model.transform(prerotate_roll)
    rotated_line.transform(prerotate_roll)
    z_line.transform(prerotate_roll)

    # manual rotation and scaling of phantom
    roll=-np.pi/2
    roll_matrix=np.array([[np.cos(roll), 0, np.sin(roll)],[0, 1, 0],[-np.sin(roll), 0, np.cos(roll)]])
    rotate_roll=np.eye(4)
    rotate_roll[0:3,0:3]=roll_matrix
    rotated_model.transform(rotate_roll)
    rotated_line.transform(rotate_roll)

    # grab normal of 2000th fiducial1 and 2
    normal_1=fiducial_1[:3,0]
    normal_2=fiducial_1[:3,0]
    average_normal=(-normal_1+normal_2)/2
    norm = np.linalg.norm(average_normal)
    average_normal=average_normal/norm
    normal_line=create_line([0,0,0],average_normal)

    #align y axis with average normal from embedded EM trackers
    alignment_rotation=align_vectors(average_normal,[0,1,0])
    alignment_transform=np.eye(4)
    alignment_transform[0:3,0:3]=alignment_rotation
    alignment_transform=get_transform_inverse(alignment_transform)
    rotated_model.transform(alignment_transform)
    rotated_line.transform(alignment_transform)
    z_line.transform(alignment_transform)

    rotated_points=np.asarray(rotated_line.points)
    tracker_2_displacement=em_line_points[0]-rotated_points[1]
    rotated_model.translate(tracker_2_displacement)
    rotated_line.translate(tracker_2_displacement)
    z_line.translate(tracker_2_displacement)

    # place this rendering elsewhere?
    em_1_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
    em_1_frame.transform(fiducial_1)

    em_2_frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
    em_2_frame.transform(fiducial_2)

    # angle checking
    origin_z=[0,0,1]
    z_line_points=np.asarray(z_line.points)
    base_z=z_line_points[1,:]-z_line_points[0,:]
    base_z=base_z/np.linalg.norm(base_z)
    angle_between,angle_radians=angle_between_vectors(base_z,origin_z)

    print("z-axis of model should be small:", angle_between)
    print("in radians", angle_radians)

    return rotated_model

# def one_fiducial_model_registration(model, fiducial_1,reference_1):

#     # simplify string phantom
#     model  = model .subdivide_midpoint(number_of_iterations=3)
#     point_cloud_1=o3d.geometry.PointCloud()
#     point_cloud_1.points=model.vertices
#     point_cloud_1.paint_uniform_color([0, 0, 1])
#     string_phantom=point_cloud_1

#     fiducial_1=transform_stamped_to_matrix(fiducial_1)
#     fiducial_1_position=fiducial_1[3,0:3]

#     # assume this is true until we get two fiducials working
#     EM_scaling=1.0 

#     # make copies for registration - needed?
#     shrunk=copy.deepcopy(string_phantom).scale(EM_scaling,center=[0,0,0])
#     z_line=create_line([0,0,0],[0,0,1])
#     rotated_model=copy.deepcopy(shrunk)

#     # small optional rotation adjustment
#     prerotation=np.array([[1, 0, 0],[0, np.cos(-0.05),-np.sin(-0.05)],[0, np.sin(-0.05), np.cos(-0.05)]])
#     prerotate_roll=np.eye(4)
#     prerotate_roll[0:3,0:3]=prerotation
#     rotated_model.transform(prerotate_roll)
#     z_line.transform(prerotate_roll)

#     prerotation=np.array([[np.cos(-0.02),-np.sin(-0.02), 0],[np.sin(-0.02), np.cos(-0.02),0],[0, 0, 1]])
#     prerotate_roll=np.eye(4)
#     prerotate_roll[0:3,0:3]=prerotation
#     rotated_model.transform(prerotate_roll)
#     z_line.transform(prerotate_roll)

#     # Large rotational adjustment
#     roll=-np.pi/2
#     roll_matrix=np.array([[np.cos(roll), 0, np.sin(roll)],[0, 1, 0],[-np.sin(roll), 0, np.cos(roll)]])
#     rotate_roll=np.eye(4)
#     rotate_roll[0:3,0:3]=roll_matrix
#     rotated_model.transform(rotate_roll)

#     # grab normal of 2000th fiducial1 and 2
#     normal_1=fiducial_1[:3,0]

#     #align y axis with average normal from embedded EM trackers
#     alignment_rotation=align_vectors(normal_1,[0,1,0])
#     alignment_transform=np.eye(4)
#     alignment_transform[0:3,0:3]=alignment_rotation
#     alignment_transform=get_transform_inverse(alignment_transform)
#     rotated_model.transform(alignment_transform)
#     z_line.transform(alignment_transform)

#     # tracker_2_displacement=em_line_points[0]-rotated_points[1]
#     # rotated_model.translate(tracker_2_displacement)
#     # z_line.translate(tracker_2_displacement)

#     # angle verification
#     origin_z=[0,0,1]
#     z_line_points=np.asarray(z_line.points)
#     base_z=z_line_points[1,:]-z_line_points[0,:]
#     base_z=base_z/np.linalg.norm(base_z)
#     angle_between,angle_radians=angle_between_vectors(base_z,origin_z)

#     print("z-axis of model should be small:", angle_between)
#     print("in radians", angle_radians)


#     return rotated_model