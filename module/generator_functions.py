import copy

from .tube_functions import *
from mpl_toolkits.mplot3d import Axes3D


def generate_vessel_3d(rng, vessel_type, control_point_path, shear, warp, spline_index=0, visualization=False):
    main_branch_properties = {
        1: {"name": "RCA", "min_length": 0.120, "max_length": 0.140, "max_diameter": 0.005}, #units in [m] not [mm]
        2: {"name": "LAD", "min_length": 0.100, "max_length": 0.130, "max_diameter": 0.005},
        3: {"name": "LCx", "min_length": 0.080, "max_length": 0.100, "max_diameter": 0.0045}
    }
    side_branch_properties = {
        1: {"name": "SA", "length": 0.035, "min_radius": 0.0009, "max_radius": 0.0011, "parametric_position": [0.03, 0.12]},
        2: {"name": "AM", "length": 0.0506, "min_radius": 0.001, "max_radius": 0.0012, "parametric_position": [0.18, 0.35]},
        3: {"name": "PDA", "length": 0.055, "min_radius": 0.001, "max_radius": 0.0012, "parametric_position": [0.55, 0.65]}
    }
    vessel_dict = {'num_stenoses': None, 'stenosis_severity': [], 'stenosis_position': [],
           'num_stenosis_points': [], 'max_radius': None, 'min_radius': None, 'branch_point': None}
    spline_index = 0
    num_centerline_points = 200
    jj = 1
    supersampled_num_centerline_points = jj * num_centerline_points
    num_branches = 3
    num_stenoses = None


    vessel_info = {'spline_index': int(spline_index), 'tree_type': [], 'num_centerline_points': num_centerline_points, 'theta_array': [], 'phi_array': [], 'main_vessel':copy.deepcopy(vessel_dict)}
    for branch_index in range(num_branches):
        vessel_info["branch{}".format(branch_index + 1)] = copy.deepcopy(vessel_dict)

    # default is RCA; LCx/LAD single vessels and LCA tree will be implemented in future
    branch_ID = 1
    vessel_info["tree_type"].append(main_branch_properties[branch_ID]["name"])

    length = random.uniform(main_branch_properties[branch_ID]['min_length'], main_branch_properties[branch_ID]['max_length']) # convert to [m] to stay consistent with projection setup
    sample_size = supersampled_num_centerline_points

    if vessel_type == 'cylinder':
        main_C, main_dC = cylinder(length, supersampled_num_centerline_points)
    elif vessel_type == 'spline':
        main_C, main_dC = random_spline(length, order, np.random.randint(order + 1, 10), sample_size)
    else:
        control_points = np.load(os.path.join(control_point_path, f"{vessel_type}_ctrl_points.npy")) / 1000 # [m] instead of [mm]
        is_left = vessel_type in ['LCX', 'LAD']
        mean_ctrl_pts = control_points.copy() if is_left else np.mean(control_points, axis=0)
        stdev_ctrl_pts = np.std(control_points, axis=0)
        main_C, main_dC = vessel_curve(sample_size, mean_ctrl_pts, stdev_ctrl_pts, length, rng, shear=shear, warp=warp, is_left=is_left)

    tree, dtree, connections = branched_tree_generator(control_point_path, main_C, main_dC, num_branches, sample_size, side_branch_properties, curve_type=vessel_type)
    for conn in connections:
        if conn is not None:
            print(main_C[conn])
    
    tree = [main_C]
    dtree = [main_dC]
    segment_ends = connections
    connections=[None]
    
    print(segment_ends)
    
    num_theta = 120
    spline_array_list = []
    surface_coords = []
    # coords = np.empty((0,3))
    coords = []

    ##############################################################
    # Generate radii and surface coordinates for centerline tree #
    ##############################################################
    skip = False
    C = tree[0]
    dC = dtree[0]
    rand_stenoses = np.random.randint(0, 3)
    key = "main_vessel"
    main_is_true = True
    max_radius = [random.uniform(0.004, main_branch_properties[branch_ID]['max_diameter']) / 2]



    percent_stenosis = None
    stenosis_pos = None
    num_stenosis_points = None

    if num_stenoses is not None:
        rand_stenoses = num_stenoses


    part1 = C[0:segment_ends[1]]
    Dpart1 = dC[0:segment_ends[1]]
    part2 = C[segment_ends[1]:segment_ends[2]]
    Dpart2 = C[segment_ends[1]:segment_ends[2]]
    part3 = C[segment_ends[2]:]
    Dpart3 = dC[segment_ends[2]:]


    X1,Y1,Z1, _, _, _, _ = get_vessel_surface(part1, Dpart1, connections, supersampled_num_centerline_points, num_theta, max_radius,
                                                                                                is_main_branch = main_is_true,
                                                                                                num_stenoses=rand_stenoses,
                                                                                                constant_radius=False,
                                                                                                stenosis_severity=None,
                                                                                                stenosis_position=None,
                                                                                                stenosis_length=None,
                                                                                                stenosis_type="gaussian",
                                                                                                return_surface=True)
    X2,Y2,Z2, _, _, _, _ = get_vessel_surface(part2, Dpart2, connections, supersampled_num_centerline_points, num_theta, max_radius,
                                                                                        is_main_branch = main_is_true,
                                                                                        num_stenoses=rand_stenoses,
                                                                                        constant_radius=False,
                                                                                        stenosis_severity=None,
                                                                                        stenosis_position=None,
                                                                                        stenosis_length=None,
                                                                                        stenosis_type="gaussian",
                                                                                        return_surface=True)
    X3,Y3,Z3, _, _, _, _ = get_vessel_surface(part3, Dpart3, connections, supersampled_num_centerline_points, num_theta, max_radius,
                                                                                is_main_branch = main_is_true,
                                                                                num_stenoses=rand_stenoses,
                                                                                constant_radius=False,
                                                                                stenosis_severity=None,
                                                                                stenosis_position=None,
                                                                                stenosis_length=None,
                                                                                stenosis_type="gaussian",
                                                                                return_surface=True)

    coords.append(np.stack((X1.flatten(), Y1.flatten(), Z1.flatten())).T)
    coords.append(np.stack((X2.flatten(), Y2.flatten(), Z2.flatten())).T)
    coords.append(np.stack((X3.flatten(), Y3.flatten(), Z3.flatten())).T)


    return coords, None, None

def make_projection(coords, theta, phi, sod, sid, spacing, img_dim=512, rescale=False):
    def standardize(arr):
        return (arr - np.mean(arr)) / np.std(arr)
    projected = project_multiple(coords, theta, phi, sod, sid, spacing, (img_dim, img_dim))
    if rescale:
        projected[:, 1] = (standardize(projected[:, 1]) * 1/4 + 1) * img_dim/2
        projected[:, 0] = (standardize(projected[:, 0]) * 1/4 +  1) * img_dim/2
    ind_lower_cutoff = np.all(projected > 0, axis=1)
    ind_upper_cutoff = np.all(projected < img_dim, axis=1)
    cutoff_array = np.stack((ind_lower_cutoff, ind_upper_cutoff), axis=1)
    valid_point_inds = np.all(cutoff_array, axis=1)
    
    projected = projected[valid_point_inds, :].astype("int")
    img = np.zeros((img_dim, img_dim))

    for x,y in projected:
        img[x, y] = 255
    # remove gaps in mask
    img = morph.binary_closing(img, morph.disk(2))
    img = filters.gaussian(img, sigma=0.5) > 0.25
    return img