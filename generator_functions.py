import copy

from tube_functions import *


def generate_vessel_3d(rng, vessel_type, control_point_path, shear, warp, spline_index=0):
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
        RCA_control_points = np.load(os.path.join(control_point_path, "RCA_ctrl_points.npy")) / 1000 # [m] instead of [mm]
        mean_ctrl_pts = np.mean(RCA_control_points, axis=0)
        stdev_ctrl_pts = np.std(RCA_control_points, axis=0)
        main_C, main_dC = RCA_vessel_curve(sample_size, mean_ctrl_pts, stdev_ctrl_pts, length, rng, shear, warp)

    tree, dtree, connections = branched_tree_generator(main_C, main_dC, num_branches, sample_size, side_branch_properties, curve_type=vessel_type)

    num_theta = 120
    spline_array_list = []
    surface_coords = []
    coords = np.empty((0,3))

    ##############################################################
    # Generate radii and surface coordinates for centerline tree #
    ##############################################################
    skip = False
    for ind in range(len(tree)):
        C = tree[ind]
        dC = dtree[ind]
        if ind == 0:
            rand_stenoses = np.random.randint(0, 3)
            key = "main_vessel"
            main_is_true = True
            max_radius = [random.uniform(0.004, main_branch_properties[branch_ID]['max_diameter']) / 2]

        else:
            rand_stenoses = np.random.randint(0, 2)
            max_radius = [random.uniform(side_branch_properties[ind]['min_radius'], side_branch_properties[ind]['max_radius'])]
            key = "branch{}".format(ind)
            main_is_true = False

        percent_stenosis = None
        stenosis_pos = None
        num_stenosis_points = None

        if num_stenoses is not None:
            rand_stenoses = num_stenoses

        try:
            X,Y,Z, new_radius_vec, percent_stenosis, stenosis_pos, num_stenosis_points = get_vessel_surface(C, dC, connections, supersampled_num_centerline_points, num_theta, max_radius,
                                                                                                        is_main_branch = main_is_true,
                                                                                                        num_stenoses=rand_stenoses,
                                                                                                        constant_radius=False,
                                                                                                        stenosis_severity=None,
                                                                                                        stenosis_position=None,
                                                                                                        stenosis_length=None,
                                                                                                        stenosis_type="gaussian",
                                                                                                        return_surface=True)
        except ValueError:
            print("Invalid sampling, skipping {}".format(spline_index))
            return None, None, None

        spline_array = np.concatenate((C, np.expand_dims(new_radius_vec, axis=-1)), axis=1)[::jj,:]
        spline_array_list.append(spline_array)

        branch_coords = np.stack((X.T,Y.T,Z.T)).T
        surface_coords.append(branch_coords)
        coords = np.concatenate((coords,np.stack((X.flatten(), Y.flatten(), Z.flatten())).T))

        vessel_info[key]['num_stenoses'] = int(rand_stenoses)
        vessel_info[key]['max_radius'] = float(new_radius_vec[0]*1000)
        vessel_info[key]['min_radius'] = float(new_radius_vec[-1]*1000)
        if connections[ind] is not None:
            vessel_info[key]['branch_point'] = int(connections[ind]/jj)
        if rand_stenoses > 0:
            vessel_info[key]['stenosis_severity'] = [float(i) for i in percent_stenosis]
            vessel_info[key]['stenosis_position'] = [int(i/jj) for i in stenosis_pos]
            vessel_info[key]['num_stenosis_points'] = [int(i/jj) for i in num_stenosis_points]

    return coords, vessel_info, spline_array_list

def make_projection(coords, theta, phi, sod, sid, spacing, img_dim=512):
    projected = project_multiple(coords, theta, phi, sod, sid, spacing, (img_dim, img_dim))
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