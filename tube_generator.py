__package__ = "generator"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from module import *
import random
import json
import copy
import argparse

# general: required arguments
parser = argparse.ArgumentParser('3D vessel tree generator')
parser.add_argument('--save_path', default=None, type=str, required=True)
parser.add_argument('--dataset_name', default="test", type=str)
parser.add_argument('--num_trees', default=10, type=int)
parser.add_argument('--save_visualization', action='store_true', help="this flag will plot the generated 3D surfaces and save it as a PNG")

# centerlines: optional
parser.add_argument('--num_branches', default=0, type=int,
                    help="Number of side branches. Set to 0 for no side branches")
parser.add_argument('--vessel_type', default="RCA", type=str, help="Options are: 'cylinder, 'spline', and 'RCA'")
parser.add_argument('--control_point_path', default="./RCA_branch_control_points/moderate", type=str)
parser.add_argument('--num_centerline_points', default=200, type=int)
parser.add_argument('--centerline_supersampling', default=1, type=int, help="factor by which to super-sample centerline points when generating vessel surface")
parser.add_argument('--shear', action='store_true', help="add random shear augmentation")
parser.add_argument('--warp', action='store_true', help="add random warping augmentation")

#radii/stenoses: optional
parser.add_argument('--constant_radius', action='store_true')
parser.add_argument('--num_stenoses', default=None, type=int)
parser.add_argument('--stenosis_position', nargs="*", default=None, type=int)
parser.add_argument('--stenosis_severity', nargs="*", default=None, type=float)
parser.add_argument('--stenosis_length', nargs="*", default=None, type=int, help="number of points in radius vector where stenosis will be introduced")


#projections: optional
parser.add_argument('--generate_projections', action="store_true")
parser.add_argument('--num_projections', default=4, type=int,
                    help="number of random projection images to generate")
# TODO: specify angles/windows for random projections
args = parser.parse_args()



save_path = args.save_path
dataset_name=args.dataset_name
num_trees = args.num_trees

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print("created {}".format(save_path))
if not os.path.exists(os.path.join(save_path,dataset_name)):
    os.makedirs(os.path.join(save_path,dataset_name))
    print("created {}".format(os.path.join(save_path,dataset_name)))

jj = args.centerline_supersampling
num_projections = args.num_projections
num_centerline_points = args.num_centerline_points # number of interpolated centerline points to save
supersampled_num_centerline_points = jj * num_centerline_points #use larger number of centerline points to create solid surface for projections, if necessary
num_branches = args.num_branches  # set to 0 if not adding side branches
order = 3

main_branch_properties = {
    1: {"name": "RCA", "min_length": 0.120, "max_length": 0.140, "max_diameter": 0.005}, #units in [m] not [mm]
    2: {"name": "LAD", "min_length": 0.100, "max_length": 0.130, "max_diameter": 0.005},
    3: {"name": "LCx", "min_length": 0.080, "max_length": 0.100, "max_diameter": 0.0045},
}

side_branch_properties = {
    1: {"name": "SA", "length": 0.035, "min_radius": 0.0009, "max_radius": 0.0011, "parametric_position": [0.03, 0.12]},
    2: {"name": "AM", "length": 0.0506, "min_radius": 0.001, "max_radius": 0.0012, "parametric_position": [0.18, 0.35]},
    3: {"name": "PDA", "length": 0.055, "min_radius": 0.001, "max_radius": 0.0012, "parametric_position": [0.55, 0.65]}
}

vessel_dict = {'num_stenoses': None, 'stenosis_severity': [], 'stenosis_position': [],
           'num_stenosis_points': [], 'max_radius': None, 'min_radius': None, 'branch_point': None}


if __name__ == "__main__":
    seed = 5
    random.seed(seed)
    rng = np.random.default_rng()

    for i in range(num_trees):
        spline_index = i

        cpp = "./LCA_branch_control_points/moderate" if args.vessel_type in ['LCX', 'LAD'] else "./RCA_branch_control_points/moderate"
        coords, vessel_info, spline_array_list = generate_vessel_3d(rng, args.vessel_type, cpp, args.shear, args.warp, spline_index)
        if coords is None:
            continue
        ###################################
        ######       projections     ######
        ###################################
        img_dim = 512
        ImagerPixelSpacing = 0.35 / 1000
        SID = 1.2
        SOD = 0.75

        vessel_info["ImagerPixelSpacing"] = ImagerPixelSpacing
        vessel_info["SID"] = SID
        vessel_info["SOD"] = SOD

        theta_array, phi_array = pick_angles(num_projections)
        
        for i in range(num_projections):
            img = make_projection(coords, theta_array[i], phi_array[i], SOD, SID, (ImagerPixelSpacing, ImagerPixelSpacing), rescale=vessel_type in ['LCX', 'LAD'])
            suffixes = ['a', 'b', 'c', 'd']

            if not os.path.exists(os.path.join(save_path, dataset_name, "images", dataset_name)):
                os.makedirs(os.path.join(save_path, dataset_name, "images", dataset_name))

            path = os.path.join(save_path, dataset_name, "images", dataset_name, "image{:04d}{}.png".format(spline_index,suffixes[i]))
            plt.imsave(path, img, cmap="gray")

        vessel_info['theta_array'] = [float(i) for i in theta_array.tolist()]
        vessel_info['phi_array'] = [float(j) for j in phi_array.tolist()]

        #saves geometry as npy file (X,Y,Z,R) matrix
        if not os.path.exists(os.path.join(save_path, dataset_name, "labels", dataset_name)):
            os.makedirs(os.path.join(save_path, dataset_name, "labels", dataset_name))
        if not os.path.exists(os.path.join(save_path, dataset_name, "info")):
            os.makedirs(os.path.join(save_path, dataset_name, "info"))

        #saves geometry as npy file (X,Y,Z,R) matrix
        tree_array = np.array(spline_array_list)
        np.save(os.path.join(save_path, dataset_name, "labels", dataset_name, "{:04d}".format(spline_index)), tree_array)

        # writes a text file for each tube with relevant parameters used to generate the geometry
        with open(os.path.join(save_path, dataset_name, "info", "{:04d}.info.0".format(spline_index)), 'w+') as outfile:
            json.dump(vessel_info, outfile, indent=2)