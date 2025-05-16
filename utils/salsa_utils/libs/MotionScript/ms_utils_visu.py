import copy
import math
import os

from tqdm import tqdm

# os.environ['PYOPENGL_PLATFORM'] = 'egl' # Comment out this line for Windows OS

# os.environ["PYOPENGL_PLATFORM"] = "windows"
# os.environ["PYOPENGL_PLATFORM"] = "glfw"
import torch
import numpy as np
import trimesh
from body_visualizer.mesh.mesh_viewer import MeshViewer
from PIL import Image


import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import io
import imageio
import MotionScript.ms_utils as utils

print("Simply reruning the app (hamburger menu > 'Rerun') may be enough to get rid of a potential 'GLError'.")

# colors (must be in format RGB)
COLORS = {}
COLORS["grey"] = [0.7, 0.7, 0.7]
COLORS["red"] = [1.0, 0.4, 0.4]
COLORS["purple"] = [0.4, 0.4, 1.0]
COLORS["blue"] = [0.4, 0.8, 1.0]


def c2c(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    return tensor.detach().cpu().numpy()


imw, imh = 1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True) # todo check this offscreen


def image_from_body_vertices(body_vertices, faces, viewpoints=[[]], color='grey'):
    body_mesh = trimesh.Trimesh(vertices=body_vertices, faces=faces, vertex_colors=np.tile(COLORS[color]+[1.] if isinstance(color,str) else color+[1.], (6890, 1)))
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.radians(90), (1, 0, 0))) # base transformation
    imgs = []
    # render the body under the different required viewpoints
    for vp in viewpoints:
        # potentially transform the mesh to look at it from another viewpoint
        if vp: # non-initial viewpoint
            b = body_mesh.copy()
            b.apply_transform(trimesh.transformations.rotation_matrix(np.radians(vp[0]), vp[1]))
        else: # initial viewpoint
            b = body_mesh
        # produce the image
        mv.set_static_meshes([b])
        body_image = mv.render(render_wireframe=False)
        imgs.append(np.array(Image.fromarray(body_image)))
    return imgs


def image_from_pose_data(pose_data, body_model, viewpoints=[[]], color='grey'):
    """
    pose_data: torch tensor of size (n_poses, n_joints*3)
    viewpoints: list of viewpoints under which to render the different body
        poses, with each viewpoint defined as a tuple where the first element is
        the rotation angle (in degrees) and the second element is a tuple of 3
        slots indicating the rotation axis (eg. (0,1,0)). The stardard viewpoint
        is indicated with `[]`.

    Returns a list of images of size n_pose * len(viewpoints), grouped by pose
    (images for each viewpoints of the same pose are consecutive).
    """
    # infer the body pose from the joints
    with torch.no_grad():
        body_out = body_model(pose_body=pose_data[:,3:66], pose_hand=pose_data[:,66:], root_orient=pose_data[:,:3])
    # render body poses as images
    all_images = []
    for i in range(len(pose_data)):
        imgs = image_from_body_vertices(c2c(body_out.v[i]), c2c(body_model.f), viewpoints=viewpoints, color=color)
        all_images += imgs
    return all_images

def anim_from_pose_data(pose_data, body_model, viewpoints=[[]], color='grey'):
    """
    pose_data: torch tensor of size (n_poses, n_joints*3)
    viewpoints: list of viewpoints under which to render the different body
        poses, with each viewpoint defined as a tuple where the first element is
        the rotation angle (in degrees) and the second element is a tuple of 3
        slots indicating the rotation axis (eg. (0,1,0)). The stardard viewpoint
        is indicated with `[]`.

    Returns a list of images of size n_pose * len(viewpoints), grouped by pose
    (images for each viewpoints of the same pose are consecutive).
    """
    # infer the body pose from the joints
    with torch.no_grad():
        if pose_data.shape[1]>66: # For SFU Salsa
            body_out = body_model(pose_body=pose_data[:,3:66],
                                  pose_hand=pose_data[:,75:],
                                  root_orient=pose_data[:,:3])
        elif pose_data.shape[1]<=66: # For HumanML3D dataset
            body_out = body_model(pose_body=pose_data[:,3:66],
                                  root_orient=pose_data[:,:3])
            body_out = body_model(pose_body=pose_data[:, 3:66],
                                  pose_hand=body_model.pose_hand.repeat(pose_data.shape[0], 1),
                                  root_orient=pose_data[:, :3])
    # render body poses as images
    all_images = []
    print("Rendering 3D mesh..!")
    for i in tqdm(range(len(pose_data))):
        imgs = image_from_body_vertices(c2c(body_out.v[i]), c2c(body_model.f), viewpoints=viewpoints, color=color)
        all_images .append( imgs )

    output_list = [[] for _ in range(len(viewpoints))]

    for sublist in all_images:
        for i in range(len(viewpoints)):
            output_list[i].append(sublist[i])

    return output_list
def img2gif(image_list, output_gif_path):


    # Given list of ndarray images (assuming you have 'image_list' with shape [1600, 1600, 3] each)
    # image_list = [...]  # Your list of images

    # Convert the list of ndarrays to a list of PIL Image objects
    pil_images = [Image.fromarray(np.uint8(image)) for image in image_list]

    # Save the images as a GIF animation
    # output_gif_path = 'output_animation.gif'

    # imageio.mimsave(output_gif_path, pil_images, duration=0.05)
    imageio.mimsave(output_gif_path, pil_images, duration=0.1, loop=0)


def plot_3d_motion_Payam(args, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    joints, out_name, title = args

    data = joints.copy().reshape(len(joints), -1, 3)



    nb_joints = data.shape[1]
    smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7],
                          [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                                                                  [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                                                                  [9, 13, 16, 18, 20]]


    limits = 1000 if nb_joints == 21 else 1
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    # Get the 'tab10' colormap
    cmap = plt.get_cmap('tab20')
    # Generate a list of 10 distinct colors
    num_colors = 20
    discriminative_colors = [cmap(i) for i in range(num_colors)]

    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    target_joint2shw = [20, 21]
    trajec_joint = data[:, :, [0, 1, 2]]


    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):

        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)

        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)

        # fig = plt.figure(figsize=(480 / 96., 320 / 96.), dpi=96) if nb_joints == 21\
        #     else plt.figure(figsize=(10, 10))
        # fig=plt.figure(figsize=(480 / 96., 320 / 96.), dpi=96)
        # fig = plt.figure(figsize=(640 / 96., 480 / 96.), dpi=96)
        fig = plt.figure(figsize=(10, 10), dpi=96)

        if title is not None:
            wraped_title = ''.join((title))
            fig.suptitle(wraped_title, fontsize=16)
        # ax = p3.Axes3D(fig)
        # fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        init()

        # ax.lines = []
        for line in ax.lines:
            ax.lines.remove(line)
        # ax.collections = []
        for collec in ax.collections:
            ax.collections.remove(collec)


        ax.view_init(elev=110, azim=-90)
        ax.quiver(0, 0, 0, 0.6, 0, 0, color='r')  # X axis
        ax.quiver(0, 0, 0, 0, 0.6, 0, color='g')  # Y axis
        ax.quiver(0, 0, 0, 0, 0, 0.5, color='b')  # Z axis
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0],
                     MAXS[0] - trajec[index, 0],
                     0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        ax.scatter(data[index, :52, 0], data[index, :52, 1], data[index, :52, 2], color='black', s=5)
        # Add numbers next to each point
        for i in range(22):
            # ax.annotate(str(i), (data[index, i, 0], data[index, i, 1], data[index, i, 2]), textcoords="offset points", xytext=(5, 5), ha='center')
            ax.text(data[index, i, 0], data[index, i, 1], data[index, i, 2], str(i), color='green', fontsize=10, ha='center', va='center')
        # Add frame number:
        ax.text2D(0.1, 0.95, s=f'Frame: {index}', fontsize=12, ha='center', va='center',
                    color='red',
                    bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'), transform=ax.transAxes)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0],
                      np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')
            # ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

            chunk_size = 50
            for i_chunk in range(index//chunk_size + 1):
                s_index_chunk = i_chunk * chunk_size
                e_index_chunk = np.min( ( (i_chunk+1) * chunk_size, index ) )
                # ax.plot3D(trajec_joint[:index, 0]  - trajec[index, 0] ,
                #           trajec_joint[:index, 1], #  - data[index, 0, 1],
                #           trajec_joint[:index, 2] - trajec[index, 1]  ,
                #           linewidth=1.0,
                #           color=colors[i_chunk])
                for t2show_index in (target_joint2shw):
                    ax.plot3D(trajec_joint[s_index_chunk:e_index_chunk, t2show_index, 0] - trajec[index, 0],
                              trajec_joint[s_index_chunk:e_index_chunk, t2show_index, 1],  # - data[index, 0, 1],
                              trajec_joint[s_index_chunk:e_index_chunk, t2show_index, 2] - trajec[index, 1],
                              linewidth=1.0,
                              color=discriminative_colors[t2show_index % len(discriminative_colors)])



        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if out_name is not None:
            plt.savefig(out_name, dpi=96)
            plt.close()

        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            # print(fig.bbox.bounds)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in tqdm(range(frame_number)):
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)

def plot_3d_motion_Scanline(args, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    joints, out_name, title = args

    data = joints.copy().reshape(len(joints), -1, 3)



    nb_joints = data.shape[1]
    smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7],
                          [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                                                                  [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                                                                  [9, 13, 16, 18, 20]]


    limits = 1000 if nb_joints == 21 else 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    # Get the 'tab10' colormap
    cmap = plt.get_cmap('tab20')
    # Generate a list of 10 distinct colors
    num_colors = 20
    discriminative_colors = [cmap(i) for i in range(num_colors)]

    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    target_joint2shw = [20, 21]
    trajec_joint = data[:, :, [0, 1, 2]]


    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):

        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)

        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)

        fig = plt.figure(figsize=(480 / 96., 320 / 96.), dpi=96) if nb_joints == 21 else plt.figure(figsize=(10, 10),
                                                                                                    dpi=96)
        if title is not None:
            wraped_title = ''.join((title))
            fig.suptitle(wraped_title, fontsize=16)
        ax = p3.Axes3D(fig)

        init()

        # ax.lines = []
        # for line in ax.lines:
        #     ax.lines.remove(line)
        # # ax.collections = []
        # for collec in ax.collections:
        #     ax.collections.remove(collec)


        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0],
                     MAXS[0] - trajec[index, 0],
                     0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        ax.scatter(data[index, :52, 0], data[index, :52, 1], data[index, :52, 2], color='black', s=5)
        # Add numbers next to each point
        for i in range(22):
            # ax.annotate(str(i), (data[index, i, 0], data[index, i, 1], data[index, i, 2]), textcoords="offset points", xytext=(5, 5), ha='center')
            ax.text(data[index, i, 0], data[index, i, 1], data[index, i, 2], str(i), color='green', fontsize=10, ha='center', va='center')
        # Add frame number:
        ax.text2D(0.1, 0.95, s=f'Frame: {index}', fontsize=12, ha='center', va='center',
                    color='red',
                    bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'), transform=ax.transAxes)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0],
                      np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')
            # ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

            chunk_size = 50
            for i_chunk in range(index//chunk_size + 1):
                s_index_chunk = i_chunk * chunk_size
                e_index_chunk = np.min( ( (i_chunk+1) * chunk_size, index ) )
                # ax.plot3D(trajec_joint[:index, 0]  - trajec[index, 0] ,
                #           trajec_joint[:index, 1], #  - data[index, 0, 1],
                #           trajec_joint[:index, 2] - trajec[index, 1]  ,
                #           linewidth=1.0,
                #           color=colors[i_chunk])
                for t2show_index in (target_joint2shw):
                    ax.plot3D(trajec_joint[s_index_chunk:e_index_chunk, t2show_index, 0] - trajec[index, 0],
                              trajec_joint[s_index_chunk:e_index_chunk, t2show_index, 1],  # - data[index, 0, 1],
                              trajec_joint[s_index_chunk:e_index_chunk, t2show_index, 2] - trajec[index, 1],
                              linewidth=1.0,
                              color=discriminative_colors[t2show_index % len(discriminative_colors)])



        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if out_name is not None:
            plt.savefig(out_name, dpi=96)
            plt.close()

        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            # print(fig.bbox.bounds)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in tqdm(range(frame_number)):
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)



def draw_to_batch_Payam( smpl_joints_batch, title_batch=None, outname=None):
    batch_size = len(smpl_joints_batch)
    out = []



    for i in range(batch_size):
        out.append(plot_3d_motion_Payam([smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None]))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), loop=0 ) # , fps=20)
    out = torch.stack(out, axis=0)
    return out

def draw_to_batch_Scanline( smpl_joints_batch, title_batch=None, outname=None):
    batch_size = len(smpl_joints_batch)
    out = []



    for i in range(batch_size):
        out.append(plot_3d_motion_Scanline([smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None]))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]) ) # , fps=20)
    out = torch.stack(out, axis=0)
    return out


def Visualize_anim(body_model, dataID_2_pose_info,
                   dataID, out_folder,
                   start_frame=None, end_frame=None,
                   Skeleton3D_create=True, Mesh3D_create=False):
    # 1. Get the pose sequence info using dataID
    pose_info = dataID_2_pose_info[dataID]
    pose_seq_data, trans = utils.get_pose_sequence_data_from_file(pose_info)
    if start_frame is None:
        start_frame=0
    if end_frame is None:
        end_frame = len(pose_seq_data)
    pose_seq_data = pose_seq_data[start_frame:end_frame] # Pose
    trans = trans[start_frame:end_frame] # Translation


    # 2. 3D Mesh Visualization Animation

    if Mesh3D_create:
        print("Plotting 3D Mesh...")
        viewpoints = [[]]
        imgs = anim_from_pose_data(pose_seq_data, body_model, viewpoints=viewpoints, color="blue")
        for i, vp in enumerate(viewpoints):
            img2gif(imgs[i], f'{out_folder}/{dataID}_3DM_View({i}).gif')

    # 3D Skeleton Visualization
    if Skeleton3D_create:
        print("Plotting 3D skeleton...")
        j_seq = body_model(pose_body=pose_seq_data[:, 3:66],
                           pose_hand=pose_seq_data[:, 66:],
                           root_orient=pose_seq_data[:, :3]).Jtr
        j_seq += trans[:, np.newaxis, :]
        j_seq = j_seq.float()
        # Transformation function:
        rotX = lambda theta: torch.tensor(
            [[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]])

        def transf(rotMat, theta_deg, values):
            theta_rad = math.pi * torch.tensor(theta_deg).float() / 180.0
            return rotMat(theta_rad).mm(values.t()).t()

        j_seq = j_seq.detach().cpu()
        for frame_i in range(j_seq.shape[0]):
            j_seq[frame_i] = transf(rotX, -90, j_seq[frame_i])

        motions = j_seq
        motions = motions.cpu().detach().numpy()

        motion_batchified = np.expand_dims(motions, axis=0)
        title = ["Test"]
        # motion_batchified = torch.unsqueeze(motions, 0)
        draw_to_batch_Payam(motion_batchified, title, [f'{out_folder}/{dataID}_3DS.gif'])

import trimesh
from pyrender.constants import RenderFlags


from MotionScript.visualize_T2MGPT.simplify_loc2rot import joints2smpl
from MotionScript.visualize_T2MGPT.rotation2xyz import Rotation2xyz
from shapely import geometry
from trimesh import Trimesh

import pyrender
def renderT2M_GPT(motions, outdir='test_vis', device_id=0, name=None, pred=True):
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)
    MAXS = motions.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    trajec = motions[:, 0, [0, 2]]

    j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True)
    rot2xyz = Rotation2xyz(device=torch.device("cuda:0"))
    faces = rot2xyz.smpl_model.faces

    if (not os.path.exists(outdir + name + '_pred.pt') and pred) or (
            not os.path.exists(outdir + name + '_gt.pt') and not pred):
        print(f'Running SMPLify, it may take a few minutes.')
        motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]

        vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                           pose_rep='rot6d', translation=True, glob=True,
                           jointstype='vertices',
                           vertstrans=True)

        if pred:
            torch.save(vertices, outdir + name + '_pred.pt')
        else:
            torch.save(vertices, outdir + name + '_gt.pt')
    else:
        if pred:
            vertices = torch.load(outdir + name + '_pred.pt')
        else:
            vertices = torch.load(outdir + name + '_gt.pt')
    frames = vertices.shape[3]  # shape: 1, nb_frames, 3, nb_joints
    print(vertices.shape)
    MINS = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0]
    MAXS = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0]
    # vertices[:,:,1,:] -= MINS[1] + 1e-5

    out_list = []

    minx = MINS[0] - 0.5
    maxx = MAXS[0] + 0.5
    minz = MINS[2] - 0.5
    maxz = MAXS[2] + 0.5
    polygon = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]])
    polygon_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)

    vid = []
    for i in range(frames):
        if i % 10 == 0:
            print(i)

        mesh = Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces)

        base_color = (0.11, 0.53, 0.8, 0.5)
        ## OPAQUE rendering without alpha
        ## BLEND rendering consider alpha
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=base_color
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        polygon_mesh.visual.face_colors = [0, 0, 0, 0.21]
        polygon_render = pyrender.Mesh.from_trimesh(polygon_mesh, smooth=False)

        bg_color = [1, 1, 1, 0.8]
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))

        sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]

        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=300)

        scene.add(mesh)

        c = np.pi / 2

        scene.add(polygon_render, pose=np.array([[1, 0, 0, 0],

                                                 [0, np.cos(c), -np.sin(c), MINS[1].cpu().numpy()],

                                                 [0, np.sin(c), np.cos(c), 0],

                                                 [0, 0, 0, 1]]))

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [1, 1, 2]
        scene.add(light, pose=light_pose.copy())

        c = -np.pi / 6

        scene.add(camera, pose=[[1, 0, 0, (minx + maxx).cpu().numpy() / 2],

                                [0, np.cos(c), -np.sin(c), 1.5],

                                [0, np.sin(c), np.cos(c), max(4, minz.cpu().numpy() + (1.5 - MINS[1].cpu().numpy()) * 2,
                                                              (maxx - minx).cpu().numpy())],

                                [0, 0, 0, 1]
                                ])

        # render scene
        r = pyrender.OffscreenRenderer(960, 960)

        color, _ = r.render(scene, flags=RenderFlags.RGBA)
        # Image.fromarray(color).save(outdir+name+'_'+str(i)+'.png')

        vid.append(color)

        r.delete()

    out = np.stack(vid, axis=0)
    if pred:
        imageio.mimsave(outdir + name + '_pred.gif', out, duration=1000* 1/20)
    else:
        imageio.mimsave(outdir + name + '_gt.gif', out, duration=1000* 1/20)

from MotionScript.stmc_renderer.humor import HumorRenderer
def render_HQ_Salsa(motions, outdir='test_vis', device_id=0, name=None, pred=True):
    R_x = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    R_z = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    motion_rotated_x = np.einsum('ij,fkj->fki', R_x, motions)
    motion_final = np.einsum('ij,fkj->fki', R_z, motion_rotated_x)  # motion_rotated_y)
    # Now `motion_final` contains the correctly oriented animation
    motions = motion_final

    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)
    MAXS = motions.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    trajec = motions[:, 0, [0, 2]]

    j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True)
    rot2xyz = Rotation2xyz(device=torch.device("cuda:0"))
    faces = rot2xyz.smpl_model.faces

    if (not os.path.exists(outdir + name + '_pred.pt') and pred) or (
            not os.path.exists(outdir + name + '_gt.pt') and not pred):
        print(f'Running SMPLify, it may take a few minutes.')
        motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]

        vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                           pose_rep='rot6d', translation=True, glob=True,
                           jointstype='vertices',
                           vertstrans=True)


    # Visualize mesh:
    FACE_PATH = "utils/salsa_utils/libs/MotionScript/stmc_renderer/humor_render_tools/smplh.faces"
    FACES = (np.int32(np.load(FACE_PATH)))

    v = vertices.squeeze().permute(2, 0, 1).cpu().detach().numpy()
    f = faces.astype(np.int32)

    smpl_renderer = HumorRenderer(20, imw=720, imh=720)
    smpl_renderer(v, FACES,
                  output=os.path.join(outdir, name),  # 'smpl_video_path.mp4',
                  progress_bar=tqdm)

    return vertices, faces

def render_HQ_Salsa_pair(motions_leader, motions_follower, outdir='test_vis', device_id=0, name=None, pred=True):
    R_x = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    R_z = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])


    for dancer in ['leader', 'follower']:
        motions = motions_leader if dancer=='leader' else motions_follower
        motion_rotated_x = np.einsum('ij,fkj->fki', R_x, motions)
        motion_final = np.einsum('ij,fkj->fki', R_z, motion_rotated_x)  # motion_rotated_y)
        # Now `motion_final` contains the correctly oriented animation
        motions = motion_final

        frames, njoints, nfeats = motions.shape
        MINS = motions.min(axis=0).min(axis=0)
        MAXS = motions.max(axis=0).max(axis=0)

        height_offset = MINS[1]
        motions[:, :, 1] -= height_offset
        trajec = motions[:, 0, [0, 2]]

        j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True)
        rot2xyz = Rotation2xyz(device=torch.device("cuda:0"))
        faces = rot2xyz.smpl_model.faces

        if (not os.path.exists(outdir + name + '_pred.pt') and pred) or (
                not os.path.exists(outdir + name + '_gt.pt') and not pred):
            print(f'Running SMPLify, it may take a few minutes.')
            motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]

            vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                               pose_rep='rot6d', translation=True, glob=True,
                               jointstype='vertices',
                               vertstrans=True)
        if dancer == 'leader':
            vertices_L = copy.deepcopy(vertices)
        else:
            vertices_F = copy.deepcopy(vertices)

    # Visualize mesh:
    FACE_PATH = "utils/salsa_utils/libs/MotionScript/stmc_renderer/humor_render_tools/smplh.faces"
    FACES = (np.int32(np.load(FACE_PATH)))

    v = vertices.squeeze().permute(2, 0, 1).cpu().detach().numpy()
    f = faces.astype(np.int32)

    smpl_renderer = HumorRenderer(20, imw=720, imh=720)



    # smpl_renderer(v, FACES,
    #               output=os.path.join(outdir, name),  # 'smpl_video_path.mp4',
    #               progress_bar=tqdm)

    smpl_renderer.two_people(vertices_L.squeeze().permute(2, 0, 1).cpu().detach().numpy(), FACES,
                             vertices_F.squeeze().permute(2, 0, 1).cpu().detach().numpy(), FACES,
                  output=os.path.join(outdir, name),  # 'smpl_video_path.mp4',
                  progress_bar=tqdm)
    return vertices, faces


#
#     R_x = np.array([
#         [1, 0, 0],
#         [0, 0, -1],
#         [0, 1, 0]
#     ])
#     R_z = np.array([
#         [-1, 0, 0],
#         [0, -1, 0],
#         [0, 0, 1]
#     ])
#     motion_rotated_x = np.einsum('ij,fkj->fki', R_x, motions)
#     motion_final = np.einsum('ij,fkj->fki', R_z, motion_rotated_x)  # motion_rotated_y)
#     # Now `motion_final` contains the correctly oriented animation
#     motions = motion_final
#
#     frames, njoints, nfeats = motions.shape
#     MINS = motions.min(axis=0).min(axis=0)
#     MAXS = motions.max(axis=0).max(axis=0)
#
#     height_offset = MINS[1]
#     motions[:, :, 1] -= height_offset
#
#
#
#
#
#
#
#
# from MotionScript.stmc_renderer.humor import HumorRenderer
# smpl_renderer = HumorRenderer(20, imw=720, imh=720)
#
#
# v =vertices.squeeze().permute(2, 0, 1).cpu().detach().numpy()
# f = faces.astype(np.int32)
# smpl_renderer(v, FACES,
#                 output=f'out_temp/{name}_3D_Mesh.mp4',  # 'smpl_video_path.mp4',
#                 progress_bar=tqdm)
#
