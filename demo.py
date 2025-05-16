import random

import tqdm
from openai import AzureOpenAI
from models.motion_agent import MotionAgent
from models.mllm import MotionLLM
from options.option_llm import get_args_parser
from utils.motion_utils import recover_from_ric, plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
import torch

import os
import sys

# Path to the FFmpeg bin directory
ffmpeg_path = r'S:\Payam\LAMMA\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin'
import matplotlib as mpl
import shutil
from models.training_utils import *
ffmpeg_path = shutil.which("ffmpeg")
mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path


# Check if the path is already in the PATH variable
if ffmpeg_path not in os.environ['PATH']:
    # Add the path to the PATH environment variable
    os.environ['PATH'] += os.pathsep + ffmpeg_path



def motion_agent_demo():
    # Initialize the client
    client = AzureOpenAI(
        api_key="DDJ7XQo5NWYSxRGwOecZNt7VOwaEEBKWh8eJVWS7YqyCGarpMlviJQQJ99BDACBsN54XJ3w3AAABACOGm6nK", # your api key
        api_version="2024-10-21",
        azure_endpoint="https://salsa.openai.azure.com/" # your azure endpoint
    )

    endpoint = "https://pjome-m9a36rct-eastus2.cognitiveservices.azure.com/"
    model_name = "gpt-35-turbo"
    deployment = "gpt-35-turbo-Salsa"
    subscription_key = "uPN20WiZ0suB5YVbTkwkVjT8Rqdwudlg0fqMaZGu9uF9pQY5oHI4JQQJ99BDACHYHv6XJ3w3AAAAACOGtmqx"
    api_version = "2024-12-01-preview"
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    # No Azure:
    from openai import OpenAI
    API_KEY = "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"
    client = OpenAI(api_key=API_KEY)

    args = get_args_parser()
    args.save_dir = "./demo"
    args.device = 'cuda:0'

    motion_agent = MotionAgent(args, client)
    motion_agent.chat()

def motionllm_demo():
    model = MotionLLM(get_args_parser())
    model.load_model('ckpt/motionllm.pth')
    model.llm.eval()
    model.llm.cuda()
    
    caption = 'A man is doing cartwheels.'
    motion = model.generate(caption)

    motion = model.denormalize(motion.detach().cpu().numpy())
    motion = recover_from_ric(torch.from_numpy(motion).float().cuda(), 22)
    print(motion.shape)
    plot_3d_motion(f"motionllm_demo.mp4", t2m_kinematic_chain, motion.squeeze().detach().cpu().numpy(), title=caption, fps=20, radius=4)

from utils.salsa_utils.salsa_dataloader import SALSA_CAPTIONS
import pickle
os.chdir('Motion-Agent-Salsa') # to refine the data loader felan.

from utils.salsa_utils.libs.MotionScript.ms_utils_visu import render_HQ_Salsa, render_HQ_Salsa_pair
def motionllm_evaluation_qualitative():


    args = get_args_parser()
    args.save_dir = "./demo/eval4Ahmet"
    args.device = 'cuda:0'


    model = MotionLLM(args)
    # fine-tuned on text-to-motion
    # model.load_model('output_trained/Second trial/Xmotionllm_epoch25.pth')

    # Baseline
    model.load_model('ckpt/motionllm.pth')

    model.llm.eval()
    model.llm.cuda()



    for style in SALSA_CAPTIONS:
        # if style != 'professional': continue
        for iterate in tqdm.tqdm(range(1)):

            motion_tokens_to_generate = []
            captions_list = []
            for t in range(3):
                caption = random.choice(SALSA_CAPTIONS[style])
                captions_list.append(caption)
                motion_tokens = model.generate(caption)
                motion_tokens_to_generate.append(motion_tokens)

            motion_tokens = torch.cat(motion_tokens_to_generate)
            motion = model.net.forward_decoder(motion_tokens)
            motion = model.denormalize(motion.detach().cpu().numpy())

            positions = recover_from_ric(torch.from_numpy(motion).float().cuda(), 22)
            # print(motion.shape)


            if iterate < 5:
                sav_path = f"./demo/eval4Ahmet/{style}"

                os.makedirs(sav_path, exist_ok=True)
                plot_3d_motion(os.path.join(sav_path,
                                f"motionllm_{style}_{iterate}.mp4"),
                                t2m_kinematic_chain, positions.squeeze().detach().cpu().numpy(),
                                title='\n'.join(captions_list),
                                fps=20, radius=4)


                vertices, faces = render_HQ_Salsa(positions.squeeze().detach().cpu().numpy(),
                                sav_path,
                                name=f'motionllm_{style}_{iterate}_3DMesh.mp4')

                # For Shay
                my_dict = {'caption': caption,
                           'HML3D_vec': motion.squeeze(),
                           'positions': positions.squeeze().detach().cpu().numpy(),
                           'vertuces': vertices,
                           'faces': faces}
                with open(f"{sav_path}/motionllm_{style}_{iterate}.pk", "wb") as f:
                    pickle.dump(my_dict, f)

                    '''
def motionllm_evaluation_qualitative():


    args = get_args_parser()
    args.save_dir = "./demo/eval_new"
    args.device = 'cuda:0'


    model = MotionLLM(args)
    # fine-tuned on text-to-motion
    # model.load_model('output_trained/Second trial/Xmotionllm_epoch25.pth')

    # Baseline
    model.load_model('ckpt/motionllm.pth')

    model.llm.eval()
    model.llm.cuda()



    for style in SALSA_CAPTIONS:
        # if style != 'professional': continue
        for iterate in tqdm.tqdm(range(1)):

            motion_tokens_to_generate = []
            captions_list = []
            for t in range(3):
                caption = random.choice(SALSA_CAPTIONS[style])
                captions_list.append(caption)
                motion_tokens = model.generate(caption)
                motion_tokens_to_generate.append(motion_tokens)

            motion_tokens = torch.cat(motion_tokens_to_generate)
            motion = model.net.forward_decoder(motion_tokens)
            motion = model.denormalize(motion.detach().cpu().numpy())

            positions = recover_from_ric(torch.from_numpy(motion).float().cuda(), 22)
            # print(motion.shape)


            if iterate < 5:
                sav_path = f"./demo/eval4Ahmet/{style}"

                os.makedirs(sav_path, exist_ok=True)
                plot_3d_motion(os.path.join(sav_path,
                                f"motionllm_{style}_{iterate}.mp4"),
                                t2m_kinematic_chain, positions.squeeze().detach().cpu().numpy(),
                                title='\n'.join(captions_list),
                                fps=20, radius=4)


                vertices, faces = render_HQ_Salsa(positions.squeeze().detach().cpu().numpy(),
                                sav_path,
                                name=f'motionllm_{style}_{iterate}_3DMesh.mp4')

                # For Shay
                my_dict = {'caption': caption,
                           'HML3D_vec': motion.squeeze(),
                           'positions': positions.squeeze().detach().cpu().numpy(),
                           'vertuces': vertices,
                           'faces': faces}
                with open(f"{sav_path}/motionllm_{style}_{iterate}.pk", "wb") as f:
                    pickle.dump(my_dict, f)
                    '''


def motionllm_evaluation_qualitative_pairs():


    args = get_args_parser()
    args.save_dir = "./demo/eval4Ahmet"
    args.device = 'cuda:0'


    model = MotionLLM(args)
    # fine-tuned on text-to-motion
    # model.load_model('output_trained/Second trial/Xmotionllm_epoch25.pth')

    # Baseline
    is_baseline = True
    model = MotionLLM(args) # this should be manually fixed in mllm.py
    model.load_model('ckpt/motionllm.pth')
    current_batch_task = 'caption_to_motion'

    # model.load_model('output_trained\pretrain_all/Xmotionllm_epoch5.pth')

    # model.load_model('output_trained\pretrain_all/Xmotionllm_epoch5.pth')

    # model.load_model('output_trained/follower_to_leader_v3/Xmotionllm_epoch10.pth') # 'follower_to_leader'
    # current_batch_task = 'follower_to_leader'

    # model.load_model('output_trained/leader_to_follower_v3/Xmotionllm_epoch42.pth')  # 'follower_to_leader'
    # current_batch_task = 'leader_to_follower'


    # model.load_model('output_trained/caption_to_motion_v3/Xmotionllm_epoch100.pth')  # 'follower_to_leader'
    # current_batch_task = 'caption_to_motion'

    model.llm.eval()
    model.llm.cuda()

    motion_tokens_to_generate = []
    follower_motion_tokens = []
    leader_motion_tokens = []

    H3D_GT_Leader = []
    H3D_GT_Follower = []

    # Pair < Number > _take < Song > _ < Take > _follower_subject.bvh
    test_set = [
        {"Pair": "Pair1", "Level": "Beginner", "SongTake": "take1_1"},
        {"Pair": "Pair2", "Level": "Intermediate", "SongTake": "take1_2"},
        {"Pair": "Pair3", "Level": "Beginner", "SongTake": "take2_1"},
        {"Pair": "Pair4", "Level": "Intermediate", "SongTake": "take2_2"},
        {"Pair": "Pair5", "Level": "Professional", "SongTake": "take3_1"},
        {"Pair": "Pair6", "Level": "Intermediate", "SongTake": "take3_2"},
        {"Pair": "Pair7", "Level": "Professional", "SongTake": "take4_1"},
        {"Pair": "Pair8", "Level": "Beginner", "SongTake": "take4_2"},
        {"Pair": "Pair9", "Level": "Professional", "SongTake": "take1_1"},
    ]

    # now iterate
    for row in test_set:
        Pair = row["Pair"]
        Style = row["Level"].lower()  # e.g. "Beginner" → "beginner"
        my_take = row["SongTake"]
        # Style, Pair = 'beginner', "Pair1"
        # Style, Pair = 'professional', "Pair3"
        # my_take = 'tale1_1'
        s, e = 0, 30
        items = load_data_pair(style=Style, pair=Pair, take=my_take, start_sec=s, end_sec=e)
        if items==[]:
            print("Missing!!! ", Pair, Style, my_take)
            with open("Missings_infer.txt", "a") as f:
                f.write(f"{Style} | {Pair} | {my_take}\n")
            continue
        args.is_MDM = True # at the inference to get GT

        motion_tokens_to_generate = []
        follower_motion_tokens = []
        leader_motion_tokens = []

        H3D_GT_Leader = []
        H3D_GT_Follower = []

        for item in items:
            if not args.is_MDM:
                level, ms_desc_L, ms_des_F, \
                    vq_tokens_L, vq_tokens_F, audio_tokens, aux_info = item

            else:
                level, HML3D_L, vq_tokens_L, HML3D_F, vq_tokens_F, audio_tokens, aux_info = item
                ms_desc_L = ms_des_F = 'a-->b'

            # captions_list = []
            level = Style

            full_prompt, input_ids = build_random_training_instance_salsa_prompt(
                tokenizer=model.tokenizer,
                leader_motion_script_segments=ms_desc_L.split('-->'),
                follower_motion_script_segments=ms_des_F.split('-->'),
                leader_motion_tokens=vq_tokens_L,
                follower_motion_tokens=vq_tokens_F,
                audio_tokens=audio_tokens,
                proficiency_level=level,
                allowed_tasks=[current_batch_task],
                snippet_prob=0.5,
                min_snippet_steps=1,
                max_snippet_steps=4,
                inference=True,
                is_baseline=True
            )



            motion_tokens = model.generate_Payam(full_prompt, input_ids)
            motion_tokens_to_generate.append(motion_tokens)

            leader_motion_tokens.append(vq_tokens_L)
            follower_motion_tokens.append(vq_tokens_F)

            # store GT:
            H3D_GT_Leader.append(HML3D_L)
            H3D_GT_Follower.append(HML3D_F)

            # the_other_motion_tokens.append(vq_tokens_F if current_batch_task=='follower_to_leader' else vq_tokens_F)

        print("Inference completed.\nExporting results...")
        #Todo ------------High priority---------------------
        # We need to feed the first frame to keep continuity and etc.
        motion_tokens = torch.cat(motion_tokens_to_generate)
        motion = model.net.forward_decoder(motion_tokens)
        motion = model.denormalize(motion.detach().cpu().numpy())
        positions = recover_from_ric(torch.from_numpy(motion).float().cuda(), 22)
        # print(motion.shape)

        # Leader:
        leader_motion_tokens = torch.cat(leader_motion_tokens)
        leader_motion = model.net.forward_decoder(leader_motion_tokens)
        leader_motion = model.denormalize(leader_motion.detach().cpu().numpy())
        leader_positions = recover_from_ric(torch.from_numpy(leader_motion).float().cuda(), 22)
        # print(motion.shape)
        # Leader:
        follower_motion_tokens = torch.cat(follower_motion_tokens)
        follower_motion = model.net.forward_decoder(follower_motion_tokens)
        follower_motion = model.denormalize(follower_motion.detach().cpu().numpy())
        follower_positions = recover_from_ric(torch.from_numpy(follower_motion).float().cuda(), 22)

        # Original motion GT
        H3D_GT_Leader = torch.cat(H3D_GT_Leader)
        H3D_GT_Follower = torch.cat(H3D_GT_Follower)

        H3D_GT_Leader_Positions = recover_from_ric(H3D_GT_Leader.unsqueeze(0).float().cuda(), 22)
        H3D_GT_Follower_positions = recover_from_ric(H3D_GT_Leader.unsqueeze(0).float().cuda(), 22)


        sav_path = f"./demo/eval4Bermet/{current_batch_task}/{level}"
        iterate = Pair + '_' + my_take
        # Todo: test two people mesh:
        # just for follower to leader task for now
        # leader_positions, follower_positions = the_other_positions, positions
        if False:
            vertices1, faces1, vertices2, faces2 = render_HQ_Salsa_pair(leader_positions.squeeze().detach().cpu().numpy(),
                                                                        follower_positions.squeeze().detach().cpu().numpy(),
                                                                          sav_path,
                                                                          name=f'motionllm_{level}_{iterate}_3DMesh_predicted.mp4')

            exit()


        if True:
            sav_path = f"./demo/eval4Bermet/{current_batch_task}/{level}/{Pair}_{my_take}"

            os.makedirs(sav_path, exist_ok=True)
            plot_3d_motion(os.path.join(sav_path,
                            f"motionllm_{level}_{iterate}_predicted.mp4"),
                            t2m_kinematic_chain, positions.squeeze().detach().cpu().numpy(),
                            title=(f"Generated {'leader' if current_batch_task=='follower_to_leader' else 'follower'}: {Style} {Pair}"),
                            fps=20, radius=4)


            # vertices, faces = render_HQ_Salsa(positions.squeeze().detach().cpu().numpy(),
            #                 sav_path,
            #                 name=f'motionllm_{level}_{iterate}_3DMesh_predicted.mp4')
            vertices, faces = None, None
            # For Shay
            my_dict = {'caption': "infer",
                       'HML3D_vec': motion.squeeze(),
                       'positions': positions.squeeze().detach().cpu().numpy(),
                       'vertuces': vertices,
                       'faces': faces}
            with open(f"{sav_path}/motionllm_{level}_{iterate}_predicted.pk", "wb") as f:
                pickle.dump(my_dict, f)




            tx = 'input(recon)' if current_batch_task=='leader_to_follower' else 'GT(recon)'
            plot_3d_motion(os.path.join(sav_path,
                                        f"motionllm_{level}_{iterate}_leader_{tx}.mp4"),
                           t2m_kinematic_chain, leader_positions.squeeze().detach().cpu().numpy(),
                           title=(f"Leader {tx}: {Style} {Pair}"),
                           fps=20, radius=4)
            my_dict = {'caption': "infer",
                       'HML3D_vec': leader_motion.squeeze(),
                       'positions': leader_positions.squeeze().detach().cpu().numpy(),
                       'vertuces': vertices,
                       'faces': faces}
            with open(f"{sav_path}/motionllm_{level}_{iterate}_leader_{tx}.pk", "wb") as f:
                pickle.dump(my_dict, f)

            tx = 'input(recon)' if current_batch_task=='follower_to_leader' else 'GT(recon)'
            plot_3d_motion(os.path.join(sav_path,
                                        f"motionllm_{level}_{iterate}_follower_{tx}.mp4"),
                           t2m_kinematic_chain, follower_positions.squeeze().detach().cpu().numpy(),
                           title=(f"Follower {tx}: {Style} {Pair}"),
                           fps=20, radius=4)
            my_dict = {'caption': "infer",
                       'HML3D_vec': follower_motion.squeeze(),
                       'positions': follower_positions.squeeze().detach().cpu().numpy(),
                       'vertuces': vertices,
                       'faces': faces}
            with open(f"{sav_path}/motionllm_{level}_{iterate}_follower_{tx}.pk", "wb") as f:
                pickle.dump(my_dict, f)





    #         -------------------


            plot_3d_motion(os.path.join(sav_path,
                                        f"motionllm_{level}_{iterate}_leader_GT.mp4"),
                           t2m_kinematic_chain, H3D_GT_Leader_Positions.squeeze().detach().cpu().numpy(),
                           title=(f"Leader GT: {Style} {Pair}"),
                           fps=20, radius=4)
            my_dict = {'caption': "infer",
                       'HML3D_vec': H3D_GT_Leader.squeeze(),
                       'positions': H3D_GT_Leader_Positions.squeeze().detach().cpu().numpy(),
                       'vertuces': vertices,
                       'faces': faces}
            with open(f"{sav_path}/motionllm_{level}_{iterate}_leader_GT.pk", "wb") as f:
                pickle.dump(my_dict, f)


            plot_3d_motion(os.path.join(sav_path,
                                        f"motionllm_{level}_{iterate}_follower_GT.mp4"),
                           t2m_kinematic_chain, H3D_GT_Follower_positions.squeeze().detach().cpu().numpy(),
                           title=(f"Follower GT: {Style} {Pair}"),
                           fps=20, radius=4)
            my_dict = {'caption': "infer",
                       'HML3D_vec': H3D_GT_Follower.squeeze(),
                       'positions': H3D_GT_Follower_positions.squeeze().detach().cpu().numpy(),
                       'vertuces': vertices,
                       'faces': faces}
            with open(f"{sav_path}/motionllm_{level}_{iterate}_follower_GT.pk", "wb") as f:
                pickle.dump(my_dict, f)


        # for Shay
        if False:
            #         Draw for shay: [22, 3] shay is [165]
            import math
            rotX = lambda theta: torch.tensor([[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]])
            rotY = lambda theta: torch.tensor([[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]])
            rotZ = lambda theta: torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]])

            def transf(rotMat, theta_deg, values):
                theta_rad = (math.pi * torch.tensor(theta_deg).float() / 180.0).to(torch.float)
                return rotMat(theta_rad).mm(values.t()).t()
            import numpy as np
            folder_path = 'Shay/duolando_follower_gen'  # replace with your folder path
            sav_path = folder_path
            for filename in os.listdir(folder_path):
                if filename.endswith('.npy'):
                    file_path = os.path.join(folder_path, filename)
                    data = np.load(file_path)
                    data = torch.from_numpy(data)
                    j_seq = data.view(data.shape[0], data.shape[1], -1, 3)
                    for b in range(j_seq.shape[0]):
                        for frame_i in range(j_seq.shape[1]):
                            j_seq[b, frame_i] = transf(rotX, -90, j_seq[b, frame_i])
                        print(f"Loaded {filename}, shape: {data.shape}")
                        plot_3d_motion(os.path.join(sav_path,
                                                    filename.replace('npy', 'mp4')),
                                       t2m_kinematic_chain, j_seq[:, :, :].squeeze().detach().cpu().numpy(),
                                       title='aa',

                                       fps=20, radius=4)



'''
def motionllm_evaluation_qualitative():


    args = get_args_parser()
    args.save_dir = "./demo/eval_new"
    args.device = 'cuda:0'


    model = MotionLLM(args)
    # fine-tuned on text-to-motion
    # model.load_model('output_trained/Second trial/Xmotionllm_epoch25.pth')

    # Baseline
    model.load_model('ckpt/motionllm.pth')

    model.llm.eval()
    model.llm.cuda()



    for style in SALSA_CAPTIONS:
        # if style != 'professional': continue
        for iterate in tqdm.tqdm(range(1)):

            motion_tokens_to_generate = []
            captions_list = []
            for t in range(3):
                caption = random.choice(SALSA_CAPTIONS[style])
                captions_list.append(caption)
                motion_tokens = model.generate(caption)
                motion_tokens_to_generate.append(motion_tokens)

            motion_tokens = torch.cat(motion_tokens_to_generate)
            motion = model.net.forward_decoder(motion_tokens)
            motion = model.denormalize(motion.detach().cpu().numpy())

            positions = recover_from_ric(torch.from_numpy(motion).float().cuda(), 22)
            # print(motion.shape)


            if iterate < 5:
                sav_path = f"./demo/eval4Ahmet/{style}"

                os.makedirs(sav_path, exist_ok=True)
                plot_3d_motion(os.path.join(sav_path,
                                f"motionllm_{style}_{iterate}.mp4"),
                                t2m_kinematic_chain, positions.squeeze().detach().cpu().numpy(),
                                title='\n'.join(captions_list),
                                fps=20, radius=4)


                vertices, faces = render_HQ_Salsa(positions.squeeze().detach().cpu().numpy(),
                                sav_path,
                                name=f'motionllm_{style}_{iterate}_3DMesh.mp4')

                # For Shay
                my_dict = {'caption': caption,
                           'HML3D_vec': motion.squeeze(),
                           'positions': positions.squeeze().detach().cpu().numpy(),
                           'vertuces': vertices,
                           'faces': faces}
                with open(f"{sav_path}/motionllm_{style}_{iterate}.pk", "wb") as f:
                    pickle.dump(my_dict, f)
'''

from utils.salsa_utils.salsa_dataloader import Salsa_Dataset
def load_data_pair(style='beginner', pair="Pair1", take='take1_1', start_sec=0, end_sec=5):
    args = get_args_parser()
    # args.is_MDM = True # to get HumanML3D outputs
    args.is_MDM = True
    train_dataset = Salsa_Dataset(args,
                    lmdb_dir='utils/salsa_utils/Salsa_Temp/lmdb_Salsa_pair/lmdb_train',
                    n_poses=100,
                    subdivision_stride=50,
                    pose_resampling_fps=20)

    item = train_dataset.__getitem__(1)
    print()

    #   Workaround striding
    #   pick style and pair:
    # style = "beginner"
    # pair = "Pair1"
    # start_sec, end_sec = 20, 35
    # find the indices '-->'
    list_of_indecis = []
    list_of_aux = []

    for index in range(len(train_dataset)):
        if not args.is_MDM:
            level, ms_desc_L, ms_des_F, \
                vq_tokens_L, vq_tokens_F, audio_tokens, aux_info = train_dataset.__getitem__(index)
        else:
            level, HML3D_L, vq_tokens_L, HML3D_F, vq_tokens_F, audio_tokens, aux_info = train_dataset.__getitem__(index)
        level = style #todo fixeme: cheating at the inference time due to incorrect lmdb processing at the time.
        if level==style and \
                aux_info['start_time']>=start_sec and \
                aux_info['end_time'] <= end_sec and \
                pair==aux_info['vid'][:5] and \
                take in aux_info['vid']: #
            if len(list_of_aux)>0:
                if aux_info['start_time']>=list_of_aux[-1]['end_time']:
                    list_of_indecis.append(index)
                    list_of_aux.append(aux_info)
            else:
                list_of_indecis.append(index)
                list_of_aux.append(aux_info)

        if len(list_of_indecis)>0 and \
                list_of_aux[-1]['end_time'] == end_sec and \
                list_of_aux[0]['start_time'] == start_sec:
            break



    items = [train_dataset.__getitem__(index) for index in list_of_indecis]
    return items

if __name__ == "__main__":

    # motion_agent_demo()


    # motionllm_evaluation_qualitative()

    motionllm_evaluation_qualitative_pairs()

