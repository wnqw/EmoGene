import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm, trange
import torch
import face_alignment
import deep_3drecon
from moviepy.editor import VideoFileClip
import copy
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, network_size=4, device='cuda')
face_reconstructor = deep_3drecon.Reconstructor()

# landmark detection in Deep3DRecon
def lm68_2_lm5(in_lm):
    # in_lm: shape=[68,2]
    lm_idx = np.array([31,37,40,43,46,49,55]) - 1
    # 将上述特殊角点的数据取出，得到5个新的角点数据，拼接起来。
    lm = np.stack([in_lm[lm_idx[0],:],np.mean(in_lm[lm_idx[[1,2]],:],0),np.mean(in_lm[lm_idx[[3,4]],:],0),in_lm[lm_idx[5],:],in_lm[lm_idx[6],:]], axis = 0)
    # 将第一个角点放在了第三个位置
    lm = lm[[1,2,0,3,4],:2]
    return lm

def process_video(fname, out_name=None):
    emotion = fname.split('_')[1]  # Extract emotion from the filename
    if out_name is None:
        out_name = f"{fname[:-4]}.npy"  # Include emotion in the output file name
    tmp_name = out_name[:-4] + '.doi'
    os.system(f"touch {tmp_name}")
    cap = cv2.VideoCapture(fname)
    print(f"loading video ...")
    num_frames = int(cap.get(7))
    h = int(cap.get(4))
    w = int(cap.get(3))
    mem = psutil.virtual_memory()
    a_mem = mem.available
    min_mem = num_frames * 68 * 2 + num_frames * 5 * 2 + num_frames * h * w * 3
    if a_mem < min_mem:
        print(f"WARNING: The physical memory is insufficient, which may result in memory swapping. Available Memory: {a_mem/1000000:.3f}M, the minimum memory required is:{min_mem/1000000:.3f}M.")
    # 初始化矩阵
    lm68_arr = np.empty((num_frames, 68, 2), dtype=np.float32)
    lm5_arr = np.empty((num_frames, 5, 2), dtype=np.float32)
    video_rgb = np.empty((num_frames, h, w, 3), dtype=np.uint8)
    cnt = 0
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_rgb[cnt] = frame_rgb
        cnt += 1
    for i in trange(num_frames, desc="Extracting 2D facial landmarks..."):
        try:
            lm68 = fa.get_landmarks(video_rgb[i])[0]
        except:
            print(f"No face detected at frame {i} in {fname}")
            os.system(f'rm {fname.replace(".mp4", ".wav")} {fname} ')
            continue
        lm5 = lm68_2_lm5(lm68)
        lm68_arr[i] = lm68
        lm5_arr[i] = lm5
    coeff_lst = []
    batch_size = 32
    for start_idx in range(0, num_frames, batch_size):
        end_idx = start_idx + batch_size
        batched_images = video_rgb[start_idx:end_idx]
        batched_lm5 = lm5_arr[start_idx:end_idx]
        try:
            coeff, align_img = face_reconstructor.recon_coeff(batched_images, batched_lm5, return_image=True)
        except:
            print(f"face_reconstructor.recon_coeff has problem at frame {i} in {fname}")
            os.system(f'rm {fname.replace(".mp4", ".wav")} {fname.replace(".mp4", "_coeff_pt.doi")} {fname} ')
            continue
        coeff_lst.append(coeff)
    try:
        coeff_arr = np.concatenate(coeff_lst, axis=0)
        coeff_arr = coeff_arr.reshape([cnt, -1])
    except:
        print(f"coeff_arr has problem in {fname}")
        os.system(f'rm {fname.replace(".mp4", ".wav")} {fname.replace(".mp4", "_coeff_pt.doi")} {fname} ')
        return
    result_dict = {
        'coeff': coeff_arr,
        'lm68': lm68_arr,
        'lm5': lm5_arr,
        'emotion': emotion  # Include emotion in results
    }
    np.save(out_name, result_dict)
    os.system(f"rm {tmp_name}")

def split_wav(mp4_name):
    wav_name = mp4_name[:-4] + '.wav'
    if os.path.exists(wav_name):
        return
    video = VideoFileClip(mp4_name,verbose=False)
    dur = video.duration
    audio = video.audio 
    assert audio is not None
    audio.write_audiofile(wav_name,fps=16000,verbose=False,logger=None)

if __name__ == '__main__':
    ### Process short video clips for MEAD dataset
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument('--mead_path', type=int, default='/home/yezhenhui/datasets/raw/lrs3_raw', help='')
    parser.add_argument('--mead_path', type=str, default='/mnt/sda4/mead_data/mead', help='')
    parser.add_argument('--process_id', type=int, default=0, help='')
    parser.add_argument('--total_process', type=int, default=1, help='')
    args = parser.parse_args()

    import os, glob
    mead_dir = args.mead_path
    mp4_name_pattern = os.path.join(mead_dir, "*/*.mp4")
    mp4_names = glob.glob(mp4_name_pattern)
    mp4_names = sorted(mp4_names)
    if args.total_process > 1:
        assert args.process_id <= args.total_process-1
        num_samples_per_process = len(mp4_names) // args.total_process
        if args.process_id == args.total_process-1:
            mp4_names = mp4_names[args.process_id * num_samples_per_process : ]
        else:
            mp4_names = mp4_names[args.process_id * num_samples_per_process : (args.process_id+1) * num_samples_per_process]
    for mp4_name in tqdm(mp4_names, desc='extracting 3DMM...'):
        split_wav(mp4_name)
        process_video(mp4_name,out_name=mp4_name.replace(".mp4", "_coeff_pt.npy"))

