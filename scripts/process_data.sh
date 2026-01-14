export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export VIDEO_ID=


ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 data/raw/videos/${VIDEO_ID}_512.mp4
mv data/raw/videos/${VIDEO_ID}.mp4 data/raw/videos/${VIDEO_ID}_to_rm.mp4
mv data/raw/videos/${VIDEO_ID}_512.mp4 data/raw/videos/${VIDEO_ID}.mp4

mkdir -p data/processed/videos/${VIDEO_ID}
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -f wav -ar 16000 data/processed/videos/${VIDEO_ID}/aud.wav 
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python data_gen/utils/process_audio/extract_hubert.py --video_id=${VIDEO_ID}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python data_gen/utils/process_audio/extract_mel_f0.py --video_id=${VIDEO_ID}

mkdir -p data/processed/videos/${VIDEO_ID}/gt_imgs
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -start_number 0 data/processed/videos/${VIDEO_ID}/gt_imgs/%08d.jpg
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 --force_single_process

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python data_gen/utils/process_video/extract_lm2d.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python data_gen/utils/process_video/fit_3dmm_landmark.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 --reset  --id_mode=global # --debug

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python data_gen/runs/binarizer_nerf.py --video_id=${VIDEO_ID}

