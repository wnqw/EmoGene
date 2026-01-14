export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export VIDEO_ID=
export NAME=


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tasks/run.py --config=egs/datasets/${VIDEO_ID}/lm3d_radnerf_sr.yaml --exp_name=motion2video_nerf/${NAME}_head --reset

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python tasks/run.py --config=egs/datasets/${VIDEO_ID}/lm3d_radnerf_torso_sr.yaml --exp_name=motion2video_nerf/${NAME}_torso --hparams=head_model_dir=checkpoints/motion2video_nerf/${NAME}_head --reset 

