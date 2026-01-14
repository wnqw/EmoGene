
export PYTHONPATH=./
export NAME=
export OUTPUT=
export WAV=

# emotion: angry, disgust, contempt, fear, happy, sad, surprise, neutral

CUDA_VISIBLE_DEVICES=1  python inference/emogene_infer.py  --head_ckpt=checkpoints/motion2video_nerf/${NAME}_head --torso_ckpt=checkpoints/motion2video_nerf/${NAME}_torso \
        --drv_aud=data/raw/val_wavs/${WAV}.wav --out_name=results/${OUTPUT}.mp4 --emotion happy --lm_deform_delta 0 --blink_mode period \
        --a2m_ckpt checkpoints/audio2motion_vae --lm_deform_ckpt checkpoints/mead/lm_deformation 
