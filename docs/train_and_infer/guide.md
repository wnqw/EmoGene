# Get Audio2Motion Model

You can download the pre-trained Audio-to-Motion model (pretrained on voxceleb2, a 2000-hour lip reading dataset) in this [Google Drive](https://drive.google.com/drive/folders/1FqvNbQgOSkvVO8i-vCDJmKM4ppPZjUpL?usp=sharing).

Place the model in the directory `checkpoints/audio2motion_vae`.

# Train Motion2Emotion Model
```
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/mead/lm_deformation.yaml --exp_name=mead/lm_deformation
```
Place the model in the directory `checkpoints/mead/lm_deformation`.


# Train Emotion2Video Model
We suppose you have prepared the dataset following `docs/prepare_data/guide.md` and you can find a binarized `.npy` file in `data/binary/videos/{Video_ID}/trainval_dataset.npy` 

```
# Train the Head NeRF
# model and tensorboard will be saved at `checkpoints/<exp_name>`
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/{Video_ID}/lm3d_radnerf_sr.yaml --exp_name=motion2video_nerf/{Video_ID}_head --reset

# Train the Torso NeRF
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/{Video_ID}/lm3d_radnerf_torso_sr.yaml --exp_name=motion2video_nerf/{Video_ID}_torso --hparams=head_model_dir=checkpoints/motion2video_nerf/{Video_ID}_head --reset
```

## How to train on your own video: 
Suppose you have a video named `{Video_ID}.mp4`
- Step1: crop your video to 512x512 and 25fps, then place it into `data/raw/videos/{Video_ID}.mp4`
- Step2: copy a config folder `egs/datasets/{Video_ID}` 
- Step3: Process the video following `docs/process_data/guide.md`, then you can get a `data/binary/videos/{Video_ID}/trainval_dataset.npy`
- Step4: Use the commandlines above to train the NeRF.

# Inference
```
# we provide a inference script in infer.sh:
export PYTHONPATH=./
export NAME=
export OUTPUT=
export WAV=

# emotion: angry, disgust, contempt, fear, happy, sad, surprise, neutral

CUDA_VISIBLE_DEVICES=1  python inference/emogene_infer.py  --head_ckpt=checkpoints/motion2video_nerf/${NAME}_head --torso_ckpt=checkpoints/motion2video_nerf/${NAME}_torso \
        --drv_aud=data/raw/val_wavs/${WAV}.wav --out_name=results/${OUTPUT}.mp4 --emotion happy --lm_deform_delta 0 --blink_mode period \
        --a2m_ckpt checkpoints/audio2motion_vae --lm_deform_ckpt checkpoints/mead/lm_deformation 
```
