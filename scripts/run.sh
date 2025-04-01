seed=1100,1200,1300,1400,1500

config='configs/inference_t2v_512_v2.0.yaml'

res_dir="results/"

name="thunder_ambulance"
ambulance="audios/ambulance.wav"
thunder="audios/thunder.wav"

ckpt='checkpoints/base_512_v2/model.ckpt'
audio_ckpt_path="checkpoints/cim"
trailblazer_yaml_path="TrailBlazer/MultiSubjects.yaml"

CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/inference_2object.py \
--seed ${seed} \
--mode 'base' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--fps 8 \
--frames 16 \
--step_ctrl 6 \
--audio_path1 $thunder \
--audio_path2 $ambulance \
--num_queries 10 \
--audio_ckpt_path $audio_ckpt_path \
--savefps 10 \
--original False \
--trailblazer_config $trailblazer_yaml_path \
--verbose \
--pos "LR"