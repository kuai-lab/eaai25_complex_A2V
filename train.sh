
    accelerate launch \
  --mixed_precision "fp16" \
  --config_file "train_config.yaml" \
  train/train_textencoder.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1" \
  --checkpointing_steps 10000 \
  --tracker_project_name "EAAI24" \
  --report_to "wandb" \
  --logging_dir "logs" \
  --train_test "train" \
  --data_type "vgg" \
  --data_dir "/hdd/vggsound/" \
  --resolution 512 \
  --num_quries 10 \
  --interpolation \
  --output_dir "./output/" \
  --dataloader_num_workers 4 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 70000 \
  --lr_scheduler "constant" \
  --lr_warmup_steps 0 \
  --learning_rate 1e-05 \
  --denoising_loss 0.9 \
  --mse_loss 0.1
  # --resume_from_checkpoint "checkpoint-30000"