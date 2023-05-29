export MODEL_NAME="Bingsu/my-korean-stable-diffusion-v1-5"
export dataset_name="jeongwoo25/mmorpg_world" # mmorpg_[world|society|science|economy]

accelerate launch --mixed_precision="fp16" --num_processes=2 train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=10000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --checkpointing_steps=1000 \
  --num_train_epochs 1 \
  --prompt_tensor_pretrained_path=null \
  --prompt_tensor_save_path="./test.pt" \
  --hidden_dim=768 \
  --output_dir="mmorpg_world" # mmorpg_[world|society|science|economy]