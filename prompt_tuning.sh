export MODEL_NAME="Bingsu/my-korean-stable-diffusion-v1-5"
export dataset_name="angdong/nate-news-science"

accelerate launch --mixed_precision="fp16" --num_processes=1 --num_machines=1 text2img_prompt_tuning.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --cache_dir=$cache_dir \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --max_train_steps=10000 \
  --validation_steps=1000 \
  --prompt_tensor_pretrained_path=null \
  --prompt_tensor_save_path="./test.pt" \
  --hidden_dim=768 \
  --output_dir="mmorpg-science-prompt-tuning"