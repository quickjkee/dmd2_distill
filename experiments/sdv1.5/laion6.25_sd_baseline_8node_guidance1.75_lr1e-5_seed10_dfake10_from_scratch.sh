export CHECKPOINT_PATH=$1
export WANDB_ENTITY=$2
export WANDB_PROJECT=$3
export MASTER_IP=$4

python3 -m torch.distributed.run --rdzv_endpoint=0.0.0.0:1206 --nproc_per_node=8  main/train_sd.py \
    --generator_lr 1e-5  \
    --guidance_lr 1e-5 \
    --train_iters 100000000 \
    --output_path  results \
    --batch_size 32 \
    --grid_size 2 \
    --initialie_generator --log_iters 1000 \
    --resolution 512 \
    --latent_resolution 64 \
    --seed 10 \
    --real_guidance_scale 1.75 \
    --fake_guidance_scale 1.0 \
    --max_grad_norm 10.0 \
    --model_id "sd-v1-5" \
    --train_prompt_path prompts/captions_laion_score6.25.pkl \
    --eval_prompt_path prompts/captions_coco14_test.pkl \
    --total_eval_samples 100 \
    --test_visual_batch_size 20 \
    --checkpointing_steps 100 \
    --wandb_iters 100000 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_name "laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch"  \
    --wandb_project $WANDB_PROJECT \
    --use_fp16 \
    --log_loss \
    --dfake_gen_update_ratio 10 \
    --gradient_checkpointing 


