PRETRAIN=checkpoints/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth
EXP_DIR=exps/r50.ua_detrac_mot

python main.py \
  --meta_arch motr \
  --dataset_file detrac_mot \
  --val_every 1 \
  --val_iou_thres 0.5 \
  --epoch 50 \
  --num_workers 12 \
  --with_box_refine \
  --lr_drop 40 \
  --save_period 1 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --pretrained ${PRETRAIN} \
  --output_dir ${EXP_DIR} \
  --batch_size 1 \
  --sample_mode 'random_interval' \
  --sample_interval 4 \
  --sampler_steps 12 22 36 \
  --sampler_lengths 2 3 4 5 \
  --update_query_pos \
  --merger_dropout 0 \
  --dropout 0 \
  --random_drop 0.1 \
  --fp_ratio 0.3 \
  --track_embedding_layer 'AttentionMergerV4' \
  --extra_track_attn \
  --data_txt_path_train ./datasets/data_path/detrac_mot.train \
  --data_txt_path_val ./datasets/data_path/detrac_mot.val \
  --mot_path ./datasets/DETRAC-MOT \
  --vis 