# GLIP_finetuning_Custom_Data

**Custom Data yaml file:** 
* Create a yaml file as custom_data.v2-raw-1024.coco.yaml.
* Make change at OVERRIDE_CATEGORY, add categories of custom data.
* Change ann_file and img_file paths for test, train, and valid to custom data file paths.
* Change NUM_CLASSES.



**Fine-Tuning:**
```
python -m torch.distributed.launch --nproc_per_node=4 tools/finetune.py \
      --config-file {config_file}  --ft-tasks {configs} --skip-test \
      --custom_shot_and_epoch_and_general_copy {custom_shot_and_epoch_and_general_copy} \
      --evaluate_only_best_on_test --push_both_val_and_test \
      MODEL.WEIGHT {model_checkpoint} \
      SOLVER.USE_AMP True TEST.DURING_TRAINING True TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 4 SOLVER.WEIGHT_DECAY 0.05 TEST.EVAL_TASK detection DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding MODEL.BACKBONE.FREEZE_CONV_BODY_AT 2 MODEL.DYHEAD.USE_CHECKPOINT True SOLVER.FIND_UNUSED_PARAMETERS False SOLVER.TEST_WITH_INFERENCE True SOLVER.USE_AUTOSTEP True DATASETS.USE_OVERRIDE_CATEGORY True SOLVER.SEED 10 DATASETS.SHUFFLE_SEED 3 DATASETS.USE_CAPTION_PROMPT True DATASETS.DISABLE_SHUFFLE True \
      SOLVER.STEP_PATIENCE 3 SOLVER.CHECKPOINT_PER_EPOCH 1.0 SOLVER.AUTO_TERMINATE_PATIENCE 8 SOLVER.MODEL_EMA 0.0 SOLVER.TUNING_HIGHLEVEL_OVERRIDE full
```      

1. config_file: pretrained model's config file path (configs/pretrain/glip_Swin_T_O365_GoldG.yaml).
2. configs: custom data config file path (custom_data.v2-raw-1024.coco.yaml).
3. model_checkpoint: pretrained model's weights (MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth).


**Reference:**
1. https://github.com/microsoft/GLIP
2. https://github.com/microsoft/GLIP/issues/85
