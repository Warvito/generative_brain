seed=42
run_dir="aekl_v0_ldm_v0_upsampler"
training_ids="/project/outputs/ids/train.tsv"
validation_ids="/project/outputs/ids/validation.tsv"
stage1_uri="/project/mlruns/751866889003045521/a2cb93e5a920445e932ce29e09bd5a82/artifacts/final_model"
config_file="/project/configs/upsampler_ldm/ldm_v0.yaml"
scale_factor=0.3
batch_size=4
n_epochs=25
eval_freq=1
num_workers=8
experiment="LDM"

runai submit \
  --name brain-upsampler-ldm-v0 \
  --image aicregistry:5000/wds20:ldm_brain \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 4 \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/generative_brain/:/project/ \
  --volume /nfs/project/AMIGO/Biobank/derivatives/super-res/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
      python3 /project/src/python/training/train_upsampler_ldm.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      stage1_uri=${stage1_uri} \
      config_file=${config_file} \
      scale_factor=${scale_factor} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      experiment=${experiment}
