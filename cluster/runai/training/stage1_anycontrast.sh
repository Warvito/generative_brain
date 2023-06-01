seed=42
run_dir="aekl_anycontrast_v0"
training_ids="/project/outputs/ids/train_anycontrast.tsv"
validation_ids="/project/outputs/ids/validation_anycontrast.tsv"
config_file="/project/configs/stage1/aekl_v0.yaml"
batch_size=1
n_epochs=20
adv_start=5
eval_freq=1
num_workers=8
experiment="AEKL"

runai submit \
  --name brain-any-aekl-v0 \
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
    python3 /project/src/python/training/train_aekl_anycontrast.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      config_file=${config_file} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      adv_start=${adv_start} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      experiment=${experiment}


seed=42
run_dir="aekl_anycontrast_v0_longer"
training_ids="/project/outputs/ids/train_anycontrast.tsv"
validation_ids="/project/outputs/ids/validation_anycontrast.tsv"
config_file="/project/configs/stage1/aekl_v0.yaml"
batch_size=1
n_epochs=50
adv_start=5
eval_freq=1
num_workers=8
experiment="AEKL"

runai submit \
  --name brain-any-aekl-v0-longer \
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
    python3 /project/src/python/training/train_aekl_anycontrast.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      config_file=${config_file} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      adv_start=${adv_start} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      experiment=${experiment}