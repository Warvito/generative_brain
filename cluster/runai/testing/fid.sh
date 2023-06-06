seed=42
sample_dir="/project/outputs/samples_unconditioned/"
test_ids="/project/outputs/ids/test.tsv"
num_workers=4
batch_size=1

runai submit \
  --name brain-fid \
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
      python3 /project/src/python/testing/compute_fid.py \
      seed=${seed} \
      sample_dir=${sample_dir} \
      test_ids=${test_ids} \
      batch_size=${batch_size} \
      num_workers=${num_workers}
