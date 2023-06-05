stage1_mlflow_path="/project/mlruns/607723384106105441/bc3dd686601e43509125c46b02c19025/artifacts/final_model"
diffusion_mlflow_path="/project/mlruns/275410112256848408/6a77cc1fd9254705ad30031247b98e10/artifacts/final_model"
output_dir="/project/outputs/trained_models/"

runai submit \
  --name brain-convert-model \
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
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/testing/convert_mlflow_to_pytorch.py \
      stage1_mlflow_path=${stage1_mlflow_path} \
      diffusion_mlflow_path=${diffusion_mlflow_path} \
      output_dir=${output_dir}
