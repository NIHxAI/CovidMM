base="/data/project/noalcohol/covid/docker_final_251112/COVID_Severity_Prediction_final/"
docker run --rm --gpus all \
  -u $(id -u):$(id -g) \
  -w /app \
  -e MODE=covid \
  -e PYTHONPATH=/app \
  -e OUTPUT_DIR=/saved_models \
  -v "$base/data":/data:rw \
  -v "$base/saved_models":/saved_models:rw \
  -v "$base/src":/app/src:ro \
  noalcohol/covid:3env-offline \
  run_mode -m src.ablation_study_622.preprocess_ehr 

docker run --rm --gpus all \
  -u $(id -u):$(id -g) \
  -w /app \
  -e MODE=covid \
  -e PYTHONPATH=/app \
  -e OUTPUT_DIR=/saved_models \
  -v "$base/data":/data:rw \
  -v "$base/saved_models":/saved_models:rw \
  -v "$base/src":/app/src:ro \
  noalcohol/covid:3env-offline \
  run_mode -m src.ablation_study_622.align_ct_cxr_paths
docker run --rm --gpus all \
  -u $(id -u):$(id -g) \
  -w /app \
  -w /data \
  -e MODE=medclip \
  -e PYTHONPATH=/app \
  -e OUTPUT_DIR=/saved_models \
  -v "$base/data":/data:rw \
  -v "$base/.cache":/.cache:rw \
  -e TMPDIR=/data/tmp -e TEMP=/data/tmp -e TMP=/data/tmp \
  -v "$base/saved_models":/saved_models:rw \
  -v "$base/src":/app/src:rw \
  noalcohol/covid:3env-offline \
  run_mode -m src.preprocess.MedicalNet.generate_ct_embeddings --input_root_dir /data/raw/CT --output_root_dir /data/embeddings/ct_embeddings2
exit
docker run --rm --gpus all \
  -u $(id -u):$(id -g) \
  -w /app \
  -w /data \
  -e MODE=medclip \
  -e PYTHONPATH=/app \
  -e OUTPUT_DIR=/saved_models \
  -v "$base/data":/data:rw \
  -v "$base/.cache":/.cache:rw \
  -e TMPDIR=/data/tmp -e TEMP=/data/tmp -e TMP=/data/tmp \
  -v "$base/saved_models":/saved_models:rw \
  -v "$base/src":/app/src:rw \
  noalcohol/covid:3env-offline \
  run_mode -m src.preprocess.MedCLIP.generate_cxr_embeddings --input_root_dir /data/raw/CXR --output_root_dir /data/embeddings/cxr_embeddings2

exit

