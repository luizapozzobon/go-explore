pip install tensorflow==1.15.2
pip install tensorflow-gpu==1.15.2
# HOROVOD_CUDA_HOME=/usr/local/cuda/bin/cuda pip install --no-cache-dir horovod==0.19.1
HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod==0.19.1
