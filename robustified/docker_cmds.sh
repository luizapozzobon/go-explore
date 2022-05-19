nvidia-docker run -ti --userns=host --shm-size 64G -v /home/luiza.pozzobon/repos/submodule/atari:/atari/ -p 9042:9042 --name luizapozzobon_atari-env luiza.pozzobon/atari-env:latest /bin/bash
