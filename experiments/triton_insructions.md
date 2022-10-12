# My Triton Workflow

``` shell
module load py-virtualenv
python -m venv moderl-venv
```

Run experiment interactively with something like,
``` shell
srun --mem-per-cpu=500M --cpus-per-task=4 --time=0:10:00 python train.py
```

Copt wandb results from triton with,
``` shell
rsync -e "ssh" -avz scannea1@triton.aalto.fi:/home/scannea1/python-projects/moderl/experiments/wandb/* ./
```
