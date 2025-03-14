# P6

Using [RecBole](https://github.com/RUCAIBox/RecBole)

## Environment
### Using Slurm and Singularity
1. Start a session on a compute node:
```bash
srun --mem=25G --pty bash
``` 
2. Setup environmental variables for temporary directories (<your_username> should be replaced with your username):
```bash
export USERNAME="<your_username>"
export TMPDIR="/scratch/$USERNAME/tmp"
export SINGULARITY_TMPDIR="/scratch/$USERNAME/tmp"
export SINGULARITY_CACHEDIR="/scratch/$USERNAME/cache"
export PIP_TMPDIR="/scratch/$USERNAME/tmp"
export PIP_CACHE_DIR="/scratch/$USERNAME/cache"
```
3. Create temporary directories:
```bash
mkdir -p $TMPDIR $SINGULARITY_TMPDIR $SINGULARITY_CACHEDIR $PIP_TMPDIR $PIP_CACHE_DIR
```
4. Build the Singularity container:
```bash
singularity build --fakeroot recbole.sif recbole.def
```
5. Remember to exit the session when you are done:
```bash
exit
```

## Running models
```bash
sbatch run_model.sh <dataset> <model>
```
For example:
```bash
bash run_model.sh ml-1m BPR
```