# P6

Using [RecBole](https://github.com/RUCAIBox/RecBole)

## Running Recbole
```bash
# All flags can be seen using -h
python3 main.py -h
```
```bash
# Training
python3 main.py -d <dataset> -m <model>
```
```bash
# Evaluation
python3 main.py -d <dataset> -m <model> -e True
```

## Environment
### Local Linux Install
1. Install packages (requires apt-get):
```bash
bash install_local.sh
```
2. Download datasets:
```bash
cd dataset && bash dataset_download.sh && cd ..
```

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
singularity build --fakeroot recbole.sif utils/recbole.def
```
5. Remember to exit the session when you are done:
```bash
exit
```

6. Download datasets:
```bash
cd dataset && bash dataset_download.sh && cd ..
```

## Running models using Slurm
Multiple flags can be set using run_recbole, these can be see using 
```bash
bash run_recbole -h
```
### Basic setup:
```bash
bash run_recbole.sh -d <dataset> -m <model> [-n <node>] [-g <#gpus>]
```
For example:
```bash
bash run_recbole.sh -d ml-1m -m BPR
```

### Using a queue file:
```bash
bash run_recbole.sh -c <file path> [-n <node>] [-g <#gpus>]
```
