# Running pytorch.distributed on Multiple Nodes

Key thing to know is that srun is like a super-ssh which means that when running `srun cmd` it actually does something like `ssh node cmd`

## task.slurm

```bash
#!/bin/bash

#SBATCH --partition=TrixieMain
#SBATCH --account=dt-mtp
#SBATCH --time=00:20:00
#SBATCH --job-name=pytorch.distributed
#SBATCH --comment="Helping Harry with pytorch distributed on multiple nodes."
#SBATCH --gres=gpu:4             # This is the number of GPUs reserved per node
#SBATCH --nodes=2                # This will run your job on 2 nodes
#SBATCH --ntasks-per-node=1      # This will run the `task.sh` job once on each node.
#SBATCH --cpus-per-task=6
#SBATCH --output=%x-%j.out


# USEFUL Bookmarks
# [Run PyTorch Data Parallel training on ParallelCluster](https://www.hpcworkshops.com/08-ml-on-parallelcluster/03-distributed-data-parallel.html)
# [slurm SBATCH - Multiple Nodes, Same SLURMD_NODENAME](https://stackoverflow.com/a/51356947)

readonly MASTER_ADDR_JOB=$SLURMD_NODENAME
readonly MASTER_PORT_JOB="35768"
readonly NGPUS=4  # This should match your --gres=gpu:4 value
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_IB_DISABLE=1   # It seems IB communication does not work on Trixie, so we turn it off here, at the expense of making training slower.

readonly srun='srun --output=%x-%j.%t.out' # This will print out a different output file for each node.

env

$srun bash \
   task.sh \
      $MASTER_ADDR_JOB \
      $MASTER_PORT_JOB \
      $NGPUS &

wait
```

## task.sh

This script will be executed on each node.
Note that we are activating the `conda` environment in this script so that each node/worker can have the proper environment.

```bash
#!/bin/bash

# USEFUL Bookmarks
# [Run PyTorch Data Parallel training on ParallelCluster](https://www.hpcworkshops.com/08-ml-on-parallelcluster/03-distributed-data-parallel.html)
# [slurm SBATCH - Multiple Nodes, Same SLURMD_NODENAME](https://stackoverflow.com/a/51356947)

#module load conda/3-24.9.0
#source activate molecule

source /gpfs/projects/DT/mtp/WMT20/opt/miniforge3/bin/activate
conda activate pytorch-1.7.1

readonly MASTER_ADDR_JOB=$1
readonly MASTER_PORT_JOB=$2
readonly NGPUS=$3
export NCCL_IB_DISABLE=1 # It seems IB communication does not work on Trixie, so we turn it off here, at the expense of making training slower.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

env

python \
   -m torch.distributed.launch \
      --nproc_per_node=$NGPUS \    # This will ensure that torch uses all available GPUs on the node
      --nnodes=$SLURM_NTASKS \
      --node_rank=$SLURM_NODEID \
      --master_addr=$MASTER_ADDR_JOB \
      --master_port=$MASTER_PORT_JOB \
      main.py \
         --batch_size 128 \
         --learning_rate 5e-5 &

wait
```
