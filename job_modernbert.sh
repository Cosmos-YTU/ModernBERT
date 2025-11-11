#!/bin/bash
#SBATCH -A etur22 # Account name
#SBATCH --qos=acc_ehpc # Queue name
#SBATCH --partition=acc # Partition
#SBATCH --job-name=modernbert_pretrain # Job name
#SBATCH --output=modbert_%j_output.log # Output file
#SBATCH --error=modbert_%j_error.log # Error file
#SBATCH --time=72:00:00 # Maximum runtime (3 days)
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of tasks
#SBATCH --gres=gpu:4 # Number of GPUs
#SBATCH --cpus-per-task=80 # CPU cores (4 GPU x 20)
#SBATCH --mail-type=END,FAIL # Email notifications
#SBATCH --mail-user=besher.alkurdi@std.yildiz.edu.tr

# Default values
CONFIG_FILE="./yamls/modernbert/modernbert-pt-tr-bsc.yaml"
JOB_NAME="modernbert_pretrain"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--config CONFIG_FILE] [--job-name JOB_NAME]"
            echo ""
            echo "Note: GPU count, account, and other SLURM resources are set via SBATCH directives"
            echo "To change them, edit the script or use sbatch options:"
            echo "  sbatch --gres=gpu:8 $0"
            echo "  sbatch --account=other_account $0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Load modules
module purge
module load cuda/12.3
source /home/ytu/$USER/envs/bert24/bin/activate
echo $USER

# Environment variables
export HF_HUB_CACHE=/gpfs/projects/etur22/HF_CACHE/
export HF_HUB_OFFLINE=1

# Run command
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Using config: $CONFIG_FILE"
echo "Allocated GPUs: $SLURM_GPUS"

# Start nvidia-smi monitoring in background
while true; do
    nvidia-smi
    sleep 60
done &
NVIDIA_SMI_PID=$!

export CUDA_VISIBLE_DEVICES=$(echo $SLURM_JOB_GPUS | tr ',' ',')

# Run Python script with torchrun for multi-GPU
HF_HUB_CACHE=/gpfs/projects/etur22/HF_CACHE/ HF_HUB_OFFLINE=1 composer -n $SLURM_GPUS ./main.py $CONFIG_FILE

# Stop nvidia-smi monitoring
kill $NVIDIA_SMI_PID 2>/dev/null

echo "Job finished at: $(date)"