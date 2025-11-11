#!/bin/bash

show_usage() {
    cat << EOF
Usage: $0 --mode <relay|direct> [options]

Direct mode: Transfer directly from source to target server
Relay mode: Transfer through local computer as intermediary (source -> local -> target)

Required arguments:
  --mode <mode>           Transfer mode: 'direct' or 'relay'

Options:
  --source-host <host>    Source host (default: root@31.22.104.121)
  --source-dir <dir>      Source directory (default: /root/berturk-corpus)
  --target-user <user>    Target user@host (default: ytu248581@alogin1.bsc.es)
  --target-dir <dir>      Target directory (default: /gpfs/projects/etur22/data/berturk-corpus)
  --local-dir <dir>       Local temp directory for relay mode (default: /tmp/berturk-transfer)
  --batch-size <num>      Number of shards per batch (default: 25)
  --folders <list>        Comma-separated folders to transfer (default: train,val)
  --password <pass>       SSH password for direct mode (reads from env if not provided)

Environment variables:
  SSHPASS                 SSH password (used for direct mode)

Examples:
  # Direct transfer with password
  $0 --mode direct --password mypassword

  # Relay transfer through local computer
  $0 --mode relay --batch-size 50

  # Custom directories and folders
  $0 --mode direct --source-dir /data/mycorpus --target-dir /gpfs/projects/data/mycorpus --folders train,val,test
EOF
    exit 0
}

MODE="direct"
SOURCE_HOST="root@31.22.104.121"
SOURCE_DIR="/root/berturk-corpus"
TARGET_USER="ytu248581@alogin1.bsc.es"
TARGET_DIR="/gpfs/projects/etur22/data/berturk-corpus"
LOCAL_TEMP_DIR="/tmp/berturk-transfer"
BATCH_SIZE=25
FOLDERS=("train" "val")

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --source-host)
            SOURCE_HOST="$2"
            shift 2
            ;;
        --source-dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
        --target-user)
            TARGET_USER="$2"
            shift 2
            ;;
        --target-dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        --local-dir)
            LOCAL_TEMP_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --folders)
            IFS=',' read -ra FOLDERS <<< "$2"
            shift 2
            ;;
        --password)
            export SSHPASS="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

if [[ "$MODE" != "direct" && "$MODE" != "relay" ]]; then
    echo "Error: Mode must be 'direct' or 'relay'"
    show_usage
fi

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

if [[ "$MODE" == "direct" ]]; then
    if ! command -v sshpass &> /dev/null; then
        echo "Installing sshpass for direct mode..."
        apt-get update && apt-get install -y sshpass
    fi

    if [[ -z "$SSHPASS" ]]; then
        echo "Error: SSHPASS environment variable not set and --password not provided"
        show_usage
    fi
fi

transfer_folder_direct() {
    local folder=$1
    echo "========================================="
    echo "Starting transfer of $folder folder (direct)"
    echo "========================================="

    local total_shards=$(ssh "$SOURCE_HOST" "ls $SOURCE_DIR/$folder/shard.*.mds 2>/dev/null | wc -l")

    if [[ "$total_shards" -eq 0 ]]; then
        echo "No shards found in $folder folder, skipping..."
        return
    fi

    echo "Total shards in $folder: $total_shards"

    echo "Creating target directory: $TARGET_DIR/$folder"
    sshpass -e ssh $SSH_OPTS "$TARGET_USER" "mkdir -p $TARGET_DIR/$folder"

    for ((batch_start=0; batch_start<total_shards; batch_start+=BATCH_SIZE)); do
        local batch_end=$((batch_start + BATCH_SIZE - 1))
        if [[ $batch_end -ge $total_shards ]]; then
            batch_end=$((total_shards - 1))
        fi

        echo "Transferring batch: shards $batch_start to $batch_end"

        local files=""
        for ((i=batch_start; i<=batch_end; i++)); do
            printf -v num "%05d" $i
            files+="$SOURCE_HOST:$SOURCE_DIR/$folder/shard.$num.mds "
        done

        echo "  Transferring batch files..."
        sshpass -e rsync -az --progress -e "ssh $SSH_OPTS" $files "$TARGET_USER:$TARGET_DIR/$folder/"

        local count=$(sshpass -e ssh $SSH_OPTS "$TARGET_USER" "ls $TARGET_DIR/$folder/shard.*.mds 2>/dev/null | wc -l")
        echo "  Files at target after batch: $count/$total_shards"

        echo "  Batch completed. Waiting 1 second before next batch..."
        sleep 1
    done

    echo "  Transferring index.json for $folder..."
    if ssh "$SOURCE_HOST" "[ -f $SOURCE_DIR/$folder/index.json ]"; then
        sshpass -e scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$SOURCE_HOST:$SOURCE_DIR/$folder/index.json" "$TARGET_USER:$TARGET_DIR/$folder/"
        echo "  ✓ index.json transferred for $folder"
    else
        echo "  ✗ index.json not found for $folder"
    fi

    local final_count=$(sshpass -e ssh $SSH_OPTS "$TARGET_USER" "ls $TARGET_DIR/$folder/shard.*.mds 2>/dev/null | wc -l")
    echo "$folder transfer completed: $final_count/$total_shards files transferred"
    echo ""
}

transfer_folder_relay() {
    local folder=$1
    echo "========================================="
    echo "Starting transfer of $folder folder (relay)"
    echo "========================================="

    local total_shards=$(ssh "$SOURCE_HOST" "ls $SOURCE_DIR/$folder/shard.*.mds 2>/dev/null | wc -l")

    if [[ "$total_shards" -eq 0 ]]; then
        echo "No shards found in $folder folder, skipping..."
        return
    fi

    echo "Total shards in $folder: $total_shards"

    ssh "$TARGET_USER" "mkdir -p $TARGET_DIR/$folder"

    for ((batch_start=0; batch_start<total_shards; batch_start+=BATCH_SIZE)); do
        local batch_end=$((batch_start + BATCH_SIZE - 1))
        if [[ $batch_end -ge $total_shards ]]; then
            batch_end=$((total_shards - 1))
        fi

        echo "Transferring batch: shards $batch_start to $batch_end"

        local pattern=""
        for ((i=batch_start; i<=batch_end; i++)); do
            printf -v num "%05d" $i
            pattern+="shard.$num.mds "
        done

        echo "  Step 1/2: Downloading to local machine..."
        mkdir -p "$LOCAL_TEMP_DIR/$folder"
        ssh "$SOURCE_HOST" "cd $SOURCE_DIR/$folder && tar czf - $pattern 2>/dev/null" | tar xzf - -C "$LOCAL_TEMP_DIR/$folder/"

        echo "  Step 2/2: Uploading to target..."
        rsync -az --progress "$LOCAL_TEMP_DIR/$folder/" "$TARGET_USER:$TARGET_DIR/$folder/"

        rm -rf "$LOCAL_TEMP_DIR/$folder"/*

        local count=$(ssh "$TARGET_USER" "ls $TARGET_DIR/$folder/shard.*.mds 2>/dev/null | wc -l")
        echo "  Files at target after batch: $count/$total_shards"

        echo "  Batch completed. Waiting 2 seconds before next batch..."
        sleep 2
    done

    echo "  Transferring index.json for $folder..."
    if ssh "$SOURCE_HOST" "[ -f $SOURCE_DIR/$folder/index.json ]"; then
        scp "$SOURCE_HOST:$SOURCE_DIR/$folder/index.json" "$LOCAL_TEMP_DIR/"
        scp "$LOCAL_TEMP_DIR/index.json" "$TARGET_USER:$TARGET_DIR/$folder/"
        echo "  ✓ index.json transferred for $folder"
        rm -f "$LOCAL_TEMP_DIR/index.json"
    else
        echo "  ✗ index.json not found for $folder"
    fi

    local final_count=$(ssh "$TARGET_USER" "ls $TARGET_DIR/$folder/shard.*.mds 2>/dev/null | wc -l")
    echo "$folder transfer completed: $final_count/$total_shards files transferred"
    echo ""
}

echo "Starting dataset transfer in $MODE mode"
echo "Source: $SOURCE_HOST:$SOURCE_DIR"
echo "Target: $TARGET_USER:$TARGET_DIR"

if [[ "$MODE" == "relay" ]]; then
    mkdir -p "$LOCAL_TEMP_DIR"
    echo "Local temp directory: $LOCAL_TEMP_DIR"
fi

echo ""

if [[ "$MODE" == "direct" ]]; then
    for folder in "${FOLDERS[@]}"; do
        if ssh "$SOURCE_HOST" "[ -d $SOURCE_DIR/$folder ]"; then
            transfer_folder_direct "$folder"
        else
            echo "Warning: Source folder $SOURCE_DIR/$folder does not exist, skipping..."
        fi
    done
else
    for folder in "${FOLDERS[@]}"; do
        if ssh "$SOURCE_HOST" "[ -d $SOURCE_DIR/$folder ]"; then
            transfer_folder_relay "$folder"
        else
            echo "Warning: Source folder $SOURCE_DIR/$folder does not exist, skipping..."
        fi
    done

    echo "Cleaning up local temporary directory..."
    rm -rf "$LOCAL_TEMP_DIR"
fi

echo "========================================="
echo "All transfers completed!"
echo "========================================="

for folder in "${FOLDERS[@]}"; do
    local count=$(ssh "$TARGET_USER" "ls $TARGET_DIR/$folder/shard.*.mds 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    echo "$folder: $count files at target"
done

echo ""
echo "Index.json verification:"
for folder in "${FOLDERS[@]}"; do
    if ssh "$TARGET_USER" "[ -f $TARGET_DIR/$folder/index.json ]" 2>/dev/null; then
        echo "✓ $folder/index.json exists at target"
    else
        echo "✗ $folder/index.json missing at target"
    fi
done