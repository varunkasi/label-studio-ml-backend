#!/bin/bash

# Syncs weights from local directories to airlab_cloud_weights and then to cloud storage. Also sycs back from cloud to local.


# Load environment variables from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Define local and cloud paths
LOCAL_BASE="airlab_cloud_weights/yoloBotSort"
CLOUD_BUCKET="s3://vlm-chiron"

# Create necessary directories in airlab_cloud_weights
mkdir -p "$LOCAL_BASE/workspace/autotrain/train_weights"
mkdir -p "$LOCAL_BASE/workspace/autolabel/saved_weights"

# Sync local directories to airlab_cloud_weights
rsync -av --progress workspace/autotrain/train_weights "$LOCAL_BASE/workspace/autotrain/"
rsync -av --progress workspace/autolabel/saved_weights "$LOCAL_BASE/workspace/autolabel/"

# Sync airlab_cloud_weights to the cloud bucket using AWS CLI
aws s3 sync "airlab_cloud_weights" "$CLOUD_BUCKET/airlab_cloud_weights" --exact-timestamps

# Sync from the cloud bucket to the local folder
aws s3 sync "$CLOUD_BUCKET/airlab_cloud_weights" "airlab_cloud_weights" --exact-timestamps

# Print completion message
echo "Sync completed: $LOCAL_BASE -> $CLOUD_BUCKET"

# Print completion message for download
echo "Sync completed: $CLOUD_BUCKET/airlab_cloud_weights -> $LOCAL_BASE"
