export LABEL_STUDIO_HOST=https://app.heartex.com
export LABEL_STUDIO_URL=https://app.heartex.com
export LABEL_STUDIO_API_KEY="af1aa191a1c73e63603dd31c97fc8ed1a205749c"

python /app/cli.py \
 --ls-url https://app.heartex.com \
 --ls-api-key "af1aa191a1c73e63603dd31c97fc8ed1a205749c" \
 --project 198563 \
 --tasks 227374342 \
 --yolo_botsort \
 --mode "inference" \
 --model_version "UAV_RGBr" 