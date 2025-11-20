# Load environment variables from .env if present
if [ -f .env ]; then
	export $(grep -v '^#' .env | xargs)
fi

python /app/cli.py \
 --ls-url "$LABEL_STUDIO_HOST" \
 --ls-api-key "$LABEL_STUDIO_API_KEY" \
 --project 198563 \
 --tasks 227374358 \
 --yolo_botsort \
 --mode "inference" \
 --keyframe_interval 2 \
 --model_version "UAV_RGBr" \
#  --reencode 