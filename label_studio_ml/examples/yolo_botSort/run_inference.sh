# Load environment variables from .env if present
if [ -f .env ]; then
	export $(grep -v '^#' .env | xargs)
fi

python /app/cli.py \
 --ls-url "$LABEL_STUDIO_HOST" \
 --ls-api-key "$LABEL_STUDIO_API_KEY" \
 --project 198563 \
 --tasks 227374342 \
 --yolo_botsort \
 --mode "inference" \
 --model_version "UAV_RGBr" 