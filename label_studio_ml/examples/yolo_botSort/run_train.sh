
# Load environment variables from .env if present
if [ -f .env ]; then
	export $(grep -v '^#' .env | xargs)
fi

python /app/cli.py \
 --ls-url "$LABEL_STUDIO_HOST" \
 --ls-api-key "$LABEL_STUDIO_API_KEY" \
 --project 205664 \
 --tasks 232794031 \
 --yolo_botsort \
 --mode "train" \
 --model_version "UAV_RGB" \
 --classes "Person" \
 --annotation_id 80755710