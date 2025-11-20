# Delete a prediction (DELETE /api/predictions/:id/)
# If delete annotation, change client.predictions.delete to client.annotations.delete

from label_studio_sdk import LabelStudio
import os

# ---------------- CONFIG ----------------
PREDICTION_IDS_TO_DELETE = [52026455]  # List of IDs of the predictions to delete

# ---------------- SETTINGS ----------------
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "your_api_key")
PROJECT_ID = 198563                                                 # your project ID

# ---------------- CONNECT ----------------
client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)

# Delete the predictions with the specified IDs
for prediction_id in PREDICTION_IDS_TO_DELETE:
    try:
        client.predictions.delete(id=prediction_id)
        print(f"Successfully deleted prediction with ID: {prediction_id}")
    except Exception as e:
        print(f"Failed to delete prediction with ID: {prediction_id}. Error: {e}")


# # Get prediction details (GET /api/predictions/:id/)

# import requests
# response = requests.get(
#   "https://app.heartex.com/api/predictions/51410640",
#   headers={
#     'Authorization': 'Token e6ba562ca7b0eb0869605e823bd444c4e4eae43e'
#   },
# )

# print(response.json())