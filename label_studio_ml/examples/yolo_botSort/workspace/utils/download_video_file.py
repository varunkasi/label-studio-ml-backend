from label_studio_sdk import LabelStudio
import os
import uuid
import requests

# ---------------- SETTINGS ----------------
LABEL_STUDIO_URL = 'https://app.heartex.com'    # your Label Studio URL
LABEL_STUDIO_API_KEY = 'e6ba562ca7b0eb0869605e823bd444c4e4eae43e'      # personal access token
PROJECT_ID = 198563                                                 # your project ID
TASK_ID = 226454004
VIDEO_TEMP_FOLDER = 'app/workspace/temp'  # folder to store temporary video files

class VideoUtils:
    def __init__(self, labelstudio_url: str = LABEL_STUDIO_URL, api_token: str = LABEL_STUDIO_API_KEY):
        """
        Initialize the VideoUtils instance.

        Args:
            labelstudio_url (str): Base URL of Label Studio instance (e.g., https://app.heartex.com)
            api_token (str): API token for authentication
        """
        self.labelstudio_url = labelstudio_url
        self.api_token = api_token
        self.client = LabelStudio(base_url=self.labelstudio_url, api_key=self.api_token)
        self.headers = {
            "Authorization": f"Token {self.api_token}",
            "User-Agent": "python-requests/2.x"  # optional, sometimes required
        }


    # ------------------------
    # Video Download Methods
    # ------------------------
    def get_video_link_and_filename_from_task_id(self, task_id: int) -> str:
        """
        Retrieve the video file URL from a Label Studio task.

        Args:
            task_id (int): Label Studio task ID

        
        Returns:            str: Video file URL
        """
        task = self.client.tasks.get(id=TASK_ID)
        video_link = task.data["video"]
        video_url = self.labelstudio_url + video_link
        filename = task.storage_filename
        return video_url, filename

    def download_video_with_task_id(self, task_id: int) -> str:
        """
        Download a video from a Label Studio task using the direct URL returned by the helper.
        Handles filenames with subfolders by creating all necessary directories.

        Args:
            task_id (int): Label Studio task ID

        Returns:
            str: Path to the downloaded video
        """
        # Get the direct-access URL and filename (may include subfolders)
        video_url, filename = self.get_video_link_and_filename_from_task_id(task_id)

        if not video_url:
            raise RuntimeError("No video URL returned from helper function")

        # Base temp folder
        os.makedirs(VIDEO_TEMP_FOLDER, exist_ok=True)

        # Unique folder to avoid collisions
        while True:
            temp_folder_name = str(uuid.uuid4())
            temp_folder_path = os.path.join(VIDEO_TEMP_FOLDER, temp_folder_name)
            if not os.path.exists(temp_folder_path):
                os.makedirs(temp_folder_path)
                break

        # Full path to save the video
        save_path = os.path.join(temp_folder_path, filename)

        # Ensure all parent directories exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Headers with API token
        headers = {"Authorization": f"Token {self.api_token}"}

        # Download the video
        response = requests.get(video_url, headers=headers, stream=True)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download video. Status code: {response.status_code}, body={response.text}")

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Video saved to:", save_path)
        return save_path

    # ------------------------
    # Video Deletion Methods
    # ------------------------
    def delete_video(self, filepath: str) -> bool:
        """
        Delete a local video file if it exists.

        Args:
            filepath (str): Path to the video file

        Returns:
            bool: True if deleted, False if file does not exist
        """
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    def delete_videos(self, filepaths: list[str]) -> dict:
        """
        Delete multiple video files.

        Args:
            filepaths (list[str]): List of file paths to delete

        Returns:
            dict: {filepath: True/False} indicating deletion success
        """
        results = {}
        for path in filepaths:
            results[path] = self.delete_video(path)
        return results



videoUtils = VideoUtils()
url = videoUtils.download_video_with_task_id(226765782)
print(url)