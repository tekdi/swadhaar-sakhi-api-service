import os
from typing import Optional, Union
from logger import logger
from google.cloud import storage
import io

from storage.base import BaseStorageClass

class GcpBucketClass(BaseStorageClass):
    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GCP_CONFIG_PATH")
        super().__init__(storage.Client())

    def upload_to_storage(self, object_name: str, file_content: io.BytesIO) -> bool:
        """
        Uploads an in-memory file-like object to a GCS bucket.
        
        Args:
            object_name: The name for the file in the GCS bucket.
            file_content: The in-memory file buffer (io.BytesIO) containing the data.
        """
        try:
            bucket = self.client.bucket(self.bucket_name)
            object_name = f"bot_responses/{object_name}"
            blob = bucket.blob(object_name)

            # IMPORTANT: Reset the buffer's cursor to the beginning
            # After pydub writes to it, the cursor is at the end.
            file_content.seek(0) 

            blob.upload_from_file(file_content, content_type="audio/mpeg")

            logger.info(f"Uploaded in-memory data to GCS as {object_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload in-memory file to GCS: {e}", exc_info=True)
            return False

    def generate_public_url(self, object_name: str):
        try:
            bucket = self.client.get_bucket(self.bucket_name)
            object_name = f"bot_responses/{object_name}"
            blob = bucket.blob(object_name)

            # blob.acl.all().grant_read()
            public_url = blob.public_url

            return public_url,  None
        except Exception as e:
            logger.error(f"Exception Preparing public URL: {e}", exc_info=True)
            return None, "Error while generating public URL"