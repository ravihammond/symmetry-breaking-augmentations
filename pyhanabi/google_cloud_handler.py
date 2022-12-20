import os
import time

from google.cloud import storage

class GoogleCloudHandler:
    def __init__(self, project, user, gc_path, local_path):
        self._project = project
        self._user = user
        self._gc_path = gc_path
        self._local_path = local_path
        self._client = storage.Client(project=self._project)
        self._bucket = self._client.get_bucket(self._project + "-data")

    def assert_directory_doesnt_exist(self):
        path = os.path.join(self._user, self._gc_path)
        blobs = list(self._client.list_blobs(self._bucket, prefix=path))
        assert len(blobs) == 0, f"Google Cloud Error: Path {path} already exists."

    def upload_from_file_name(self, file_name, except_retry_wait=1):
        file_path = os.path.join(self._local_path, file_name)
        if not os.path.exists(file_path):
            return

        blob_name = os.path.join(self._user, self._gc_path, file_name)
        blob = self._bucket.blob(blob_name)

        while True:
            try:
                blob.upload_from_filename(file_path)
                break
            except Exception as e:
                print(e)
                time.sleep(except_retry_wait)


    def list_user_blobs(self):
        for blob in self._client.list_blobs(self._bucket):
            if self._user in str(blob.name):
                print(blob)

