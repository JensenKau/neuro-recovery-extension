from __future__ import annotations

import boto3
import boto3.exceptions

class CloudStorage:
    ENDPOINT_URL = "http://localhost.localstack.cloud:4566"
    AWS_S3_CREDS = {
        "aws_access_key_id":"foobar",
        "aws_secret_access_key":"foobar"
    }
    
    BUCKET_NAME = "djangobucket"

    
    def __init__(self) -> None:
        self.client = boto3.client("s3", endpoint_url=self.ENDPOINT_URL, **self.AWS_S3_CREDS)
        
        try:
            self.client.head_bucket(Bucket=self.BUCKET_NAME)
        except:
            self.initialize_bucket()
            

    def initialize_bucket(self) -> None:
        self.client.create_bucket(Bucket=self.BUCKET_NAME)
        
        
    def upload_file(self, file: str, remote_path: str, remote_name: str) -> None:
        self.client.upload_file(file, self.BUCKET_NAME, f"{remote_path}{remote_name}")
    
    
    def download_file(self, file: str, remote_path: str, remote_name: str) -> None:
        self.client.download_file(self.BUCKET_NAME, f"{remote_path}{remote_name}", file)
    
    
    def remove_object(self, remote_path: str, remote_name: str) -> None:
        self.client.delete_object(
            Bucket=self.BUCKET_NAME,
            Key=f"{remote_path}{remote_name}"
        )


if __name__ == "__main__":
    storage = CloudStorage()