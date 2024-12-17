from __future__ import annotations
from typing import BinaryIO, List

import boto3
from botocore.exceptions import ClientError
import logging


class S3Client:
    def __init__(self, bucket: str):
        self.client = boto3.client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id="root",
            aws_secret_access_key="password",
            aws_session_token=None
        )
        self.bucket = bucket
    
    
    def upload_file(self, filename: str, objectname: str) -> bool:
        try:
            self.client.upload_file(filename, self.bucket, objectname)
        except ClientError as e:
            logging.error(e)
            return False
        return True


    def download_file(self, objectname: str, filename: str) -> bool:
        try:
            self.client.download_file(self.bucket, objectname, filename)
        except ClientError as e:
            logging.error(e)
            return False
        return True


    def upload_file_obj(self, fileobject: BinaryIO, objectname: str) -> bool:
        try:
            self.client.upload_fileobj(fileobject, self.bucket, objectname)
        except ClientError as e:
            logging.error(e)
            return False
        return True


    def download_file_obj(self, objectname: str, fileobject: BinaryIO) -> bool:
        try:
            self.client.download_fileobj(self.bucket, objectname, fileobject)
        except ClientError as e:
            logging.error(e)
            return False
        return True


    def delete_file(self, objectname: str) -> bool:
        try:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=objectname
            )
        except ClientError as e:
            logging.error(e)
            return False
        return True


    def delete_folder(self, foldername: str) -> bool:
        files = self.list_all(foldername)

        try:
            for i in range(0, len(files), 1000):
                self.client.delete_objects(
                    Bucket=self.bucket, 
                    Delete={"Objects": list(map(lambda x: {"Key": x}, files[i:min(i+1000, len(files))]))}
                )
        except ClientError as e:
            logging.error(e)
            return False

        return True


    def list_files(self, prefix: str = "") -> List[str]:
        files = self.list_all(prefix)

        if files is not None:
            return list(filter(
                lambda x: "/" not in x, 
                map(lambda x: x.replace(prefix, ""), files)
            ))

        return None


    def list_folders(self, prefix: str = "") -> List[str]:
        files = self.list_all(prefix)

        if files is not None:
            return sorted(set(map(
                lambda x: x.split("/")[0] + "/",
                filter(
                    lambda x: "/" in x,
                    map(lambda x: x.replace(prefix, ""), files)
            ))))

        return None


    def list_files_folders(self, prefix: str = "") -> List[str]:
        folders = self.list_folders(prefix)
        files = self.list_files(prefix)

        if files is not None and folders is not None:
            return folders + files

        return None


    def list_all(self, prefix: str = "") -> List[str]:
        try:
            output = []
            paginator = self.client.get_paginator('list_objects_v2')
            next_token = None
            first_run = True
            pagination_config = {
                "MaxItems": 1000,
                "PageSize": 200
            }

            while first_run or next_token is not None:
                first_run = False

                if next_token is not None:
                    pagination_config["StartingToken"] = next_token
                elif "StartingToken" in pagination_config:
                    del pagination_config["StartingToken"]

                for item in paginator.paginate(Bucket=self.bucket, Prefix=prefix, PaginationConfig=pagination_config):
                    next_token = item["NextContinuationToken"] if "NextContinuationToken" in item else None

                    if item["Contents"] is not None:
                        for file in item["Contents"]:
                            output.append(file["Key"])

            return output
            
        except ClientError as e:
            logging.error(e)
            return None
    


if __name__ == "__main__":
    pass