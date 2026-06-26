"""
Temporary file management.
"""

from pathlib import Path
import shutil
import tempfile


class FileManager:

    @staticmethod
    def create_temp_directory():

        return tempfile.mkdtemp()

    @staticmethod
    def save_upload_file(
        upload_file,
        directory,
    ):

        file_path = (
            Path(directory)
            / upload_file.filename
        )

        with open(
            file_path,
            "wb",
        ) as buffer:

            shutil.copyfileobj(
                upload_file.file,
                buffer,
            )

        return file_path

    @staticmethod
    def cleanup(
        directory,
    ):

        shutil.rmtree(
            directory,
            ignore_errors=True,
        )