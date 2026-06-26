from fastapi import (
    APIRouter,
    File,
    UploadFile,
)

from typing import List

from app.utils.file_manager import (
    FileManager
)

from app.services.rpeak_service import (
    RPeakService
)

router = APIRouter()

service = (
    RPeakService()
)


@router.post(
    "/rpeaks"
)
async def detect_rpeaks(

    files: List[
        UploadFile
    ] = File(...)
):

    temp_dir = (
        FileManager
        .create_temp_directory()
    )

    try:

        for file in files:

            FileManager.save_upload_file(
                file,
                temp_dir,
            )

        return service.detect(
            temp_dir
        )

    finally:

        FileManager.cleanup(
            temp_dir
        )