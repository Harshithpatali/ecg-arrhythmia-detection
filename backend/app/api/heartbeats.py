from fastapi import (
    APIRouter,
    File,
    UploadFile,
)

from typing import List

from backend.app.utils.file_manager import (
    FileManager
)

from backend.app.services.heartbeat_service import (
    HeartbeatService
)

router = APIRouter()

service = (
    HeartbeatService()
)


@router.post(
    "/heartbeats"
)
async def heartbeats(

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

        return service.get_beats(
            temp_dir
        )

    finally:

        FileManager.cleanup(
            temp_dir
        )