from fastapi import (
    APIRouter,
    File,
    UploadFile,
)

from typing import List

from backend.app.utils.file_manager import (
    FileManager
)

from backend.app.services.waveform_service import (
    WaveformService
)

router = APIRouter()

waveform_service = (
    WaveformService()
)


@router.post(
    "/waveform"
)
async def waveform(

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

        result = (

            waveform_service
            .get_signal(
                temp_dir
            )

        )

        return result

    finally:

        FileManager.cleanup(
            temp_dir
        )