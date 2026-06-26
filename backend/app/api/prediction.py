from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
)

from typing import List

from backend.app.services.prediction_service import (
    BackendPredictionService
)

from backend.app.utils.file_manager import (
    FileManager
)

from backend.app.utils.response_formatter import (
    ResponseFormatter
)

router = APIRouter()

prediction_service = (
    BackendPredictionService()
)


@router.post(
    "/predict",
)
async def predict_ecg(

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

            prediction_service
            .analyze(
                temp_dir
            )

        )

        return (
            ResponseFormatter
            .format(
                result
            )
        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )

    finally:

        FileManager.cleanup(
            temp_dir
        )