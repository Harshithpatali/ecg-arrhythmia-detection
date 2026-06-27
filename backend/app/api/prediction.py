from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
)

from typing import List
import traceback

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

        print("\n========== ECG PREDICTION START ==========")

        for file in files:

            print(f"Saving file: {file.filename}")

            FileManager.save_upload_file(

                file,

                temp_dir,

            )

        print("Files saved successfully")

        result = (

            prediction_service
            .analyze(
                temp_dir
            )

        )

        print("Prediction completed successfully")

        return (
            ResponseFormatter
            .format(
                result
            )
        )

    except Exception as e:

        print("\n========== ECG PREDICTION ERROR ==========")
        print(traceback.format_exc())
        print("==========================================\n")

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )

    finally:

        FileManager.cleanup(
            temp_dir
        )