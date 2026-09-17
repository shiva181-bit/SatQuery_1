from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile, HTTPException

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/tiff",
}


async def save_upload(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            400,
            "Only JPEG, PNG and TIFF images are supported.",
        )

    image_id = str(uuid4())
    suffix = Path(file.filename or "").suffix.lower()

    path = UPLOAD_DIR / f"{image_id}{suffix}"

    path.write_bytes(await file.read())

    return {
        "image_id": path.name,
        "filename": file.filename,
        "content_type": file.content_type,
    }