import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from ..schemas.query import QueryRequest
from ..supervisor import Supervisor
from ..tools.registry import ToolRegistry
from .upload import save_upload

load_dotenv()

app = FastAPI(title="SatQuery AI", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "application": "SatQuery AI",
        "status": "online",
        "api_version": "0.6.0",
        "docs": "/docs",
        "endpoints": ["/health", "/upload", "/query"],
    }

supervisor = Supervisor()

BIFOLD_CHECKPOINT = Path(
    os.getenv(
        "SATQUERY_BIFOLD_CHECKPOINT",
        "models/bifold_resnet50_12ch.pt",
    )
)

BIGEARTHNET_DIR = Path(
    os.getenv("SATQUERY_BIGEARTHNET_DIR", "BigEarthNet_20")
)

vqa_model = None

registry = ToolRegistry(
    bifold_checkpoint=(
        str(BIFOLD_CHECKPOINT)
        if BIFOLD_CHECKPOINT.exists()
        else None
    ),
    vqa_model=vqa_model,
)
vqa_tool = registry.get("TOOL_A_SINGLE_IMAGE_VQA")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "application": "SatQuery AI",
        "version": "0.6.0",
        "available_tools": registry.available(),
        "unavailable_tools": registry.unavailable_tools(),
        "tool_a": vqa_tool.configuration(),
        "bifold_checkpoint_present": BIFOLD_CHECKPOINT.is_file(),
        "bigearthnet_dataset_present": BIGEARTHNET_DIR.is_dir(),
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    return await save_upload(file)

@app.post("/query")
def query(request: QueryRequest):
    start = time.perf_counter()

    tool_id, reason = supervisor.select(
        request.query,
        request.image_id_t2,
        request.image_id_s1,
        request.image_id_s2,
        request.bigearthnet_patch_id,
    )

    tool = registry.get(tool_id)

    trace = {
        "tool_selected": tool_id,
        "routing_reason": reason,
        "status": "RUNNING",
    }

    if tool is None:
        trace["status"] = "FAILED"
        return {
            "query_type": tool_id,
            "status": "TOOL_UNAVAILABLE",
            "execution_trace": trace,
        }

    try:
        if tool_id == "TOOL_A_SINGLE_IMAGE_VQA":
            image = _upload_file(request.image_id)
            result = tool.run(image, request.query)

        elif tool_id == "TOOL_B_CROSS_MODAL_S1_S2":
            if request.bigearthnet_patch_id:
                result = tool.run_bigearthnet_patch(
                    _bigearthnet_patch(request.bigearthnet_patch_id)
                )
            elif not request.image_id_s1 or not request.image_id_s2:
                return {
                    "query_type": tool_id,
                    "status": "MISSING_S1_S2_IMAGES",
                    "execution_trace": {
                        **trace,
                        "status": "FAILED",
                    },
                }

            else:
                result = tool.run(
                    _upload_file(request.image_id_s1),
                    _upload_file(request.image_id_s2),
                )

        elif tool_id == "TOOL_C_BITEMPORAL_CHANGE":
            if not request.image_id_t2:
                return {
                    "query_type": tool_id,
                    "status": "MISSING_SECOND_IMAGE",
                    "execution_trace": {
                        **trace,
                        "status": "FAILED",
                    },
                }

            result = tool.run(_upload_file(request.image_id), _upload_file(request.image_id_t2))

        else:
            result = {"status": "UNKNOWN_TOOL"}

    except HTTPException:
        raise
    except Exception as exc:
        trace["status"] = "FAILED"
        trace["error"] = str(exc)

        return {
            "query_type": tool_id,
            "status": "ERROR",
            "error": str(exc),
            "execution_trace": trace,
        }

    runtime = round(
        (time.perf_counter() - start) * 1000,
        2,
    )

    trace.update(
        {
            "runtime_ms": runtime,
            "active_engine": "pytorch",
            "model_versions": (
                {"tool_b": tool.CHECKPOINT_PROFILE}
                if tool_id == "TOOL_B_CROSS_MODAL_S1_S2"
                else {}
            ),
            "tool_status": result.get("status"),
            "status": "COMPLETED",
        }
    )

    return {
        "query_type": tool_id,
        "status": result.get("status"),
        "result": result,
        "execution_trace": trace,
    }


def _upload_file(image_id: str | None) -> Path:
    if not image_id:
        raise HTTPException(400, "image_id is required for this tool.")
    candidate = Path(image_id)
    if candidate.name != image_id or image_id in {"", ".", ".."}:
        raise HTTPException(400, "image_id must be an uploaded filename.")

    path = Path("data/uploads") / candidate
    if not path.is_file():
        raise HTTPException(404, f"Uploaded image not found: {image_id}")
    return path


def _bigearthnet_patch(patch_id: str) -> Path:
    candidate = Path(patch_id)
    if candidate.name != patch_id or image_id_invalid(patch_id):
        raise HTTPException(400, "bigearthnet_patch_id must be a dataset folder name.")

    path = BIGEARTHNET_DIR / candidate
    if not path.is_dir():
        raise HTTPException(404, f"BigEarthNet patch not found: {patch_id}")
    return path


def image_id_invalid(value: str) -> bool:
    return value in {"", ".", ".."}
