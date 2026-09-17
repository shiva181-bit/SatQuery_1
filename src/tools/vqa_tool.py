import os
from pathlib import Path
from urllib.parse import urlparse

import requests


class VQATool:
    ID = "TOOL_A_SINGLE_IMAGE_VQA"

    def __init__(self, model=None):
        self.model = model
        self.api_url = (os.getenv("SATQUERY_TOOL_A_URL") or "").rstrip("/")

    def configuration(self) -> dict:
        parsed = urlparse(self.api_url)
        configured = bool(parsed.scheme in {"http", "https"} and parsed.netloc)
        return {"configured": configured, "endpoint_host": parsed.netloc if configured else None}

    def run(self, image_path: str | Path, question: str) -> dict:
        config = self.configuration()
        if not config["configured"]:
            return {
                "tool": self.ID, "status": "API_NOT_CONFIGURED", "question": question,
                "answer": None, "description": None,
                "message": "Set SATQUERY_TOOL_A_URL to a live inference service URL.",
            }
        try:
            with open(image_path, "rb") as image:
                response = requests.post(
                    f"{self.api_url}/inference",
                    files={"file": (Path(image_path).name, image, "image/jpeg")},
                    data={"question": question}, timeout=(10, 120),
                )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            return {
                "tool": self.ID, "status": "INFERENCE_SERVICE_UNAVAILABLE",
                "question": question, "answer": None, "description": None,
                "endpoint_host": config["endpoint_host"],
                "message": "The configured Tool A inference service could not be reached.",
                "error_type": type(exc).__name__,
            }
        except ValueError:
            return {
                "tool": self.ID, "status": "INFERENCE_SERVICE_INVALID_RESPONSE",
                "question": question, "answer": None, "description": None,
                "endpoint_host": config["endpoint_host"],
                "message": "The Tool A service did not return valid JSON.",
            }
        return {
            "tool": self.ID, "status": data.get("status", "SUCCESS"),
            "question": question, "answer": data.get("answer"),
            "description": data.get("answer"), "runtime_ms": data.get("runtime_ms"),
            "device": data.get("device"),
        }
