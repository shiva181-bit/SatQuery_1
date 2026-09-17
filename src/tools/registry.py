from .bifold_tool import BIFOLDTool
from .change_tool import ChangeTool
from .vqa_tool import VQATool


class ToolRegistry:
    def __init__(self, bifold_checkpoint: str | None = None, vqa_model=None):
        self.tools = {
            "TOOL_C_BITEMPORAL_CHANGE": ChangeTool(),
            "TOOL_A_SINGLE_IMAGE_VQA": VQATool(vqa_model),
        }
        self.unavailable = {}

        if bifold_checkpoint:
            try:
                self.tools["TOOL_B_CROSS_MODAL_S1_S2"] = BIFOLDTool(
                    bifold_checkpoint
                )
            except Exception as exc:
                self.unavailable["TOOL_B_CROSS_MODAL_S1_S2"] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }

    def get(self, tool_id: str):
        return self.tools.get(tool_id)

    def available(self):
        return list(self.tools)

    def unavailable_tools(self):
        return self.unavailable
