class Supervisor:
    TOOLS = {
        "single_image": "TOOL_A_SINGLE_IMAGE_VQA",
        "cross_modal": "TOOL_B_CROSS_MODAL_S1_S2",
        "bitemporal": "TOOL_C_BITEMPORAL_CHANGE",
    }

    def select(
        self,
        query: str,
        image_id_t2: str | None = None,
        image_id_s1: str | None = None,
        image_id_s2: str | None = None,
        bigearthnet_patch_id: str | None = None,
    ):
        q = query.lower()

        if image_id_t2 or any(
            x in q
            for x in ("change", "changed", "before", "after", "temporal")
        ):
            return (
                self.TOOLS["bitemporal"],
                "Detected bi-temporal comparison intent",
            )

        if bigearthnet_patch_id:
            return (
                self.TOOLS["cross_modal"],
                "Detected documented BigEarthNet Sentinel-1/Sentinel-2 patch input",
            )

        if (image_id_s1 and image_id_s2) or any(
            x in q
            for x in (
                "sar",
                "radar",
                "sentinel-1",
                "backscatter",
                "cross-modal",
                "optical and sar",
            )
        ):
            return (
                self.TOOLS["cross_modal"],
                "Detected Sentinel-1/Sentinel-2 cross-modal intent",
            )

        return (
            self.TOOLS["single_image"],
            "Defaulted to single-image VQA/description",
        )
