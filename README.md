# SatQuery AI / ANTARDRISHTI integration notes

## Tool B: documented BigEarthNet input

The local `bifold_resnet50_12ch.pt` is integrated as the legacy
`resnet50-all-v0.1.1` checkpoint profile. It must receive these 12 channels:

`B02, B03, B04, B08, B05, B06, B07, B11, B12, B8A, VH, VV`

The supplied `BigEarthNet_20/patch_*` folders document their 14-channel source
array in `patch_info.txt`. Tool B verifies that metadata, removes only `B01`
and `B09`, reorders the remaining channels for the legacy checkpoint, and uses
fixed BigEarthNet-v2 train-split normalization. It does not infer semantic
meaning from unlabelled multi-band GeoTIFFs.

To query a bundled patch through `/query`, send:

```json
{
  "query": "What land-cover classes are likely in this Sentinel-1/Sentinel-2 patch?",
  "bigearthnet_patch_id": "patch_01"
}
```

`image_id` is only required by the image-question and temporal tools. Set
`SATQUERY_BIGEARTHNET_DIR` if the dataset is stored elsewhere.

## Tool A: live inference service

Set `SATQUERY_TOOL_A_URL` to the current public base URL of the BLIP-2 service,
without `/inference`. The previous Cloudflare hostname is no longer resolvable.
When the service cannot be reached, Tool A returns
`INFERENCE_SERVICE_UNAVAILABLE` as structured output rather than presenting the
network failure as a model answer.

## Verification

Use Python 3.9–3.12 with the project dependencies, then run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The BIFOLD checkpoint execution test is marked `integration` and runs a real
forward pass on `BigEarthNet_20/patch_01`.
