from src.tools.change_tool import ChangeTool

T1 = "data/uploads/patch_0000_s2.tif"
T2 = "data/uploads/patch_0000_s2.tif"

result = ChangeTool().run(T1, T2)

print(result)