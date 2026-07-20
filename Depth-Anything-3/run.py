import glob, os, torch
from depth_anything_3.api import DepthAnything3
import time

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)

example_path = "/home/xingyu/3D_Recognization/imdata/floor_display1/images"
images = sorted(glob.glob(os.path.join(example_path, "*.jpg")))
assert images, f"No images found under {example_path}"
a = time.time()
prediction = model.inference(
    images,
    export_dir="output",
    export_format="glb"  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
)
b = time.time()

print(f'total inference time: {a-b}')
# prediction.processed_images : [N, H, W, 3] uint8   array
print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print(prediction.depth.shape)  
# prediction.conf             : [N, H, W]    float32 array
print(prediction.conf.shape)  
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
print(prediction.intrinsics.shape)