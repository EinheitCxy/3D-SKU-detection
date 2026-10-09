"""Freeze cameras and independent image references before candidate rendering."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
root = Path(__file__).resolve().parents[2]
out = root/'modules/viewer_web/public/comparison'
out.mkdir(exist_ok=True)
z = np.load(root/'runtime/surfel-comparison/baseline-cache.npz')
ext = np.tile(np.eye(4), (31,1,1)); ext[:,:3] = z['extrinsic']
c2w = np.linalg.inv(ext)
ks = z['intrinsic'].copy(); ks[:,:2] *= 2
frames=[]
for i in range(31):
    frames.append(dict(name=f'原拍摄位置 {i}', kind='source', source=i, extrinsic=ext[i].tolist(), intrinsic=ks[i].tolist()))
    if i < 30:
        pose=np.eye(4); pose[:3,:3]=Slerp([0,1],Rotation.from_matrix(c2w[[i,i+1],:3,:3]))([.5]).as_matrix()[0]
        pose[:3,3]=(c2w[i,:3,3]+c2w[i+1,:3,3])/2
        frames.append(dict(name=f'位置 {i} 与 {i+1} 之间',kind='interpolated',source=None,extrinsic=np.linalg.inv(pose).tolist(),intrinsic=((ks[i]+ks[i+1])/2).tolist()))
for offset in np.linspace(-.10,.10,9):
    pose=c2w[15].copy();pose[:3,3]+=pose[:3,0]*offset
    frames.append(dict(name=f'位置 15 横移 {offset*100:.1f} 厘米',kind='lateral',source=None,extrinsic=np.linalg.inv(pose).tolist(),intrinsic=ks[15].tolist()))
(out/'cameras.json').write_text(json.dumps(dict(width=560,height=1008,frames=frames),ensure_ascii=False))
(out/'reference').mkdir(exist_ok=True)
for i in range(31):
    im=cv2.imread(str(root/f'runtime/reproduce-video1-1fps/frames/{i}.jpg'))
    # Match renderer pixel centers: screen x+0.5 -> processed (x+0.5)/2.
    affine=np.eye(3);affine[:2]=z['source_to_processed_affine'][i]
    yy,xx=np.mgrid[:1008,:560].astype('float32')
    inv=np.linalg.inv(affine)
    x=inv[0,0]*((xx+.5)/2)+inv[0,2];y=inv[1,1]*((yy+.5)/2)+inv[1,2]
    cv2.imwrite(str(out/f'reference/{2*i:03}.png'),cv2.remap(im,x.astype('float32'),y.astype('float32'),cv2.INTER_LINEAR))
print(f'fixed {len(frames)} cameras, 31 source references, 560x1008')
