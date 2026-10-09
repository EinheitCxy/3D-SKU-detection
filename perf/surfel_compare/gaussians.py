"""隔离 DA3 高斯实验：完整 SH2 米制导出与基线相机尺度对齐。"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'runtime/surfel-comparison/deps'), str(ROOT / 'Depth-Anything-3/src')]
import numpy as np
import torch
from safetensors import safe_open
from scipy.spatial.transform import Rotation
from depth_anything_3.api import DepthAnything3


def poses4(e):
    out = np.tile(np.eye(4), (len(e), 1, 1))
    out[:, :3] = e[:, :3]
    return out


def sim3(src, dst):
    x, y = src - src.mean(0), dst - dst.mean(0)
    u, d, vt = np.linalg.svd(y.T @ x / len(x))
    sgn = np.eye(3)
    sgn[-1, -1] = np.linalg.det(u @ vt)
    r = u @ sgn @ vt
    scale = np.sum(d * np.diag(sgn)) / np.mean(np.sum(x*x, axis=1))
    t = dst.mean(0) - scale * r @ src.mean(0)
    residual = np.linalg.norm(scale * (src @ r.T) + t - dst, axis=1)
    return dict(scale=float(scale), rotation=r.tolist(), translation=t.tolist(), center_rmse_m=float(np.sqrt(np.mean(residual**2))), center_max_m=float(residual.max()))


def ply(path, g):
    n = len(g['means'])
    sh = g['harmonics']
    names = ['x','y','z','nx','ny','nz'] + [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(24)] + ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
    data = np.empty((n,len(names)), dtype='<f4')
    data[:,:3] = g['means']; data[:,3:6] = 0
    data[:,6:9] = sh[:,:,0]; data[:,9:33] = sh[:,:,1:].reshape(n,24)
    op = np.clip(g['opacities'].reshape(n),1e-7,1-1e-7)
    data[:,33] = np.log(op/(1-op)); data[:,34:37] = np.log(g['scales']); data[:,37:41] = g['rotations']
    with path.open('wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(n)+'\n'+''.join('property float '+k+'\n' for k in names)+'end_header\n').encode())
        data.tofile(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count',type=int,default=31)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    modelpath = ROOT/'runtime/surfel-comparison/gs-model'
    torch.manual_seed(0); torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    model=DepthAnything3.from_pretrained(str(modelpath)).cuda().eval()
    with safe_open(modelpath/'model.safetensors',framework='pt',device='cpu') as f:
        keys=[k for k in f.keys() if 'gs_head' in k]
        state=model.state_dict()
        verified=all(k in state and torch.equal(state[k].cpu(),f.get_tensor(k)) for k in keys)
    assert keys and verified, 'GS head checkpoint load mismatch'
    loaded=time.perf_counter()
    paths=[str(ROOT/f'runtime/reproduce-video1-1fps/frames/{i}.jpg') for i in range(args.count)]
    pred=model.inference(paths,infer_gs=True,export_dir=None,process_res=504,process_res_method='upper_bound_resize')
    torch.cuda.synchronize(); inferred=time.perf_counter()
    assert int(pred.is_metric)==1 and pred.scale_factor>0 and pred.gaussians is not None
    factor=float(pred.scale_factor)
    g={k:getattr(pred.gaussians,k)[0].detach().float().cpu().numpy() for k in ['means','scales','rotations','harmonics','opacities']}
    for k in ['means','scales']: g[k] *= factor
    assert all(np.isfinite(v).all() for v in g.values()) and (g['scales']>0).all()
    np.savez(args.output/'gaussians-metric.npz',**g)
    ply(args.output/'gaussians-metric.ply',g)
    np.savez(args.output/'prediction.npz',depth=pred.depth,extrinsics=pred.extrinsics,intrinsics=pred.intrinsics,processed_images=pred.processed_images,scale_factor=factor)
    report=dict(frames=args.count,process_res=504,load_seconds=loaded-started,inference_seconds=inferred-loaded,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,scale_factor=factor,metric_correction='means and scales multiplied by scale_factor; rotations and SH unchanged',gs_head_keys=len(keys),gs_head_weights_equal_checkpoint=verified,sh_degree=2,shapes={k:list(v.shape) for k,v in g.items()},finite=True,export_seconds=time.perf_counter()-inferred)
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2))
    if args.count==31:
        with np.load(ROOT/'runtime/surfel-comparison/baseline-cache.npz') as baseline:
            b=poses4(baseline['extrinsic']); e=poses4(pred.extrinsics)
            bc=np.linalg.inv(b)[:,:3,3]; ec=np.linalg.inv(e)[:,:3,3]
            align=sim3(bc,ec)
            r=np.asarray(align['rotation'])
            relative=e[:,:3,:3] @ r @ np.transpose(b[:,:3,:3],(0,2,1))
            angles=np.degrees(Rotation.from_matrix(relative).magnitude())
            align['rotation_mean_degrees']=float(angles.mean()); align['rotation_max_degrees']=float(angles.max())
            report['baseline_to_gs_sim3']=align
            report['baseline_scale_factor']=float(baseline['scale_factor'])
            report['intrinsics_abs_max']=float(np.abs(baseline['intrinsic']-pred.intrinsics).max())
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
