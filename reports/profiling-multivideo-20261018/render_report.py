"""Render the canonical profiling document from this cohort's measured JSON."""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    aggregate = json.loads((HERE / 'aggregate.json').read_text())
    cases = [json.loads((HERE / f'video{video}-summary.json').read_text()) for video in aggregate['video_ids']]
    n = len(cases)
    metrics, details = aggregate['metrics'], aggregate['detail_metrics']

    def fmt(value, digits=3):
        return '—' if value is None else f'{value:,.{digits}f}'

    def cell(key, scale=1, digits=3, detail=False):
        row = (details if detail else metrics).get(key, {})
        return fmt(row.get('mean') / scale if row.get('mean') is not None else None, digits)

    def wall(role, label, field='inclusive_s'):
        return cell(f'wall/{role}/{label}/{field}', detail=True)

    def count(label):
        return cell(f'pointflow/core/{label}/value', digits=0, detail=True)

    def table(headers, rows):
        return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---:' if i else '---' for i in range(len(headers))) + ' |',
                          *('| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows)])

    def wall_table(role, rows):
        return table(['同次运行的内部区间', '平均 wall 秒', '平均调用次数', 'n', '做什么 / 包含关系'],
                     [(title, wall(role, label), cell(f'wall/{role}/{label}/calls', digits=1, detail=True),
                       details.get(f'wall/{role}/{label}/inclusive_s',{}).get('n',0), explanation)
                      for title, label, explanation in rows])

    headings = [
        ('完整流水线', 'total_wall_s', 1, '秒'),
        ('Viewer 可用（export 返回）', 'milestones_s.viewer_export_return', 1, '秒'),
        ('ZIP 就绪（累计）', 'milestones_s.zip_ready', 1, '秒'),
        ('GLB 导出自身', 'phase_s.glb_export', 1, '秒'),
        ('任务 driver 显存采样峰', 'memory.gpu_task_driver_peak_mib', 1024, 'GiB'),
        ('最终 Viewer 点数', 'outputs.points', 1, '点'),
    ]
    headlines = table(['指标', f'平均（n={n}）', '中位数', '最小–最大', '单位'], [
        (title, fmt(metrics[key]['mean']/scale, 1 if unit=='点' else 3),
         fmt(metrics[key]['median']/scale, 1 if unit=='点' else 3),
         fmt(metrics[key]['min']/scale, 0 if unit=='点' else 3)+'–'+fmt(metrics[key]['max']/scale, 0 if unit=='点' else 3), unit)
        for title,key,scale,unit in headings])
    per_video = table(['video', '帧 / 检测', '完整秒', 'Viewer累计秒', 'ZIP累计秒', 'GLB自身秒', '任务峰GiB', 'Viewer点'], [
        (case['video_id'], f"{case['input']['frames']} / {case['input']['objects']}", fmt(case['total_wall_s']),
         fmt(case['milestones_s']['viewer_export_return']), fmt(case['milestones_s']['zip_ready']),
         fmt(case['spans']['glb_export']['total_s']), fmt(case['memory']['gpu_task_driver_peak_mib']/1024), fmt(case['outputs']['points'],0))
        for case in cases])
    phase_titles = [
        ('core imports + 仪器安装', 'core_imports_and_instrumentation'), ('DA3重建', 'run_reconstruction'),
        ('PVS + matching', 'run_sku_matching'), ('Dedup', 'run_dedup_sequence'),
        ('最终Viewer导出', 'export_web_viewer_bundle'), ('delivery imports', 'delivery_imports'),
        ('delivery仪器安装', 'delivery_detail_setup'), ('ZIP打包', 'zip_pack'), ('GLB导出', 'glb_export'),
        ('进程启动/退出及其余间隙', 'process_setup_teardown_and_other_gaps')]
    phase_rows = []
    for title, name in phase_titles:
        prefix = 'prefix_s.'
        phase_rows.append((title, cell(prefix+'full_pipeline.'+name), cell(prefix+'viewer_export_return.'+name),
                           cell(prefix+'zip_ready.'+name)))
    phase_table = table(['互斥阶段', '完整平均秒', 'Viewer前缀贡献秒', 'ZIP前缀贡献秒'], phase_rows)
    da3 = wall_table('da3', [
        ('导入与AST仪器安装','da3.imports_and_instrumentation','在重建子进程内；含依赖导入和插桩准备'),
        ('JPEG读取 / RGB','src/da3_runner.py:main:L220','逐全部F帧；PIL decode/convert'),
        ('from_pretrained CPU总段','da3.load_cpu','父区间，含构造与权重加载'),
        ('↳ 网络构造','da3.network_constructor','创建Nested两分支和head，含默认参数初始化'),
        ('↳ safetensors load_model','da3.safetensors_load_model','含下面load_file和assignment'),
        ('　↳ load_file','da3.safetensors_load_file','打开/mmap；不等于全部物理磁盘读取'),
        ('　↳ state assignment','da3.state_assignment','参数赋值及可能的页触发'),
        ('模型to(cuda)','da3.model_to_device','原生调用wall；没有额外同步'),
        ('inference API','da3.inference_api','父区间，含预处理、forward、输出整理'),
        ('↳ preprocess','da3.preprocess','resize/normalize/stack'),
        ('↳ 输入整理/H2D提交','da3.input_enqueue','主机调用，不称completed-copy CUDA时间'),
        ('↳ forward原生同步区间','da3.forward_native_sync','保留算法原有前后synchronize'),
        ('↳ Prediction / D2H','da3.prediction_D2H','转CPU Prediction数组'),
        ('↳ processed RGB','da3.processed_RGB','NHWC、反归一化、clip、uint8'),
        ('CPU反投影','da3.CPU_backprojection','逐F帧K^-1 p d，再C2W'),
        ('NPZ整条保存with语句','src/da3_runner.py:main:L295','包含参数astype等求值及函数内保存'),
        ('↳ np.savez函数内','da3.NPZ_write_excluding_argument_evaluation','未压缩序列化/写入；无fsync承诺'),
    ])
    matching = table(['Matching计时区间', '平均秒', '平均次数', '含义'], [(title, cell('matching.'+key+'.total'), cell('matching.'+key+'.calls', digits=1), text)
        for title,key,text in [
            ('SAM masks','sam3_mask','父区间，含首次加载、predict及cache包装'),
            ('数据读取','load_data','每ref输入'), ('场景Tensor构造','scene_data_build','首ref场景准备'),
            ('transform构造','build_transforms','每ref camera transforms'),
            ('参考mask点采样','ref_point_sampling','每检测最多70点'), ('world→camera','world_to_camera','含设备等待'),
            ('3D→2D投影','projection_3d_to_2d','投影到next帧'), ('投影后处理','projection_postprocess','有效域等'),
            ('目标bbox匹配','target_bbox_match','候选采样、SVD/进深和评分'), ('唯一性','uniqueness_constraint','每对约束'),
            ('结果后处理','post_process','匹配结果结构'),
            ('process_images父段','process_images','嵌套父区间，不能再加到以上子项'),
        ]])
    pvs = wall_table('core', [
        ('首次模型加载','pvs.model_load','一次/视频；与NN/SOR准备线程重叠'),
        ('↳ model_builder导入','utils/sam31_pvs.py:load_model:L19','动态导入'),
        ('↳ CPU模型构造','utils/sam31_pvs.py:load_model:L23','位于cuda.device作用域内；不等于读取权重'),
        ('checkpoint mmap','utils/sam31_pvs.py:load_model:L31','weights_only / mmap'),
        ('严格state assignment','utils/sam31_pvs.py:load_model:L48','只保留tracker和共享backbone权重'),
        ('eval/to(cuda)','utils/sam31_pvs.py:load_model:L49','实际设备搬运'),
        ('全部predict API','pvs.predict','每图一次encoder，box batch=4'),
        ('图像transform/H2D语句','utils/sam31_pvs.py:predict:L77','1008×1008，归一化后提交GPU'),
        ('box Tensor构造','utils/sam31_pvs.py:predict:L78','缩放语句L79另有AST记录'),
        ('encoder主机调用','pvs.encoder.host','enqueue wall，不能视作GPU完成时间'),
        ('decoder主机调用','pvs.decoder.host','ceil(boxes/4)批，不同视频批数不同'),
        ('logits上采样','utils/sam31_pvs.py:predict:L111','bilinear到源输入图大小'),
        ('finite检测','utils/sam31_pvs.py:predict:L115','可能引发GPU→CPU依赖'),
        ('mask阈值/D2H','utils/sam31_pvs.py:predict:L117','CPU numpy'),
        ('按框裁剪mask','utils/sam31_pvs.py:predict:L122','逐box'),
        ('scores D2H','utils/sam31_pvs.py:predict:L123','CPU list'),
        ('presence D2H','utils/sam31_pvs.py:predict:L124','CPU list'),
        ('最终同步','utils/sam31_pvs.py:predict:L125','保留原算法边界'),
        ('stack/返回','utils/sam31_pvs.py:predict:L126','含np.stack；native总量包含encoder'),
    ])
    prepare = wall_table('core', [
        ('几何准备总段','prepare_web_viewer_geometry','后台线程；不能加到matching主表'),
        ('cache读取/校验','viewer._load_da3_cache','全native grid'),
        ('U/V与depth','surfel.grid_tangents','相邻grid差分与连续性'),
        ('raw filter准备','filter.prepare_scene_filter','snapshot/cloud/NN/median/SOR raw mask'),
        ('↳ NN原生','filter.NN_native','建索引、每点最近邻；包含在prepare中'),
        ('↳ SOR原生','filter.SOR_native','k=20/std=2；尚未套用商品保护'),
        ('源照片/相机纹理','surfel.prepare_surfel_textures','JPEG与affine/camera metadata'),
    ])
    viewer = wall_table('core', [
        ('最终Viewer总段','export_web_viewer_bundle','当前可用里程碑的最后主阶段'),
        ('thumbnails','viewer._generate_thumbnails','逐observation，Viewer必要缩略图'),
        ('显示方向','viewer.get_scene_orientation','与ground删除不同的RANSAC拟合'),
        ('↳ 方向NN','orientation.NN_native','最多200K候选域'),
        ('↳ 方向RANSAC','orientation.RANSAC_native','候选方向平面'),
        ('选点总段','viewer._sample_points','包含以下标签/过滤/预算等'),
        ('↳ mask→商品label','sampling.labels','canonical processed PVS mask'),
        ('↳ 最终几何过滤','filter.filter_scene_points','raw NN/SOR复用，再保护/DBSCAN/ground'),
        ('　↳ SOR应用整段','filter.SOR_apply_block','商品保护、20% guard、select；不含提前SOR原生'),
        ('　↳ DBSCAN整段','filter.DBSCAN_block','native+簇阈值+保护+select'),
        ('　　↳ DBSCAN原生','filter.DBSCAN_native','eps=5×medianNN，min_points=10'),
        ('　↳ ground整段','filter.ground_block','RANSAC+双侧判断+保护+select'),
        ('　　↳ ground RANSAC','filter.RANSAC_native','当前点域，最多1000假设，可能提前停止'),
        ('↳ full-grid特征','sampling.features','depth/valid/语义region生成edge、halo、weights、caps'),
        ('↳ 背景voxel','sampling.postfilter_voxels','5mm，置信度优先'),
        ('↳ 商品/背景预算','sampling.budget','分ID预算与adaptive scale/selection'),
        ('　↳ adaptive.select','adaptive.select','包含于预算分组；勿重复相加'),
        ('↳ 选点数组序列化','sampling.serialize_arrays','稳定索引及紧凑数组'),
        ('Surfel打包','surfel.prepare_surfels','U/V/scale/depth与纹理metadata'),
        ('point ranges','viewer._attach_point_index_ranges','每observation关联选点半开区间'),
        ('generation发布','viewer._publish_bundle','写binary/JSON/JPEG及CURRENT'),
    ])
    dedup = wall_table('core', [(title,label,text) for title,label,text in [
        ('Dedup总段','run_dedup_sequence','全部F帧'), ('parse summaries','dedup.parse_all_matches','匹配边'),
        ('union-find/GID','dedup.build_global_mapping','同图冲突不合并'), ('逐帧global ID','dedup.add_global_id_to_jsons','分类聚合'),
        ('global pair发布','dedup._publish_global_pair','原子发布相关JSON')]])
    zip_table = table(['成员类别', '平均成员数', '平均原始MiB', '平均归档payload MiB', '平均成员write秒'], [
        (name, cell('zip.'+key+'.calls',digits=1), cell('zip.'+key+'.input_bytes',1024**2),
         cell('zip.'+key+'.compressed_bytes',1024**2), cell('zip.'+key+'.member_wall_s'))
        for name,key in [('binary / DEFLATE level1','binary'), ('JPEG / STORED','jpeg'), ('JSON / STORED','json')]])
    glb = wall_table('delivery', [
        ('GLB总段','glb.total','同一次实际导出，默认encoder_workers=1'),
        ('输入_load','glb._load','generation文件读取与校验'),
        ('保守2×2 LOD','lod.select_surfel_lod','商品block满足法向/depth/coverage才合并'),
        ('↳ coverage','lod._covered','每批最多8192组，已含于LOD'),
        ('全部bake batch','bake._bake_batch','按frame，每批最多1024候选'),
        ('↳ 中心/相机可见性','bake.block.camera_visibility','七个连续AST块之一'),
        ('↳ 初始footprint/连续性','bake.block.footprint_and_continuity','含_bound_footprint'),
        ('↳ radius/最终直径','bake.block.radius_final_diameter','保持长边≤6×native最大间距'),
        ('　↳ 最终长边限制','bake._limit_final_diameter','内层函数，不再加到七块'),
        ('↳ 8×8 texel投影','bake.block.texel_projection_mask','world→camera、圆盘/边界mask'),
        ('↳ depth/source mask','bake.block.depth_and_source_mask','遮挡、来源/面积与有效tile'),
        ('↳ source/RGB采样','bake.block.bilinear_rgb','source→texture坐标及双线性采样'),
        ('　↳ bilinear函数','bake._bilinear','包含于上行'),
        ('↳ tile/corners/延展','bake.block.tile_corners_dilation','包含RGB延展'),
        ('　↳ RGB延展','bake._dilate_rgb','固定四方向、旧轮known，alpha不变'),
        ('atlas组页','bake._page','8×8内部+一圈gutter；最多41616 tiles/页'),
        ('几何量化/写BIN','compact.geometry','u16 position/UV/index，含bufferView'),
        ('atlas编码总段','texture.encode_atlas','包含alpha、palette、PNG'),
        ('↳ alpha检查','texture._binary_alpha','二值alpha'),
        ('↳ compact RGBA','texture._compact_rgba','含MEDIANCUT和PNG子项'),
        ('　↳ palette量化','PIL.quantize','128色，无dither'),
        ('　↳ PNG save','PIL.save','optimize=True；PIL累计按contexts可再分'),
        ('图片写BIN','compact.image','编码字节，不重解码'),
        ('最终GLB写入','compact._write_glb','JSON/BIN chunk与header，文件写入返回'),
    ])
    point_rows = [
        ('dense native grid','dense_native'), ('export-valid','valid_geometry'), ('原SAM商品','product_before_filters'),
        ('原背景','background_before_filters'), ('SOR raw保留','SOR_prepared_raw_keep'),
        ('SOR商品救回','SOR_protected_rescued'), ('SOR应用后','SOR_after_apply'),
        ('DBSCAN raw保留','DBSCAN_raw_keep'), ('DBSCAN商品救回','DBSCAN_protected_rescued'), ('DBSCAN应用后','DBSCAN_after_apply'),
        ('ground RANSAC输入','ground_RANSAC_input'), ('ground RANSAC内点','ground_RANSAC_inliers'),
        ('ground raw保留','ground_raw_keep'), ('ground商品救回','ground_protected_rescued'), ('ground应用后（仅调用apply时）','ground_after_apply'),
        ('最终filter后','after_all_filters'), ('filter后商品','product_after_filters'), ('filter后背景','background_after_filters'),
        ('背景voxel候选','background_voxel_candidates'), ('商品选入','product_selected'), ('背景选入','background_selected'), ('Viewer选入总量','selected_total')]
    point_table = table(['点数域', '跨视频平均点数', *('video'+str(case['video_id']) for case in cases)], [
        (title, count(key)+f" (n={details.get(f'pointflow/core/{key}/value',{}).get('n',0)})", *(fmt(details.get(f'pointflow/core/{key}/value', {}).get('per_video',{}).get(str(case['video_id'])),0) for case in cases))
        for title,key in point_rows])
    glb_counts = table(['输出元素', '平均', '最小–最大'], [(title, cell('glb_counts.'+key,digits=0),
        fmt(metrics['glb_counts.'+key]['min'],0)+'–'+fmt(metrics['glb_counts.'+key]['max'],0)) for title,key in [
            ('GLB LOD选入点','selected_point_count'), ('LOD移除点','lod_removed_point_count'),
            ('实际baked Surfel','baked_surfel_count'), ('无有效tile跳过','skipped_surfel_count'),
            ('三角形','triangle_count'), ('atlas页','atlas_pages')]])
    glb_videos = table(['video','LOD选入','实际baked','atlas页','bake批次','GLB MiB','GLB秒'], [
        (case['video_id'],fmt(case['delivery_detail']['counts']['selected_point_count'],0),
         fmt(case['delivery_detail']['counts']['baked_surfel_count'],0),case['delivery_detail']['counts']['atlas_pages'],
         case['delivery_detail']['stats']['bake._bake_batch']['calls'],fmt(case['outputs']['glb_bytes']/1024**2),fmt(case['spans']['glb_export']['total_s']))
        for case in cases])
    resource_table = table(['阶段', '任务driver阶段峰均值GiB', '最大一条video阶段峰GiB'], [
        (title, cell('phase_gpu_peak_mib.'+key,1024), fmt(metrics.get('phase_gpu_peak_mib.'+key,{}).get('max')/1024 if metrics.get('phase_gpu_peak_mib.'+key,{}).get('max') is not None else None))
        for title,key in [('DA3重建','run_reconstruction'), ('PVS模型加载','pvs_model_load'), ('全部PVS predict中的最高采样','pvs_predict'),
                          ('matching','run_sku_matching'), ('几何准备（与matching重叠）','prepare_web_viewer_geometry'),
                          ('最终Viewer','export_web_viewer_bundle'), ('GLB','glb_export')]])
    allocator = table(['进程 / 边界', '参数GiB', 'buffer GiB', '当时allocated GiB', '当时reserved GiB', '当时未用reservation GiB'], [
        (title, *(cell(f'allocator_snapshot/{role}/{label}/sample_mean_'+field,1024**3,detail=True) for field in
                 ['gpu_parameter_bytes','gpu_buffer_bytes','allocated_bytes','reserved_bytes','unused_reserved_bytes']))
        for title,role,label in [('DA3 model ready','da3','da3.model_ready'), ('DA3 forward之前','da3','da3.forward.before'),
                                ('DA3 forward之后','da3','da3.forward.after'), ('PVS model ready','core','pvs.model_ready'),
                                ('PVS predict结束边界的均值','core','pvs.predict_end')]])
    lifetime = table(['进程', 'allocated峰均值GiB', 'reserved峰均值GiB', 'RSS HWM均值GiB', 'user CPU秒均值', 'system CPU秒均值'], [
        (role, cell(f'process.{role}.cuda_allocated_peak_bytes',1024**3), cell(f'process.{role}.cuda_reserved_peak_bytes',1024**3),
         cell(f'process.{role}.rss_hwm_kib',1024**2), cell(f'process.{role}.cpu_user_s'), cell(f'process.{role}.cpu_system_s'))
        for role in ('da3','core','delivery')])
    cuda_rows = []
    for key,row in sorted(details.items()):
        if row['section']=='cuda_events' and row['field']=='event_s':
            cuda_rows.append((row['process_role'], '`'+row['label'].replace('|','\\|')+'`', fmt(row['mean']), row['n']))
    cuda = table(['进程', 'CUDA event范围（保留父上下文）', '平均秒', 'n'], cuda_rows)
    prep_rows = [(key, cell('input.timings_s.'+key)) for key in sorted({key for case in cases for key in case['input']['timings_s']})]
    preprocessing = table(['视频输入准备区间', '平均秒'], prep_rows+[('完整输入准备wall（父区间）', cell('input.preprocessing_wall_s'))])
    checkpoint = '、'.join('video'+str(case['video_id']) for case in cases)
    point_probe = ''
    if (HERE / 'point-range-probe.json').is_file():
        probe = json.loads((HERE / 'point-range-probe.json').read_text())
        point_probe = f'''完整视频结束后另做一次有界CPU合成probe：250万int32有序labels、200个query值，Python int逐个left/right共{probe['timings_s']['python_int']:.6f}秒，同dtype逐标量{probe['timings_s']['typed_scalar']:.6f}秒，typed vector两次整批查询{probe['timings_s']['typed_vector']:.6f}秒，全部range端点精确一致。它没有使用实际生产labels，也没有修改生产代码；不计入六video均值，不支持直接从13.950秒扣除差额。[probe原始结果](../reports/profiling-multivideo-20261018/point-range-probe.json)。

生产labels为int32，而enumerate产生Python int。[NumPy1.26.4搜索源码](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/multiarray/item_selection.c#L1964)先求共同dtype再准备整个haystack；[Python整数dtype发现](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/multiarray/abstractdtypes.c)与[dtype提升](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/multiarray/array_coercion.c#L643)支持本机64位long导致重复int32→int64临时转换的解释。这是源码机制与合成现象一致的推断，未直接测生产转换次数。下一步值得先在真实labels上验证typed vector精确替换、空range/缺label边界，再跑完整交付对照。'''
    text = f'''# 最新 profiling：GPU2 / 多视频 / 2026-10-18

本文件是项目唯一当前 profiling 入口。当前本地 `main`（`bf9bde8`），本轮成功 **{n} 个不同视频**：{checkpoint}，共{sum(case['input']['frames'] for case in cases)}帧/{sum(case['input']['objects'] for case in cases)}检测。完整流水线平均 **{cell('total_wall_s')} 秒**，Viewer 平均 **{cell('milestones_s.viewer_export_return')} 秒**可用，ZIP 平均 **{cell('milestones_s.zip_ready')} 秒**就绪。全部内部数字来自各视频**同一次完整运行**，不再把独立组件重放的时间填到总表里。生产代码、默认参数和服务未修改。

## 1. 六项指标的平均值与逐视频结果

{headlines}

{per_video}

均值为每个成功视频等权算术平均，非按帧或检测数加权。每视频只跑一次，是不同场景的横截面，不是同视频重复或热服务延迟分布；不提供P95。任务峰均值与最大video任务峰不同，显存峰不能按时间相加。平均点数只是规模统计，不能实际生成一份“平均点云”。

本轮 requested=7（有JSON的1–7），attempted={aggregate['attempted']}，success={aggregate['succeeded']}；失败：`{json.dumps(aggregate['failed'],ensure_ascii=False)}`；未运行：`{json.dumps(aggregate['pending'],ensure_ascii=False)}`。video8缺同名检测JSON，排除同协议集合。video2有12帧，可能1.5s前段或2s全程，原始时间戳缺失、尚未确认，不混入均值。

采样协议：video3沿用既有0起1.5s recipe；1/4/5/6/7按JSON帧数吻合的0起1.5s建立显式实验recipe，**没有验证原检测与画面完全对齐**。因此这些是该recipe下的性能结果，不作匹配准确率或画质结论。全部JSON帧保留，没有为通过测试截断长视频；输入依据、源frame index和预处理时间见每case input.json及[输入说明](../reports/profiling-multivideo-20261018/inputs.md)。

| 环境与范围 | 本次条件 |
|---|---|
| 代码 | 本地main `bf9bde8`，生产源码/默认参数未改；原有dirty工作保留 |
| GPU | 物理GPU2，RTX4090D，24564MiB，UUID `GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7`；进程内逻辑cuda:0 |
| 共享负载 | 既有两compute进程约5030MiB保留；主机CPU共享，未设置affinity |
| 模型 | DA3NESTED-GIANT-LARGE-1.1，同snapshot `b2359bdf726fb44ef62acca04d629dcf158053e7`；SAM3.1 PVS |
| 软件 | Python3.11.11，Torch2.7.1+cu126，NumPy1.26.4，Open3D0.19.0，Pillow12.3.0；复用同test-env，未安装/升级 |
| CPU | 40逻辑/20物理核；OMP/OpenBLAS/MKL环境=4，Torch interop默认未强制修改 |
| 输入 | 1080×1920 JPEG quality95、原JSON canonical分类；504×280 DA3 grid；全部F帧一次推理 |
| 主计时起止 | 启动core进程之前的单调时钟 → delivery退出被观察到；含导入/进程生命周期/必要文件写入/轮询间隙 |
| 输出截止 | 本地Viewer generation/CURRENT、ZIP和GLB；写入返回，无fsync承诺，无浏览器首屏承诺 |

根失效`.venv`和旧HF DA3链接未改。使用现成独立uv环境、复用上一轮从本地项目镜像恢复的同模型snapshot；本轮没有下载/另复制权重或启动模型服务。

[均值JSON](../reports/profiling-multivideo-20261018/aggregate.json) · [所有内部计时表](../reports/profiling-multivideo-20261018/INTERNALS.md) · [全部内部均值CSV](../reports/profiling-multivideo-20261018/detail-averages.csv) · [逐视频全部指标CSV](../reports/profiling-multivideo-20261018/per-video-metrics.csv) · [逐reference表](../reports/profiling-multivideo-20261018/references.csv) · [复现与逐视频JSON](../reports/profiling-multivideo-20261018/README.md)

## 2. 完整流水线、Viewer和ZIP：同一时间线的三种前缀

{phase_table}

每列可相加，各列是同一工作量的不同截止点，**不能将三列再相加**。Viewer/ZIP是从主起点起的累计时间；GLB是自身耗时。Viewer就绪以后仍有core退出、delivery导入/启动和ZIP打包，因此ZIP累计值不是Viewer加GLB。精确单调时钟与各milestone前缀闭合误差保留在逐视频JSON。

准备线程平均 {cell('preparation.seconds')} 秒，其中与matching交集 {cell('preparation.matching_overlap_s')} 秒，matching之外 {cell('preparation.seconds_outside_matching')} 秒。它是并发支线，未额外加到上表。重叠区间不能等价为净节省；NN/SOR绑定的GIL机制与PVS构造膨胀见第11节。

## 3. DA3内部：构造、权重、推理、反投影、保存

DA3父阶段平均 {cell('phase_s.run_reconstruction')} 秒，含子进程导入、runner、父进程必要检查和退出。下表子项有明确包含关系，不能把父项和子项相加。

{da3}

一次输入全部F帧；processed grid为 `[F,504,280]`，不是逐帧独立重建。Nested顺序跑anyview和metric分支，再做metric scaling/depth alignment/sky处理。构造时间含参数默认初始化，不能叫作全部“读权重耗时”。source图像读取、NPZ参数astype、父子进程间隙均有各自范围，不用np.load打开archive的时间代表完整解包。

## 4. Matching与PVS：逐ref、逐框batch

`pairing_3d=next` 环形，共F个reference；每ref只配下一帧，最后回第0帧。全局RNG关键区串行，首次PVS模型构造发生在本次请求内。box batch=4，不同视频帧数/框数不同。

{matching}

{pvs}

PVS native encoder平均 {cell('pvs_native.encoder_s')} 秒；native inference_and_postprocess平均 {cell('pvs_native.inference_and_postprocess_including_encoder_s')} 秒，后者包含encoder。`mask_postprocess`原有空span不作为有效工作；`cache_npz_load`只覆盖archive打开，不代表完整NPZ读取。所有ref的具体wall、boxes、首次模型加载与native区间在CSV中，不能把某一个后续ref的耗时当作整次热请求。

## 5. 几何准备、最终Viewer、Dedup

{prepare}

{viewer}

{dedup}

NN用全有效点的最近邻距离中位数设置随尺度变化的阈值；SOR检查每点20个近邻的平均距离，std_ratio=2，raw布尔结果提前计算。最终label到达后用商品点保护修复raw拒绝，并执行20%最小保留guard。DBSCAN在SOR应用后的点域做半径邻域和连通扩展；ground在当前点域做RANSAC、双侧判断、保护与guard。显示方向另用最多200K候选拟合，不能与ground计时混用。

Python计时只能分开native NN/SOR/DBSCAN/RANSAC整体。其内部KD-tree建树、邻域查询、DBSCAN扩展、RANSAC实际迭代数**本轮没有逐段测量**；没有用复杂度推导冒充实测秒数。[原生循环说明](viewer_export_algorithm.md#61-open3d-0190-原生循环)保留算法细节。

## 6. 最终点数的完整流向

{point_table}

商品保护在几何过滤时生效；最终商品预算仍为2M、背景预算500K，商品超过预算时也会采样。raw保留+商品救回不一定等于应用后：20% guard若触发会跳过该过滤，guard记录在完整CSV里。没有以cluster的非负标签冒充SAM商品标签。

ground没有进入apply时，该行记为缺失，均值按实际n；这不等于RANSAC未计算，也不等于输出0点。具体例子：video6的RANSAC为352892/5933195≈5.947%内点，未达到8%条件，所以没有应用删除；最终filter仍保留5,933,195点。

{glb_counts}

Viewer选点与GLB LOD/bake是不同表示，GLB不回写Viewer点集。完整depth grid仍用于连续性、遮挡和来源判定。选点相同或总数相同都不能证明来源/背景/方向或画面相同。Open3D RANSAC流程没有显式固定随机seed，本轮逐阶段计数显示实际发生了什么，不以旧独立诊断补写。

## 7. ZIP内部：成员读入、压缩、写入

ZIP自身平均 {cell('phase_s.zip_pack')} 秒；累计就绪平均 {cell('milestones_s.zip_ready')} 秒，前缀组成见第2节。

{zip_table}

成员write wall包含文件读入、压缩与归档写入，不是纯zlib。binary为8个文件、DEFLATE level1；JPEG/JSON为STORED。上表采用成员wrapper范围，含每成员metadata/统计小开销；INTERNALS的zip.member.*是更内层的write span，两者不可当同一口径。类别加总不覆盖成员外的枚举、归档metadata和关闭。最终ZIP包含必要Viewer数据、源纹理与缩略图，不含GLB/模型或本轮没有提供的masterdata。

## 8. GLB内部：LOD、七块bake、RGB、atlas、最终文件

{glb_videos}

{glb}

生产report另外提供两个调度区间：bake/load/LOD侧平均 {cell('glb_timing.bake_load_lod_wall_s')} 秒，compact/encode/write侧平均 {cell('glb_timing.compact_encode_write_join_wall_s')} 秒；后者包含串行encode调用（encoding_join）平均 {cell('glb_timing.encoding_join_s')} 秒和main_thread_write平均 {cell('glb_timing.main_thread_write_s')} 秒。生产迭代器的计时归属与函数span不同，不能与上表再合并求和。

实际循环：frame → 每1024候选bake → 有效10×10 RGBA tile → 每41616个tile组atlas → 编码 → geometry/image/material顺序写BIN → 最终GLB。7个bake块连续覆盖函数主体，内层bilinear/RGB/长边函数已在所属块内。MEDIANCUT与PNG同样是encode子项。`self_s`仅减已测直接子区间，仍含其余操作和仪器成本，不能叫“纯Python时间”。

当前默认 `encoder_workers=1`，本轮保持1；已有2/4worker功能没有自动开启。保留原始128色、PNG optimize、LOD和清理，尤其保持最终长直径≤6×native最大间距。没有靠放宽长边、删DBSCAN或降纹理获得本轮数字。

## 9. 显存与RAM：阶段峰、模型参数、当前占用、allocator峰分开

{resource_table}

{allocator}

{lifetime}

表中每video先取同一时点任务PID的driver和，再取该阶段最大；跨video平均的是这些峰。predict行先取每video全部predict里的最大，不按ref数加权。GPU已有resident约5030MiB排除在任务峰外；整卡峰平均 {cell('memory.gpu_device_peak_mib',1024)} GiB。200ms采样会漏短峰，极短阶段可能无采样，按缺失处理。

阶段内有完整compute-app查询、且无本任务GPU进程的时点记为观测0（例如CPU的GLB阶段）；没有查询覆盖的短段才记缺失。进程生命周期allocator未初始化则保留缺失，不用driver的0替代。

allocated是PyTorch活跃分配，reserved是allocator保留；同一snapshot的reserved−allocated是未用reservation，不含driver/context。参数/buffer按实际device/numel/element_size计算。snapshot剩余allocated还含场景Tensor、激活和其他数据，不称KV cache；forward后也不是最大激活的时点。生命周期峰、阶段driver峰、当时快照和各进程峰互不相加。

同时进程树RSS采样和峰的跨video均值为 {cell('memory.host_tree_peak_bytes',1024**3)} GiB；共享页可能重复计。单进程HWM和CPU user/system分别列出，CPU秒可超过wall。时钟用紧邻采集的epoch/monotonic锚点映射GPU时间戳，并裁剪到每case主窗口；video3锚点在本次运行中补采、未改变其定时或进程。

## 10. CUDA events与视频输入准备（均不再加到主表）

{cuda}

这些是同一运行内CUDA stream elapsed，保留嵌套父上下文；包含CPU enqueue间隙的可能性，不等于exclusive kernel和。不增加逐层synchronize，只在算法原有同步后读取events。host wall与CUDA event不能相减得到“纯CPU”。

{preprocessing}

输入准备使用视频顺序grab、选帧retrieve、JPEG quality95及现有canonical分类适配。视频抽帧/检测JSON适配在主计时外，单独如上；detector和classifier均没有推理，沿用JSON。若要原视频至全部产物的本地耗时，可按每video加其preprocessing wall，但这仍不包含原检测/分类/HTTP/COS/下载/浏览器。

## 11. 解释边界、为什么耗时、下一步

大点域上的DBSCAN/ground、PVS CPU构造及Surfel逐texel颜色/透明延展、palette/PNG是不同工作。F决定全grid规模，检测数还决定decoder批次、缩略图和point ranges；Viewer预算封顶不会让此前全grid过滤消失。GLB LOD减少候选，但每个候选仍可能处理64texels，编码还要遍历整atlas透明区，所以不能按最终GLB字节数推断导出应很快。

具体例子：video6的ground RANSAC计算了18.154秒，因内点比例不足最终未应用删除；这不是0成本的跳过。point ranges关联在video6为13.950秒、video3为3.429秒，属于观察数和选点规模共同影响的CPU工作。这里给出成本与分支事实，不据单次跨场景差异推导严格复杂度或可回收时间。

{point_probe}

上一轮独立初始化复现与固定版Open3D绑定/pybind规则支持调查NN/SOR持GIL造成的PVS构造膨胀；该证据保留于[历史单视频记录](../reports/profiling-20261018/README.md)及其overlap结果。它不是本轮的受控加速实验，不能把线程重叠秒数直接承诺为可回收秒数。下一步仍宜验证GIL隔离、现有atlas worker的完整交付收益，再做DBSCAN-off/proxy/early-background的视觉对照，本轮没有实施或promotion。

这是统一**细分profiling模式**：AST span、scalar点数观察、CPU时间读取与CUDA events增加了未量化开销。跨video同模式适合描述这些输入的规模差异；不宣称无仪器生产延迟。GPU2共享resident、主机CPU共享、OS页缓存/文件缓存未清空；新进程/新DA3-PVS结果cache不等于物理冷盘。

前一轮video3的248.900/128.750/137.148秒仍是那次实测，保留[历史summary](../reports/profiling-20261018/summary.json)，**未混进本轮均值**。不同抽帧还原、共享负载、随机拟合和仪器条件不支持把前后差额叫作受控提速率。当前文档取代旧单样本作为性能入口。

验证只覆盖诊断脚本接线、完整阶段/帧数/输出、计时闭合和必要格式边界；没有修改生产算法，因此未重复跑全量pytest或额外hash/CRC。未测热服务、HTTP/BSON/COS、网络、浏览器首屏、人工画质、匹配/面积准确率、native内部细分、精确GIL等待归因和仪器开销。逐视频JSON与所有内部均值/范围/n完整保留，失败和未测不静默删除。
'''
    (ROOT / 'docs/profiling_breakdown.md').write_text(text)
    internal = ['# 同次完整运行内的全部内部计时均值',
                f'\n成功video {aggregate["video_ids"]}；每项独立等video平均，n按实际观测。缺失不补0。父子inclusive、wall/context、CUDA不能相加。完整标量、source pointer和语义见[CSV](detail-averages.csv)及[JSON](aggregate.json)。\n']
    for role in ('core','da3','delivery'):
        rows = [(key,row) for key,row in details.items() if row['section']=='wall' and row['process_role']==role and row['field']=='inclusive_s']
        internal.append('## '+role+' host wall\n')
        internal.append(table(['label','均值秒','self均值秒','min–max秒','平均calls','n'], [
            ('`'+row['label'].replace('|','\\|')+'`',fmt(row['mean']),wall(role,row['label'],'self_s'),fmt(row['min'])+'–'+fmt(row['max']),
             cell(f'wall/{role}/{row["label"]}/calls',digits=1,detail=True),row['n'])
            for key,row in sorted(rows,key=lambda pair: pair[1]['mean'] or 0,reverse=True)]))
        internal.append('\n')
    internal.append('## CUDA event（嵌套区间）\n\n'+cuda)
    internal.append('\n\n## 点数流向\n\n'+point_table)
    (HERE / 'INTERNALS.md').write_text('\n'.join(internal)+'\n')
    print(json.dumps(dict(videos=aggregate['video_ids'], canonical_lines=len(text.splitlines()), detail_metrics=len(details))))


if __name__ == '__main__':
    main()
