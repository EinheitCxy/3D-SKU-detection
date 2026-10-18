# 多视频 profiling 输入准备与采样证据

核对日期：2026-10-18。这里只准备独立输入，不启动DA3/PVS/GPU、完整benchmark或服务。`prepare_inputs.py`仅写调用者指定的新case目录，固定数据集名为`dataset`；生产源码和先前report不变。

## 已核对的输入数量与范围

当前`fd-videos/1.mp4`至`7.mp4`均有同名检测JSON。JSON均为frame list，每帧只有`classes/objects`，对象只有position/classes/confidences等业务字段，没有原始video frame index、timestamp或sampling interval。帧数/objects已直接读取JSON复核；以下时长/fps/分辨率引用协调者的本次metadata核对，脚本实际运行时会再读取视频metadata并记录。

| video | JSON帧数 | objects | 时长s | 当前可用采样证据 |
|---|---:|---:|---:|---|
| 1 | 21 | 342 | 30.567 | 未找到对应21帧的原始采样记录 |
| 2 | 12 | 131 | 23.367 | 未确认；1.5s前段与2s全程均可能 |
| 3 | 29 | 711 | 42.900 | 已有profiling使用0起1.5s recipe |
| 4 | 28 | 1,381 | 41.000 | 未找到原始采样记录 |
| 5 | 19 | 899 | 27.867 | 未找到原始采样记录 |
| 6 | 48 | 2,780 | 71.633 | 未找到原始采样记录 |
| 7 | 16 | 395 | 23.233 | 未找到原始采样记录 |
| 8 | 无JSON | — | — | 缺检测/分类输入，不纳入同口径 |

1..7视频均为30fps、1080×1920。1/3/4/5/6/7的JSON数量与从0起每1.5s覆盖视频的数量吻合，2的12帧与每2s覆盖23.367s吻合；**数量吻合只是推断，不是对齐证据**。尤其不能给2默套1.5s或2s。未取得明确采样依据前，1/2/4/5/6/7不得生成声称已对齐的benchmark输入。

## 找到了什么、哪些不能当作依据

- `runtime/video3-full-profile-20261016/run_profile.py:prepare_input` 明确`round(i*1.5*fps)`，29张quality95 JPEG，对应0、45…1260源帧；这是可重现旧video3 profiling的recipe，不证明找回了原始检测时间戳。
- `reports/viewer_extent.md:113`记录旧`runtime/fd-videos-20260911/`在2026-10-13已获授权删除；曾用video2的`dataset/images`路径及12帧仅在历史报告留存。当前源目录、旧`prepare_video2.py`和对应图像均未找到，不能直接恢复时间表。
- `reports/video1_1fps_reproduction.md:29–32`用的是31帧`1-classifier.json`和0..30s/1fps；本次`1.json`只有21帧，不能套用该31帧记录。
- `imdata/floor_display2`、fd2匹配研究等名称不等于`fd-videos/2.mp4`；未找到连接它们的原始采样metadata，未据此混用图片。
- `fd-videos/README.md`讲的是SDR/30fps目录转码历史，与当前根目录布局不符；只能作历史说明，不能当JSON/video对齐真相。
- 搜索限于JSON小metadata、既有输入脚本/报告、perf中的fd-videos引用及相关目录名；未扫大数组或整批图像，未运行图像匹配。

协调者可向用户确认的最小问题是：1/3/4/5/6/7是否均从0起间隔1.5s，2是否从0起间隔2s；也可提供每个JSON对应的原video frame-index列表。得到答复后，把确认内容保存为下述sampling record，明确标记为用户确认而非找回的历史timestamp。

## CLI、显式sampling record与接口

已建立的video3 recipe可以直接运行：

```bash
UV_CACHE_DIR=/tmp/3d-profile-20261018-uv CUDA_VISIBLE_DEVICES='' \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python python \
reports/profiling-multivideo-20261018/prepare_inputs.py \
--video 3 --out runtime/profiling-multivideo-20261018/video3
```

其他视频需明确传`--sampling-record <已确认的JSON文件>`。record格式为：

```json
{
  "video_id": 2,
  "video_frame_indices": [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660],
  "sampling_basis": {
    "source": "填写实际原始时间表路径，或用户明确确认的指令及日期",
    "type": "user_confirmed_or_original_record"
  }
}
```

**上例只是字段格式示例，video2的2s cadence尚未确认，不可据此启动benchmark。** record要求video_id匹配CLI、严格递增整数索引、与全部JSON帧数完全相同且不越视频末尾；缺依据报错，不用自动猜测或fallback。video3也可用显式record覆盖旧recipe，以采用更权威的原始时间表。

每个case布局为：

```text
<out>/input.json
<out>/dataset/images/0.jpg ... (F-1).jpg
<out>/dataset/detections_results/0.json ... (F-1).json
```

F和objects总量来自该video的JSON，动态变化；**没有max-frame截断**。调用者应读取`input.json.frames`、`objects`、`objects_per_frame`，不得硬编码29/711。JSON帧槽位从0连续编号，`video_frame_indices[i]`和`frame_times_s[i]`记录它映射到的源帧和实际index/fps时刻。

抽帧使用顺序`VideoCapture.grab()`，只对选中源帧`retrieve()`，OpenCV JPEG quality=95。保留全部JSON帧，失败即报错并留存失败`input.json`及已有文件；out必须事前不存在，不覆盖旧输入。

分类适配直接从`runtime/worktrees/mapping-docker-rebuild/processor.py:132–249` AST提取**现有五个helpers**：`_canonical_frame`、`_canonical_object`、`_validate_labels`、`_validate_confidence`、`_reject_nonfinite`。保留原`build_resolved_classification(domain=51)`调用与校验，输出原有`skus`包裹结构；不运行classifier、不自造label/分类、不加入fallback。

## 计时与当前验证

`input.json`记录fps、视频总帧数、源帧indices/times、F/objects、图像shape、采样依据、canonical来源、JPEG设置及细分计时：

- JSON/视频metadata/采样计划；canonical helper装载；
- grab/retrieve累计；JPEG编码与写入累计；
- canonical frame适配累计；canonical JSON编码与写入累计；
- 完整`preprocessing_wall_s`。

这些全是**预处理**，单独报告，不计core pipeline。细分是完整wall的部分，不能把细分再叠加到wall。

已在指定现成uv Python、`CUDA_VISIBLE_DEVICES=''`下通过syntax、五helper AST wiring、1..7 JSON数量和每视频首帧canonical适配的短metadata smoke；核对video3索引0..1260 step45，并验证其余未确认cadence拒绝默猜。没有解码视频、写case产物或运行benchmark。实际抽帧及正式输入准备由协调者在采样依据明确后执行。
