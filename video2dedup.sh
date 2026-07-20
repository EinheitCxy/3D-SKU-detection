# 1. 视频抽帧
cd /home/xingyu/3D_Recognization/frame_sampler

uv run python cli.py \
    ../small_fd_video/video-test/6-1.mp4 \
    --fps 1.0 \
    -o ../small_fd_video/video-test/6-1_frames

# 2. SKU 检测（生成 bbox + JSON）
cd /home/xingyu/3D_Recognization
uv run python bbox_gen.py \
    small_fd_video/video-test/6-1_frames/images \
    -o small_fd_video/video-test/6-1_frames

# 3. 运行完整 Pipeline
cd /home/xingyu/3D_Recognization/code

.venv/ python main.py \
    --mode pipeline \
    --dataset ../small_fd_video/video-test/6-1_frames \
    --algorithm 3d \
    --recon_backend da3 \
    --match_backend da3 \
    --save_root ../Output