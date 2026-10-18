# 同次完整运行内的全部内部计时均值

成功video [1, 3, 4, 5, 6, 7]；每项独立等video平均，n按实际观测。缺失不补0。父子inclusive、wall/context、CUDA不能相加。完整标量、source pointer和语义见[CSV](detail-averages.csv)及[JSON](aggregate.json)。

## core host wall

| label | 均值秒 | self均值秒 | min–max秒 | 平均calls | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| `export_web_viewer_bundle` | 57.844 | 0.002 | 33.619–108.261 | 1.0 | 6 |
| `viewer.export_web_viewer_bundle` | 57.843 | 0.066 | 33.618–108.260 | 1.0 | 6 |
| `_sample_points` | 49.267 | 0.003 | 31.302–85.071 | 1.0 | 6 |
| `viewer._sample_points` | 49.264 | 0.007 | 31.300–85.067 | 1.0 | 6 |
| `sampling.geometry_filter` | 44.105 | 0.004 | 28.218–74.787 | 1.0 | 6 |
| `filter.filter_scene_points` | 44.101 | 0.145 | 28.216–74.781 | 1.0 | 6 |
| `filter.DBSCAN_block` | 32.943 | 0.146 | 21.276–55.092 | 1.0 | 6 |
| `filter._cluster_labels` | 32.265 | 0.006 | 20.875–53.830 | 1.0 | 6 |
| `filter.DBSCAN_native` | 32.259 | 32.259 | 20.871–53.819 | 1.0 | 6 |
| `run_sku_matching` | 31.871 | 0.020 | 21.032–57.773 | 1.0 | 6 |
| `reference_matching` | 31.851 | 8.171 | 21.020–57.733 | 26.8 | 6 |
| `run_reconstruction` | 24.199 | 24.199 | 23.190–26.200 | 1.0 | 6 |
| `pvs_model_load` | 18.686 | 0.006 | 15.345–25.489 | 1.0 | 6 |
| `pvs.model_load` | 18.680 | 0.048 | 15.339–25.484 | 1.0 | 6 |
| `utils/sam31_pvs.py:load_model:L22` | 16.979 | 0.000 | 13.631–23.876 | 1.0 | 6 |
| `utils/sam31_pvs.py:load_model:L23` | 16.979 | 16.979 | 13.631–23.875 | 1.0 | 6 |
| `prepare_web_viewer_geometry` | 14.131 | 0.002 | 9.362–23.646 | 1.0 | 6 |
| `viewer.prepare_web_viewer_geometry` | 14.129 | 0.217 | 9.360–23.643 | 1.0 | 6 |
| `filter.prepare_scene_filter` | 11.034 | 1.933 | 7.468–17.998 | 1.0 | 6 |
| `filter.ground_block` | 10.316 | 0.117 | 6.421–18.190 | 1.0 | 6 |
| `filter.RANSAC_native` | 9.868 | 9.868 | 6.018–18.154 | 1.0 | 6 |
| `filter.SOR_native` | 5.736 | 5.736 | 3.563–9.970 | 1.0 | 6 |
| `pvs_predict` | 4.995 | 0.144 | 2.642–10.301 | 26.8 | 6 |
| `pvs.predict` | 4.850 | 0.008 | 2.553–10.041 | 26.8 | 6 |
| `utils/sam31_pvs.py:predict:L84` | 4.343 | 0.005 | 2.229–9.126 | 26.8 | 6 |
| `_attach_point_index_ranges` | 4.336 | 0.003 | 0.405–13.954 | 1.0 | 6 |
| `viewer._attach_point_index_ranges` | 4.333 | 4.333 | 0.404–13.950 | 1.0 | 6 |
| `filter._median_nn_distance` | 3.365 | 0.039 | 2.129–5.773 | 1.0 | 6 |
| `filter.NN_native` | 3.326 | 3.326 | 2.112–5.699 | 1.0 | 6 |
| `sampling.budget` | 2.520 | 0.077 | 1.246–5.652 | 1.0 | 6 |
| `utils/sam31_pvs.py:predict:L94` | 2.400 | 0.032 | 0.982–5.833 | 26.8 | 6 |
| `adaptive.select` | 2.332 | 2.332 | 1.186–5.001 | 223.5 | 6 |
| `get_scene_orientation` | 1.981 | 0.000 | 1.018–4.302 | 1.0 | 6 |
| `viewer.get_scene_orientation` | 1.980 | 0.364 | 1.018–4.302 | 1.0 | 6 |
| `utils/sam31_pvs.py:predict:L103` | 1.977 | 0.031 | 0.819–4.772 | 281.2 | 6 |
| `pvs.decoder.host` | 1.946 | 1.946 | 0.807–4.694 | 281.2 | 6 |
| `surfel.grid_tangents` | 1.790 | 1.790 | 1.104–3.278 | 1.0 | 6 |
| `_generate_thumbnails` | 1.650 | 0.000 | 0.606–3.708 | 1.0 | 6 |
| `viewer._generate_thumbnails` | 1.650 | 1.650 | 0.606–3.707 | 1.0 | 6 |
| `orientation.fit_scene_orientation` | 1.616 | 0.038 | 0.788–3.644 | 1.0 | 6 |
| `viewer._sample_product_points` | 1.486 | 0.111 | 0.329–4.212 | 1.0 | 6 |
| `filter.pcd_select` | 1.465 | 1.465 | 0.989–2.047 | 2.8 | 6 |
| `orientation.RANSAC_native` | 1.431 | 1.431 | 0.621–3.417 | 2.8 | 6 |
| `utils/sam31_pvs.py:predict:L87` | 1.263 | 0.005 | 0.866–2.045 | 26.8 | 6 |
| `pvs.encoder.host` | 1.258 | 1.258 | 0.863–2.037 | 26.8 | 6 |
| `sampling.postfilter_voxels` | 1.027 | 0.231 | 0.589–1.655 | 1.0 | 6 |
| `surfel.prepare_surfel_textures` | 0.903 | 0.903 | 0.536–1.651 | 1.0 | 6 |
| `_select_background_voxels` | 0.796 | 0.000 | 0.447–1.211 | 1.0 | 6 |
| `viewer._select_background_voxels` | 0.795 | 0.795 | 0.446–1.211 | 1.0 | 6 |
| `filter.SOR_apply_block` | 0.698 | 0.095 | 0.431–1.217 | 1.0 | 6 |
| `utils/sam31_pvs.py:predict:L91` | 0.673 | 0.673 | 0.376–1.235 | 26.8 | 6 |
| `utils/sam31_pvs.py:load_model:L19` | 0.663 | 0.663 | 0.637–0.694 | 1.0 | 6 |
| `utils/sam31_pvs.py:load_model:L49` | 0.581 | 0.581 | 0.526–0.705 | 1.0 | 6 |
| `utils/sam31_pvs.py:predict:L77` | 0.444 | 0.444 | 0.295–0.763 | 26.8 | 6 |
| `sampling.features` | 0.436 | 0.079 | 0.261–0.771 | 1.0 | 6 |
| `sampling.serialize_arrays` | 0.404 | 0.404 | 0.191–0.631 | 1.0 | 6 |
| `adaptive.features` | 0.357 | 0.357 | 0.211–0.628 | 1.0 | 6 |
| `sampling.setup_geometry` | 0.356 | 0.356 | 0.222–0.611 | 1.0 | 6 |
| `sampling.labels` | 0.328 | 0.006 | 0.127–0.783 | 1.0 | 6 |
| `_instance_labels_v2` | 0.322 | 0.000 | 0.122–0.774 | 1.0 | 6 |
| `viewer._instance_labels_v2` | 0.322 | 0.322 | 0.122–0.773 | 1.0 | 6 |
| `utils/sam31_pvs.py:load_model:L48` | 0.252 | 0.252 | 0.212–0.363 | 1.0 | 6 |
| `run_dedup_sequence` | 0.248 | 0.076 | 0.075–0.501 | 1.0 | 6 |
| `utils/sam31_pvs.py:predict:L120` | 0.196 | 0.018 | 0.066–0.502 | 281.2 | 6 |
| `_load_da3_cache` | 0.185 | 0.000 | 0.128–0.329 | 1.0 | 6 |
| `viewer._load_da3_cache` | 0.185 | 0.185 | 0.128–0.329 | 1.0 | 6 |
| `utils/sam31_pvs.py:load_model:L31` | 0.155 | 0.155 | 0.148–0.166 | 1.0 | 6 |
| `surfel.prepare_surfels` | 0.153 | 0.153 | 0.081–0.229 | 1.0 | 6 |
| `viewer._compute_world_to_view` | 0.152 | 0.152 | 0.070–0.346 | 1.0 | 6 |
| `_publish_bundle` | 0.140 | 0.000 | 0.055–0.334 | 1.0 | 6 |
| `viewer._publish_bundle` | 0.140 | 0.140 | 0.055–0.334 | 1.0 | 6 |
| `orientation.NN_native` | 0.138 | 0.138 | 0.129–0.148 | 1.0 | 6 |
| `utils/sam31_pvs.py:predict:L121` | 0.110 | 0.110 | 0.036–0.279 | 1,084.7 | 6 |
| `dedup.add_global_id_to_jsons` | 0.092 | 0.092 | 0.018–0.307 | 1.0 | 6 |
| `viewer._robust_display_bounds` | 0.091 | 0.091 | 0.043–0.127 | 1.0 | 6 |
| `sampling.selection_filters` | 0.081 | 0.081 | 0.047–0.165 | 1.0 | 6 |
| `utils/sam31_pvs.py:predict:L122` | 0.069 | 0.069 | 0.024–0.178 | 1,084.7 | 6 |
| `utils/sam31_pvs.py:predict:L117` | 0.064 | 0.064 | 0.022–0.162 | 281.2 | 6 |
| `utils/sam31_pvs.py:predict:L115` | 0.063 | 0.063 | 0.028–0.146 | 281.2 | 6 |
| `dedup._publish_global_pair` | 0.043 | 0.043 | 0.014–0.112 | 1.0 | 6 |
| `utils/sam31_pvs.py:predict:L126` | 0.040 | 0.040 | 0.013–0.112 | 26.8 | 6 |
| `dedup.build_global_mapping` | 0.033 | 0.033 | 0.011–0.082 | 1.0 | 6 |
| `utils/sam31_pvs.py:predict:L123` | 0.024 | 0.024 | 0.009–0.061 | 281.2 | 6 |
| `utils/sam31_pvs.py:predict:L98` | 0.018 | 0.018 | 0.007–0.045 | 281.2 | 6 |
| `orientation.pcd_select` | 0.017 | 0.017 | 0.007–0.029 | 4.0 | 3 |
| `utils/sam31_pvs.py:predict:L124` | 0.011 | 0.011 | 0.004–0.029 | 281.2 | 6 |
| `utils/sam31_pvs.py:predict:L111` | 0.010 | 0.010 | 0.004–0.025 | 281.2 | 6 |
| `viewer._minimal_objects` | 0.008 | 0.008 | 0.003–0.022 | 1.0 | 6 |
| `utils/sam31_pvs.py:predict:L69` | 0.006 | 0.006 | 0.004–0.010 | 26.8 | 6 |
| `dedup.parse_all_matches` | 0.004 | 0.004 | 0.003–0.007 | 1.0 | 6 |
| `utils/sam31_pvs.py:predict:L79` | 0.003 | 0.003 | 0.003–0.005 | 26.8 | 6 |
| `utils/sam31_pvs.py:predict:L78` | 0.003 | 0.003 | 0.002–0.005 | 26.8 | 6 |
| `utils/sam31_pvs.py:predict:L118` | 0.002 | 0.002 | 0.001–0.005 | 281.2 | 6 |
| `utils/sam31_pvs.py:predict:L95` | 0.002 | 0.002 | 0.001–0.005 | 281.2 | 6 |
| `utils/sam31_pvs.py:predict:L82` | 0.002 | 0.002 | 0.001–0.003 | 26.8 | 6 |
| `utils/sam31_pvs.py:predict:L63` | 0.001 | 0.001 | 0.001–0.002 | 26.8 | 6 |
| `utils/sam31_pvs.py:predict:L125` | 0.001 | 0.001 | 0.001–0.002 | 26.8 | 6 |
| `utils/sam31_pvs.py:predict:L90` | 0.001 | 0.001 | 0.000–0.001 | 26.8 | 6 |
| `utils/sam31_pvs.py:predict:L89` | 0.000 | 0.000 | 0.000–0.001 | 26.8 | 6 |
| `utils/sam31_pvs.py:load_model:L36` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `utils/sam31_pvs.py:load_model:L41` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `utils/sam31_pvs.py:predict:L60` | 0.000 | 0.000 | 0.000–0.000 | 26.8 | 6 |
| `utils/sam31_pvs.py:predict:L88` | 0.000 | 0.000 | 0.000–0.000 | 26.8 | 6 |
| `utils/sam31_pvs.py:predict:L61` | 0.000 | 0.000 | 0.000–0.000 | 26.8 | 6 |
| `utils/sam31_pvs.py:predict:L58` | 0.000 | 0.000 | 0.000–0.000 | 26.8 | 6 |
| `validate_dataset` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `utils/sam31_pvs.py:predict:L92` | 0.000 | 0.000 | 0.000–0.000 | 26.8 | 6 |
| `utils/sam31_pvs.py:predict:L65` | 0.000 | 0.000 | 0.000–0.000 | 26.8 | 6 |
| `utils/sam31_pvs.py:predict:L93` | 0.000 | 0.000 | 0.000–0.000 | 26.8 | 6 |
| `utils/sam31_pvs.py:predict:L83` | 0.000 | 0.000 | 0.000–0.000 | 26.8 | 6 |
| `utils/sam31_pvs.py:load_model:L17` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `utils/sam31_pvs.py:load_model:L50` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `utils/sam31_pvs.py:load_model:L32` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |


## da3 host wall

| label | 均值秒 | self均值秒 | min–max秒 | 平均calls | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| `da3.runner_main` | 17.349 | 0.006 | 16.244–19.373 | 1.0 | 6 |
| `src/da3_runner.py:main:L228` | 14.089 | 0.009 | 13.975–14.367 | 1.0 | 6 |
| `da3.load_cpu` | 12.203 | 0.001 | 12.081–12.465 | 1.0 | 6 |
| `da3.network_constructor` | 11.320 | 11.320 | 11.171–11.555 | 1.0 | 6 |
| `da3.imports_and_instrumentation` | 5.082 | 5.082 | 4.951–5.177 | 1.0 | 6 |
| `src/da3_runner.py:main:L237` | 2.452 | 0.000 | 1.710–3.957 | 1.0 | 6 |
| `da3.inference_api` | 2.452 | 0.018 | 1.710–3.957 | 1.0 | 6 |
| `da3.forward_native_sync` | 2.078 | 0.001 | 1.452–3.367 | 1.0 | 6 |
| `da3.nested_forward.host` | 2.077 | 0.000 | 1.450–3.366 | 1.0 | 6 |
| `da3.branch_forward.host` | 2.072 | 0.000 | 1.445–3.359 | 2.0 | 6 |
| `da3.model_to_device` | 1.878 | 1.878 | 1.870–1.893 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L362` | 1.669 | 0.000 | 1.139–2.763 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net.forward:L132` | 1.244 | 0.000 | 0.827–2.121 | 2.0 | 6 |
| `da3.backbone_forward.host` | 1.244 | 1.244 | 0.827–2.121 | 2.0 | 6 |
| `da3.safetensors_load_model` | 0.881 | 0.133 | 0.760–1.012 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net.forward:L139` | 0.709 | 0.000 | 0.522–1.015 | 2.0 | 6 |
| `da3.state_assignment` | 0.638 | 0.638 | 0.535–0.736 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net._process_depth_head:L209` | 0.602 | 0.602 | 0.389–0.801 | 2.0 | 6 |
| `src/da3_runner.py:main:L220` | 0.436 | 0.436 | 0.281–0.778 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L365` | 0.403 | 0.000 | 0.306–0.596 | 1.0 | 6 |
| `src/da3_runner.py:main:L295` | 0.218 | 0.035 | 0.131–0.376 | 1.0 | 6 |
| `da3.NPZ_write_excluding_argument_evaluation` | 0.183 | 0.183 | 0.111–0.321 | 1.0 | 6 |
| `da3.preprocess` | 0.182 | 0.182 | 0.132–0.272 | 1.0 | 6 |
| `da3.processed_RGB` | 0.150 | 0.150 | 0.093–0.264 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net.forward:L148` | 0.118 | 0.000 | 0.074–0.223 | 2.0 | 6 |
| `da3.safetensors_load_file` | 0.110 | 0.110 | 0.077–0.144 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net._process_camera_estimation:L215` | 0.106 | 0.106 | 0.065–0.213 | 2.0 | 6 |
| `src/da3_runner.py:main:L271` | 0.085 | 0.000 | 0.055–0.139 | 1.0 | 6 |
| `da3.CPU_backprojection` | 0.085 | 0.085 | 0.055–0.139 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net._process_mono_sky_estimation:L162` | 0.060 | 0.060 | 0.017–0.164 | 1.0 | 6 |
| `src/da3_runner.py:main:L233` | 0.052 | 0.052 | 0.031–0.097 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net._process_mono_sky_estimation:L173` | 0.042 | 0.042 | 0.036–0.048 | 1.0 | 6 |
| `da3.prediction_D2H` | 0.013 | 0.013 | 0.007–0.021 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net._process_mono_sky_estimation:L167` | 0.011 | 0.011 | 0.004–0.018 | 1.0 | 6 |
| `da3.input_enqueue` | 0.010 | 0.010 | 0.006–0.015 | 1.0 | 6 |
| `src/da3_runner.py:main:L230` | 0.005 | 0.005 | 0.005–0.005 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net._process_mono_sky_estimation:L168` | 0.004 | 0.004 | 0.003–0.005 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L369` | 0.004 | 0.000 | 0.003–0.005 | 1.0 | 6 |
| `da3.depth_alignment.host` | 0.004 | 0.000 | 0.003–0.005 | 1.0 | 6 |
| `src/da3_runner.py:main:L267` | 0.003 | 0.003 | 0.002–0.006 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_depth_alignment:L397` | 0.001 | 0.001 | 0.001–0.002 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L370` | 0.001 | 0.000 | 0.001–0.001 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_depth_alignment:L408` | 0.001 | 0.001 | 0.001–0.001 | 1.0 | 6 |
| `da3.sky_handling.host` | 0.001 | 0.000 | 0.001–0.001 | 1.0 | 6 |
| `src/da3_runner.py:main:L197` | 0.001 | 0.001 | 0.001–0.001 | 1.0 | 6 |
| `src/da3_runner.py:main:L208` | 0.000 | 0.000 | 0.000–0.001 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net._process_mono_sky_estimation:L176` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._handle_sky_regions:L435` | 0.000 | 0.000 | 0.000–0.001 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_depth_alignment:L398` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L277` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L231` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L368` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L279` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._handle_sky_regions:L438` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `da3.metric_scaling.host` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net._process_mono_sky_estimation:L164` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L243` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_metric_scaling:L379` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_depth_alignment:L401` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net._process_mono_sky_estimation:L161` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_depth_alignment:L407` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_depth_alignment:L393` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L329` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_depth_alignment:L396` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_depth_alignment:L406` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L222` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._handle_sky_regions:L429` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L218` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L292` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._handle_sky_regions:L430` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L254` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_depth_alignment:L412` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L224` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L284` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_depth_alignment:L411` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_depth_alignment:L390` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._handle_sky_regions:L425` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net.forward:L151` | 0.000 | 0.000 | 0.000–0.000 | 2.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_depth_alignment:L414` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L285` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L221` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L291` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L226` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L207` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L256` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L288` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L278` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_depth_alignment:L413` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net.forward:L136` | 0.000 | 0.000 | 0.000–0.000 | 2.0 | 6 |
| `src/da3_runner.py:main:L198` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L272` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L289` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L204` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net._process_mono_sky_estimation:L159` | 0.000 | 0.000 | 0.000–0.000 | 2.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net.forward:L153` | 0.000 | 0.000 | 0.000–0.000 | 2.0 | 6 |
| `src/da3_runner.py:main:L266` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L262` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L205` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L236` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net._process_camera_estimation:L228` | 0.000 | 0.000 | 0.000–0.000 | 2.0 | 6 |
| `src/da3_runner.py:main:L247` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net.forward:L126` | 0.000 | 0.000 | 0.000–0.000 | 2.0 | 6 |
| `src/da3_runner.py:main:L257` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L227` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L260` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L199` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net._process_mono_sky_estimation:L179` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L216` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L253` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L261` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._handle_sky_regions:L442` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_metric_scaling:L383` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L248` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L372` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `src/da3_runner.py:main:L265` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |
| `Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net._apply_depth_alignment:L416` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |


## delivery host wall

| label | 均值秒 | self均值秒 | min–max秒 | 平均calls | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| `delivery.total` | 103.393 | 0.001 | 58.685–175.538 | 1.0 | 6 |
| `glb.total` | 101.179 | 0.722 | 57.584–172.135 | 1.0 | 6 |
| `bake._bake_batch` | 54.401 | 0.183 | 32.202–91.090 | 1,181.5 | 6 |
| `texture.encode_atlas` | 39.802 | 0.005 | 22.179–70.265 | 28.8 | 6 |
| `texture._compact_rgba` | 38.864 | 0.774 | 21.617–68.687 | 28.8 | 6 |
| `bake.block.tile_corners_dilation` | 22.070 | 1.451 | 13.184–36.643 | 1,181.5 | 6 |
| `PIL.save` | 20.842 | 20.842 | 11.573–35.554 | 28.8 | 6 |
| `bake._dilate_rgb` | 20.619 | 20.619 | 12.331–34.199 | 1,181.5 | 6 |
| `PIL.quantize` | 17.018 | 17.018 | 9.458–31.446 | 28.8 | 6 |
| `bake.block.bilinear_rgb` | 14.674 | 0.341 | 8.588–25.061 | 1,181.5 | 6 |
| `bake._bilinear` | 14.333 | 14.333 | 8.394–24.486 | 1,181.5 | 6 |
| `bake.block.texel_projection_mask` | 10.704 | 10.704 | 6.325–17.857 | 1,181.5 | 6 |
| `bake.block.depth_and_source_mask` | 4.816 | 4.816 | 2.835–8.041 | 1,181.5 | 6 |
| `delivery.imports` | 3.552 | 3.552 | 3.505–3.580 | 1.0 | 6 |
| `lod.select_surfel_lod` | 3.303 | 2.382 | 0.956–5.207 | 1.0 | 6 |
| `zip.total` | 2.214 | 0.034 | 1.100–3.402 | 1.0 | 6 |
| `zip.member.binary` | 2.059 | 2.059 | 1.027–3.046 | 8.0 | 6 |
| `compact.geometry` | 1.570 | 1.509 | 0.933–2.629 | 28.8 | 6 |
| `bake.block.footprint_and_continuity` | 1.010 | 0.668 | 0.588–1.619 | 1,181.5 | 6 |
| `texture._binary_alpha` | 0.933 | 0.933 | 0.559–1.571 | 28.8 | 6 |
| `lod._covered` | 0.921 | 0.875 | 0.276–1.505 | 45.8 | 6 |
| `PIL.convert` | 0.771 | 0.771 | 0.463–1.382 | 55.7 | 6 |
| `bake._page` | 0.587 | 0.587 | 0.365–0.997 | 28.8 | 6 |
| `bake.block.radius_final_diameter` | 0.505 | 0.073 | 0.311–0.845 | 1,181.5 | 6 |
| `bake.block.camera_visibility` | 0.440 | 0.440 | 0.264–0.718 | 1,181.5 | 6 |
| `bake._limit_final_diameter` | 0.432 | 0.432 | 0.267–0.724 | 1,181.5 | 6 |
| `bake._bound_footprint` | 0.342 | 0.342 | 0.202–0.560 | 1,181.5 | 6 |
| `glb._load` | 0.144 | 0.144 | 0.078–0.226 | 1.0 | 6 |
| `zip.member.jpeg` | 0.119 | 0.119 | 0.058–0.269 | 1,111.5 | 6 |
| `compact.view` | 0.093 | 0.093 | 0.056–0.165 | 285.3 | 6 |
| `compact.finish` | 0.077 | 0.000 | 0.045–0.135 | 1.0 | 6 |
| `compact._write_glb` | 0.077 | 0.077 | 0.045–0.135 | 1.0 | 6 |
| `lod._bound_footprint` | 0.045 | 0.045 | 0.014–0.073 | 45.8 | 6 |
| `delivery.detail_setup` | 0.042 | 0.042 | 0.040–0.045 | 1.0 | 6 |
| `compact.image` | 0.032 | 0.001 | 0.019–0.061 | 28.8 | 6 |
| `zip.member.json` | 0.001 | 0.001 | 0.001–0.003 | 3.0 | 6 |
| `delivery.resolve_generation` | 0.000 | 0.000 | 0.000–0.000 | 1.0 | 6 |


## CUDA event（嵌套区间）

| 进程 | CUDA event范围（保留父上下文） | 平均秒 | n |
| --- | ---: | ---: | ---: |
| core | `pvs.decoder` | 1.962 | 6 |
| core | `pvs.encoder` | 1.936 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L362 > da3.branch_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net.forward:L132 > da3.backbone_forward` | 1.248 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L362 > da3.branch_forward` | 1.669 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L365 > da3.branch_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net.forward:L132 > da3.backbone_forward` | 0.145 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L365 > da3.branch_forward` | 0.403 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L368 > da3.metric_scaling` | 0.000 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L369 > da3.depth_alignment` | 0.004 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L370 > da3.sky_handling` | 0.001 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward` | 2.077 | 6 |


## 点数流向

| 点数域 | 跨视频平均点数 | video1 | video3 | video4 | video5 | video6 | video7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense native grid | 3,786,720 (n=6) | 2,963,520 | 4,092,480 | 3,951,360 | 2,681,280 | 6,773,760 | 2,257,920 |
| export-valid | 3,786,720 (n=6) | 2,963,520 | 4,092,480 | 3,951,360 | 2,681,280 | 6,773,760 | 2,257,920 |
| 原SAM商品 | 1,560,418 (n=6) | 997,310 | 1,902,589 | 1,462,913 | 884,008 | 3,778,160 | 337,531 |
| 原背景 | 2,226,302 (n=6) | 1,966,210 | 2,189,891 | 2,488,447 | 1,797,272 | 2,995,600 | 1,920,389 |
| SOR raw保留 | 3,624,269 (n=6) | 2,835,143 | 3,888,295 | 3,803,234 | 2,588,164 | 6,453,776 | 2,177,002 |
| SOR商品救回 | 1,032 (n=6) | 196 | 825 | 579 | 506 | 3,298 | 788 |
| SOR应用后 | 3,625,301 (n=6) | 2,835,339 | 3,889,120 | 3,803,813 | 2,588,670 | 6,457,074 | 2,177,790 |
| DBSCAN raw保留 | 3,136,689 (n=6) | 2,357,677 | 3,402,853 | 3,244,481 | 2,070,858 | 5,894,636 | 1,849,627 |
| DBSCAN商品救回 | 34,408 (n=6) | 2,108 | 22,822 | 87,114 | 23,102 | 38,559 | 32,740 |
| DBSCAN应用后 | 3,171,096 (n=6) | 2,359,785 | 3,425,675 | 3,331,595 | 2,093,960 | 5,933,195 | 1,882,367 |
| ground RANSAC输入 | 3,171,096 (n=6) | 2,359,785 | 3,425,675 | 3,331,595 | 2,093,960 | 5,933,195 | 1,882,367 |
| ground RANSAC内点 | 324,564 (n=6) | 364,414 | 313,004 | 333,295 | 398,691 | 352,892 | 185,086 |
| ground raw保留 | 2,299,778 (n=5) | 1,995,371 | 3,112,671 | 2,998,300 | 1,695,269 | — | 1,697,281 |
| ground商品救回 | 91,223 (n=5) | 6 | 230,825 | 172,123 | 0 | — | 53,161 |
| ground应用后（仅调用apply时） | 2,391,001 (n=5) | 1,995,377 | 3,343,496 | 3,170,423 | 1,695,269 | — | 1,750,442 |
| 最终filter后 | 2,981,367 (n=6) | 1,995,377 | 3,343,496 | 3,170,423 | 1,695,269 | 5,933,195 | 1,750,442 |
| filter后商品 | 1,560,418 (n=6) | 997,310 | 1,902,589 | 1,462,913 | 884,008 | 3,778,160 | 337,531 |
| filter后背景 | 1,420,948 (n=6) | 998,067 | 1,440,907 | 1,707,510 | 811,261 | 2,155,035 | 1,412,911 |
| 背景voxel候选 | 1,081,097 (n=6) | 803,686 | 1,084,992 | 1,273,600 | 705,892 | 1,715,050 | 903,363 |
| 商品选入 | 1,264,058 (n=6) | 997,310 | 1,902,589 | 1,462,913 | 884,008 | 2,000,000 | 337,531 |
| 背景选入 | 500,000 (n=6) | 500,000 | 500,000 | 500,000 | 500,000 | 500,000 | 500,000 |
| Viewer选入总量 | 1,764,058 (n=6) | 1,497,310 | 2,402,589 | 1,962,913 | 1,384,008 | 2,500,000 | 837,531 |
