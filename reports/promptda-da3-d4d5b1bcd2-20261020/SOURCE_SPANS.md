# Source statement detail

来自当前 native 函数的 AST 包装，源码文件和原行号保留；不改算法语句。exclusive wall 仅扣除已插桩的直接子区间，不等于纯算法 CPU 指令时间。

## da3: model_load

| 区间 | 次数 | inclusive wall / s | exclusive wall / s | CUDA stream / s |
| --- | --- | --- | --- | --- |
| load.module_to | 1 | 1.972140 | 1.972140 | — |
| load.cpu_network_construct | 1 | 1.562570 | 1.562570 | — |
| load.safetensors_load_model | 1 | 0.956338 | 0.148132 | — |
| load.state_assignment | 1 | 0.682409 | 0.682409 | — |
| load.checkpoint_coverage_audit | 1 | 0.317384 | 0.317384 | — |
| load.safetensors_load_file | 1 | 0.125798 | 0.125798 | — |

## da3: instrumented_request

| 区间 | 次数 | inclusive wall / s | exclusive wall / s | CUDA stream / s |
| --- | --- | --- | --- | --- |
| request.da3._run_model_forward | 1 | 1.534299 | 0.001551 | 1.534379 |
| module.model | 1 | 1.532748 | 0.001152 | 1.532791 |
| module.model.da3 | 1 | 1.267203 | 0.319678 | 1.267463 |
| module.model.da3.backbone | 1 | 0.800738 | 0.021794 | 0.829837 |
| request.da3._preprocess_inputs | 1 | 0.414373 | 0.414373 | — |
| module.model.da3_metric | 1 | 0.261384 | 0.215549 | 0.261499 |
| request.da3._add_processed_images | 1 | 0.156420 | 0.156420 | — |
| module.model.da3.head | 1 | 0.142170 | 0.142170 | 0.275285 |
| module.model.da3.backbone.pretrained.blocks.13 | 1 | 0.125324 | 0.000622 | 0.032567 |
| module.model.da3.backbone.pretrained.blocks.13.attn | 1 | 0.124020 | 0.124020 | 0.025715 |
| module.model.da3.backbone.pretrained.blocks.16 | 1 | 0.034241 | 0.000556 | 0.018574 |
| module.model.da3_metric.backbone | 1 | 0.033560 | 0.005832 | 0.145661 |
| module.model.da3.backbone.pretrained.blocks.16.attn | 1 | 0.033097 | 0.033097 | 0.011721 |
| module.model.da3.backbone.pretrained.blocks.28 | 1 | 0.032997 | 0.002312 | 0.016656 |
| module.model.da3.backbone.pretrained.blocks.20 | 1 | 0.032728 | 0.000515 | 0.016603 |
| module.model.da3.backbone.pretrained.blocks.34 | 1 | 0.032723 | 0.000506 | 0.016618 |
| module.model.da3.backbone.pretrained.blocks.36 | 1 | 0.032515 | 0.000520 | 0.016677 |
| module.model.da3.backbone.pretrained.blocks.38 | 1 | 0.032374 | 0.000540 | 0.016641 |
| module.model.da3.backbone.pretrained.blocks.30 | 1 | 0.032286 | 0.000493 | 0.016627 |
| module.model.da3.backbone.pretrained.blocks.32 | 1 | 0.032280 | 0.000509 | 0.016617 |
| module.model.da3.backbone.pretrained.blocks.18 | 1 | 0.032261 | 0.000473 | 0.016600 |
| module.model.da3.backbone.pretrained.blocks.26 | 1 | 0.032217 | 0.000483 | 0.016612 |
| module.model.da3.backbone.pretrained.blocks.22 | 1 | 0.032189 | 0.000471 | 0.016599 |
| module.model.da3.backbone.pretrained.blocks.14 | 1 | 0.032127 | 0.000545 | 0.016664 |
| module.model.da3.backbone.pretrained.blocks.24 | 1 | 0.032118 | 0.000489 | 0.016597 |
| module.model.da3.backbone.pretrained.blocks.34.attn | 1 | 0.031665 | 0.031665 | 0.009751 |
| module.model.da3.backbone.pretrained.blocks.20.attn | 1 | 0.031621 | 0.031621 | 0.009757 |
| module.model.da3.backbone.pretrained.blocks.36.attn | 1 | 0.031463 | 0.031463 | 0.009807 |
| module.model.da3.backbone.pretrained.blocks.18.attn | 1 | 0.031265 | 0.031265 | 0.009746 |
| module.model.da3.backbone.pretrained.blocks.38.attn | 1 | 0.031263 | 0.031263 | 0.009773 |
| module.model.da3.backbone.pretrained.blocks.30.attn | 1 | 0.031262 | 0.031262 | 0.009762 |
| module.model.da3.backbone.pretrained.blocks.26.attn | 1 | 0.031246 | 0.031246 | 0.009761 |
| module.model.da3.backbone.pretrained.blocks.32.attn | 1 | 0.031240 | 0.031240 | 0.009748 |
| module.model.da3.backbone.pretrained.blocks.22.attn | 1 | 0.031224 | 0.031224 | 0.009745 |
| module.model.da3.backbone.pretrained.blocks.24.attn | 1 | 0.031146 | 0.031146 | 0.009744 |
| module.model.da3.backbone.pretrained.blocks.14.attn | 1 | 0.030917 | 0.030917 | 0.009814 |
| module.model.da3.backbone.pretrained.blocks.28.attn | 1 | 0.030081 | 0.030081 | 0.009796 |
| request.da3._align_to_input_extrinsics_intrinsics | 1 | 0.018220 | 0.018220 | — |
| module.model.da3.backbone.pretrained.blocks.23 | 1 | 0.017060 | 0.000482 | 0.032889 |
| module.model.da3.backbone.pretrained.blocks.37 | 1 | 0.016490 | 0.000520 | 0.032548 |
| module.model.da3.backbone.pretrained.blocks.31 | 1 | 0.016478 | 0.000518 | 0.032501 |
| module.model.da3.backbone.pretrained.blocks.19 | 1 | 0.016459 | 0.000509 | 0.032438 |
| module.model.da3.backbone.pretrained.blocks.27 | 1 | 0.016433 | 0.000484 | 0.032482 |
| module.model.da3.backbone.pretrained.blocks.33 | 1 | 0.016419 | 0.000561 | 0.032498 |
| module.model.da3.backbone.pretrained.blocks.25 | 1 | 0.016409 | 0.000484 | 0.032401 |
| module.model.da3.backbone.pretrained.blocks.21 | 1 | 0.016324 | 0.000487 | 0.032416 |
| module.model.da3.backbone.pretrained.blocks.29 | 1 | 0.016277 | 0.000493 | 0.032478 |
| module.model.da3.backbone.pretrained.blocks.35 | 1 | 0.016276 | 0.000479 | 0.032461 |
| module.model.da3.backbone.pretrained.blocks.15 | 1 | 0.016263 | 0.000531 | 0.032452 |
| module.model.da3.backbone.pretrained.blocks.39 | 1 | 0.016261 | 0.000485 | 0.032472 |
| module.model.da3.backbone.pretrained.blocks.17 | 1 | 0.016148 | 0.000472 | 0.032404 |
| module.model.da3.backbone.pretrained.blocks.23.attn | 1 | 0.015932 | 0.015932 | 0.026035 |
| module.model.da3.backbone.pretrained.blocks.27.attn | 1 | 0.015453 | 0.015453 | 0.025616 |
| module.model.da3.backbone.pretrained.blocks.31.attn | 1 | 0.015418 | 0.015418 | 0.025632 |
| module.model.da3.backbone.pretrained.blocks.19.attn | 1 | 0.015409 | 0.015409 | 0.025576 |
| module.model.da3.backbone.pretrained.blocks.37.attn | 1 | 0.015407 | 0.015407 | 0.025678 |
| module.model.da3.backbone.pretrained.blocks.25.attn | 1 | 0.015405 | 0.015405 | 0.025549 |
| module.model.da3.backbone.pretrained.blocks.33.attn | 1 | 0.015338 | 0.015338 | 0.025629 |
| module.model.da3.backbone.pretrained.blocks.21.attn | 1 | 0.015323 | 0.015323 | 0.025564 |
| module.model.da3.backbone.pretrained.blocks.35.attn | 1 | 0.015315 | 0.015315 | 0.025591 |
| module.model.da3.backbone.pretrained.blocks.39.attn | 1 | 0.015245 | 0.015245 | 0.025600 |
| module.model.da3.backbone.pretrained.blocks.29.attn | 1 | 0.015241 | 0.015241 | 0.025607 |
| module.model.da3.backbone.pretrained.blocks.17.attn | 1 | 0.015165 | 0.015165 | 0.025550 |
| module.model.da3.backbone.pretrained.blocks.15.attn | 1 | 0.015134 | 0.015134 | 0.025599 |
| module.model.da3_metric.head | 1 | 0.012275 | 0.012275 | 0.114255 |
| request.da3._prepare_model_inputs | 1 | 0.010885 | 0.010885 | 0.010996 |
| request.da3._convert_to_prediction | 1 | 0.009141 | 0.009141 | 0.009226 |
| module.model.da3.cam_enc | 1 | 0.004172 | 0.004172 | 0.004238 |
| module.model.da3_metric.backbone.pretrained.blocks.15 | 1 | 0.002440 | 0.001855 | 0.005693 |
| request.da3._apply_depth_alignment | 1 | 0.002043 | 0.002043 | 0.002108 |
| module.model.da3.backbone.pretrained.blocks.0 | 1 | 0.001602 | 0.000512 | 0.010568 |
| module.model.da3_metric.backbone.pretrained.blocks.0 | 1 | 0.001585 | 0.000686 | 0.005688 |
| module.model.da3.backbone.pretrained.blocks.6 | 1 | 0.001365 | 0.000525 | 0.010516 |
| module.model.da3.backbone.pretrained.blocks.9 | 1 | 0.001352 | 0.000507 | 0.010537 |
| module.model.da3.backbone.pretrained.blocks.2 | 1 | 0.001351 | 0.000519 | 0.010508 |
| module.model.da3.backbone.pretrained.blocks.4 | 1 | 0.001349 | 0.000517 | 0.010506 |
| module.model.da3.backbone.pretrained.blocks.7 | 1 | 0.001331 | 0.000490 | 0.010527 |
| module.model.da3.backbone.pretrained.blocks.3 | 1 | 0.001307 | 0.000485 | 0.010506 |
| module.model.da3.backbone.pretrained.blocks.12 | 1 | 0.001293 | 0.000467 | 0.010520 |
| module.model.da3.backbone.pretrained.blocks.10 | 1 | 0.001284 | 0.000486 | 0.010528 |
| module.model.da3.backbone.pretrained.blocks.1 | 1 | 0.001284 | 0.000487 | 0.010509 |
| module.model.da3.backbone.pretrained.blocks.5 | 1 | 0.001271 | 0.000476 | 0.010506 |
| module.model.da3.backbone.pretrained.blocks.11 | 1 | 0.001268 | 0.000482 | 0.010527 |
| module.model.da3.backbone.pretrained.blocks.8 | 1 | 0.001208 | 0.000450 | 0.010534 |
| module.model.da3_metric.backbone.pretrained.blocks.12 | 1 | 0.001124 | 0.000505 | 0.005689 |
| module.model.da3_metric.backbone.pretrained.blocks.1 | 1 | 0.001115 | 0.000510 | 0.005686 |
| module.model.da3_metric.backbone.pretrained.blocks.20 | 1 | 0.001104 | 0.000510 | 0.005698 |
| module.model.da3_metric.backbone.pretrained.blocks.3 | 1 | 0.001103 | 0.000518 | 0.005687 |
| module.model.da3_metric.backbone.pretrained.blocks.13 | 1 | 0.001102 | 0.000500 | 0.005695 |
| module.model.da3_metric.backbone.pretrained.blocks.16 | 1 | 0.001099 | 0.000512 | 0.005698 |
| module.model.da3_metric.backbone.pretrained.blocks.4 | 1 | 0.001098 | 0.000509 | 0.005683 |
| module.model.da3_metric.backbone.pretrained.blocks.17 | 1 | 0.001096 | 0.000497 | 0.005695 |
| module.model.da3_metric.backbone.pretrained.blocks.19 | 1 | 0.001091 | 0.000497 | 0.005700 |
| module.model.da3_metric.backbone.pretrained.blocks.22 | 1 | 0.001088 | 0.000496 | 0.005697 |
| module.model.da3_metric.backbone.pretrained.blocks.23 | 1 | 0.001081 | 0.000499 | 0.005694 |
| module.model.da3_metric.backbone.pretrained.blocks.18 | 1 | 0.001080 | 0.000506 | 0.005699 |
| module.model.da3_metric.backbone.pretrained.blocks.21 | 1 | 0.001079 | 0.000497 | 0.005697 |
| module.model.da3_metric.backbone.pretrained.blocks.8 | 1 | 0.001076 | 0.000492 | 0.005675 |
| module.model.da3_metric.backbone.pretrained.blocks.2 | 1 | 0.001073 | 0.000490 | 0.005680 |
| module.model.da3_metric.backbone.pretrained.blocks.9 | 1 | 0.001051 | 0.000485 | 0.005684 |
| module.model.da3_metric.backbone.pretrained.blocks.5 | 1 | 0.001049 | 0.000485 | 0.005682 |
| module.model.da3_metric.backbone.pretrained.blocks.11 | 1 | 0.001045 | 0.000471 | 0.005673 |
| module.model.da3_metric.backbone.pretrained.blocks.14 | 1 | 0.001045 | 0.000482 | 0.005691 |
| module.model.da3_metric.backbone.pretrained.blocks.10 | 1 | 0.001041 | 0.000492 | 0.005693 |
| module.model.da3_metric.backbone.pretrained.blocks.7 | 1 | 0.001036 | 0.000479 | 0.005678 |
| module.model.da3_metric.backbone.pretrained.blocks.6 | 1 | 0.001028 | 0.000472 | 0.005686 |
| request.da3._normalize_extrinsics | 1 | 0.000846 | 0.000846 | 0.000926 |
| request.da3._handle_sky_regions | 1 | 0.000794 | 0.000794 | 0.000854 |
| module.model.da3.backbone.pretrained.blocks.0.mlp | 1 | 0.000728 | 0.000728 | 0.005224 |
| module.model.da3.backbone.pretrained.blocks.13.mlp | 1 | 0.000682 | 0.000682 | 0.005237 |
| module.model.da3.backbone.pretrained.blocks.14.mlp | 1 | 0.000665 | 0.000665 | 0.005236 |
| module.model.da3.backbone.pretrained.blocks.23.mlp | 1 | 0.000647 | 0.000647 | 0.005236 |
| module.model.da3.backbone.pretrained.blocks.28.mlp | 1 | 0.000605 | 0.000605 | 0.005249 |
| module.model.da3.backbone.pretrained.blocks.15.mlp | 1 | 0.000598 | 0.000598 | 0.005237 |
| module.model.da3.backbone.pretrained.blocks.20.mlp | 1 | 0.000592 | 0.000592 | 0.005235 |
| module.model.da3.backbone.pretrained.blocks.16.mlp | 1 | 0.000588 | 0.000588 | 0.005234 |
| module.model.da3_metric.backbone.pretrained.blocks.0.attn | 1 | 0.000583 | 0.000583 | 0.001904 |
| module.model.da3.backbone.pretrained.blocks.38.mlp | 1 | 0.000571 | 0.000571 | 0.005248 |
| module.model.da3.backbone.pretrained.blocks.37.mlp | 1 | 0.000563 | 0.000563 | 0.005248 |
| module.model.da3.backbone.pretrained.blocks.34.mlp | 1 | 0.000552 | 0.000552 | 0.005253 |
| module.model.da3.backbone.pretrained.blocks.29.mlp | 1 | 0.000543 | 0.000543 | 0.005250 |
| module.model.da3.backbone.pretrained.blocks.31.mlp | 1 | 0.000543 | 0.000543 | 0.005250 |
| module.model.da3.backbone.pretrained.blocks.19.mlp | 1 | 0.000541 | 0.000541 | 0.005241 |
| module.model.da3.backbone.pretrained.blocks.36.mlp | 1 | 0.000533 | 0.000533 | 0.005250 |
| module.model.da3.backbone.pretrained.blocks.39.mlp | 1 | 0.000531 | 0.000531 | 0.005250 |
| module.model.da3.backbone.pretrained.blocks.32.mlp | 1 | 0.000531 | 0.000531 | 0.005248 |
| module.model.da3.backbone.pretrained.blocks.30.mlp | 1 | 0.000531 | 0.000531 | 0.005249 |
| module.model.da3.backbone.pretrained.blocks.2.mlp | 1 | 0.000530 | 0.000530 | 0.005226 |
| module.model.da3.backbone.pretrained.blocks.6.mlp | 1 | 0.000530 | 0.000530 | 0.005231 |
| module.model.da3.backbone.pretrained.blocks.4.mlp | 1 | 0.000526 | 0.000526 | 0.005222 |
| module.model.da3.backbone.pretrained.blocks.12.mlp | 1 | 0.000525 | 0.000525 | 0.005234 |
| module.model.da3.backbone.pretrained.blocks.18.mlp | 1 | 0.000523 | 0.000523 | 0.005232 |
| module.model.da3.backbone.pretrained.blocks.33.mlp | 1 | 0.000520 | 0.000520 | 0.005251 |
| module.model.da3.backbone.pretrained.blocks.25.mlp | 1 | 0.000520 | 0.000520 | 0.005234 |
| module.model.da3.backbone.pretrained.blocks.9.mlp | 1 | 0.000519 | 0.000519 | 0.005242 |
| module.model.da3.backbone.pretrained.blocks.21.mlp | 1 | 0.000515 | 0.000515 | 0.005237 |
| module.model.da3.backbone.pretrained.blocks.17.mlp | 1 | 0.000510 | 0.000510 | 0.005234 |
| module.model.da3.backbone.pretrained.blocks.7.mlp | 1 | 0.000506 | 0.000506 | 0.005233 |
| module.model.da3.backbone.pretrained.blocks.1.mlp | 1 | 0.000505 | 0.000505 | 0.005226 |
| module.model.da3.backbone.pretrained.blocks.3.mlp | 1 | 0.000502 | 0.000502 | 0.005224 |
| module.model.da3.backbone.pretrained.blocks.10.mlp | 1 | 0.000502 | 0.000502 | 0.005235 |
| module.model.da3.backbone.pretrained.blocks.27.mlp | 1 | 0.000497 | 0.000497 | 0.005251 |
| module.model.da3.backbone.pretrained.blocks.5.mlp | 1 | 0.000494 | 0.000494 | 0.005224 |
| module.model.da3.backbone.pretrained.blocks.22.mlp | 1 | 0.000494 | 0.000494 | 0.005236 |
| module.model.da3.backbone.pretrained.blocks.11.mlp | 1 | 0.000492 | 0.000492 | 0.005234 |
| module.model.da3.backbone.pretrained.blocks.26.mlp | 1 | 0.000489 | 0.000489 | 0.005234 |
| module.model.da3.backbone.pretrained.blocks.24.mlp | 1 | 0.000483 | 0.000483 | 0.005235 |
| module.model.da3.backbone.pretrained.blocks.35.mlp | 1 | 0.000482 | 0.000482 | 0.005249 |
| module.model.da3.backbone.pretrained.blocks.8.mlp | 1 | 0.000478 | 0.000478 | 0.005237 |
| module.model.da3.cam_dec | 1 | 0.000446 | 0.000446 | 0.000178 |
| module.model.da3.backbone.pretrained.blocks.0.attn | 1 | 0.000361 | 0.000361 | 0.003664 |
| module.model.da3_metric.backbone.pretrained.blocks.1.attn | 1 | 0.000347 | 0.000347 | 0.001904 |
| module.model.da3_metric.backbone.pretrained.blocks.4.attn | 1 | 0.000344 | 0.000344 | 0.001904 |
| module.model.da3_metric.backbone.pretrained.blocks.12.attn | 1 | 0.000343 | 0.000343 | 0.001904 |
| module.model.da3_metric.backbone.pretrained.blocks.13.attn | 1 | 0.000342 | 0.000342 | 0.001909 |
| module.model.da3_metric.backbone.pretrained.blocks.22.attn | 1 | 0.000340 | 0.000340 | 0.001912 |
| module.model.da3_metric.backbone.pretrained.blocks.19.attn | 1 | 0.000340 | 0.000340 | 0.001913 |
| module.model.da3_metric.backbone.pretrained.blocks.8.attn | 1 | 0.000338 | 0.000338 | 0.001902 |
| module.model.da3_metric.backbone.pretrained.blocks.16.attn | 1 | 0.000336 | 0.000336 | 0.001910 |
| module.model.da3.backbone.pretrained.blocks.7.attn | 1 | 0.000336 | 0.000336 | 0.003671 |
| module.model.da3_metric.backbone.pretrained.blocks.23.attn | 1 | 0.000332 | 0.000332 | 0.001912 |
| module.model.da3_metric.backbone.pretrained.blocks.20.attn | 1 | 0.000331 | 0.000331 | 0.001913 |
| module.model.da3_metric.backbone.pretrained.blocks.17.attn | 1 | 0.000330 | 0.000330 | 0.001910 |
| module.model.da3_metric.backbone.pretrained.blocks.21.attn | 1 | 0.000327 | 0.000327 | 0.001911 |
| module.model.da3_metric.backbone.pretrained.blocks.2.attn | 1 | 0.000326 | 0.000326 | 0.001904 |
| module.model.da3.backbone.pretrained.blocks.9.attn | 1 | 0.000326 | 0.000326 | 0.003677 |
| module.model.da3_metric.backbone.pretrained.blocks.9.attn | 1 | 0.000324 | 0.000324 | 0.001906 |
| module.model.da3_metric.backbone.pretrained.blocks.18.attn | 1 | 0.000322 | 0.000322 | 0.001913 |
| module.model.da3_metric.backbone.pretrained.blocks.14.attn | 1 | 0.000322 | 0.000322 | 0.001908 |
| module.model.da3_metric.backbone.pretrained.blocks.11.attn | 1 | 0.000321 | 0.000321 | 0.001902 |
| module.model.da3.backbone.pretrained.blocks.3.attn | 1 | 0.000320 | 0.000320 | 0.003662 |
| module.model.da3_metric.backbone.pretrained.blocks.3.attn | 1 | 0.000319 | 0.000319 | 0.001906 |
| module.model.da3_metric.backbone.pretrained.blocks.15.attn | 1 | 0.000318 | 0.000318 | 0.001910 |
| module.model.da3_metric.backbone.pretrained.blocks.5.attn | 1 | 0.000318 | 0.000318 | 0.001904 |
| module.model.da3_metric.backbone.pretrained.blocks.0.mlp | 1 | 0.000316 | 0.000316 | 0.002702 |
| module.model.da3_metric.backbone.pretrained.blocks.7.attn | 1 | 0.000316 | 0.000316 | 0.001903 |
| module.model.da3.backbone.pretrained.blocks.6.attn | 1 | 0.000310 | 0.000310 | 0.003666 |
| module.model.da3_metric.backbone.pretrained.blocks.6.attn | 1 | 0.000309 | 0.000309 | 0.001904 |
| module.model.da3_metric.backbone.pretrained.blocks.10.attn | 1 | 0.000308 | 0.000308 | 0.001904 |
| module.model.da3.backbone.pretrained.blocks.4.attn | 1 | 0.000305 | 0.000305 | 0.003664 |
| module.model.da3.backbone.pretrained.blocks.2.attn | 1 | 0.000302 | 0.000302 | 0.003663 |
| module.model.da3.backbone.pretrained.blocks.12.attn | 1 | 0.000301 | 0.000301 | 0.003672 |
| module.model.da3.backbone.pretrained.blocks.5.attn | 1 | 0.000301 | 0.000301 | 0.003663 |
| module.model.da3.backbone.pretrained.blocks.10.attn | 1 | 0.000296 | 0.000296 | 0.003675 |
| module.model.da3.backbone.pretrained.blocks.11.attn | 1 | 0.000295 | 0.000295 | 0.003675 |
| module.model.da3.backbone.pretrained.blocks.1.attn | 1 | 0.000291 | 0.000291 | 0.003665 |
| module.model.da3.backbone.pretrained.blocks.8.attn | 1 | 0.000280 | 0.000280 | 0.003678 |
| module.model.da3_metric.backbone.pretrained.blocks.12.mlp | 1 | 0.000276 | 0.000276 | 0.002712 |
| module.model.da3_metric.backbone.pretrained.blocks.17.mlp | 1 | 0.000269 | 0.000269 | 0.002713 |
| module.model.da3_metric.backbone.pretrained.blocks.15.mlp | 1 | 0.000267 | 0.000267 | 0.002711 |
| module.model.da3_metric.backbone.pretrained.blocks.3.mlp | 1 | 0.000266 | 0.000266 | 0.002703 |
| module.model.da3_metric.backbone.pretrained.blocks.20.mlp | 1 | 0.000263 | 0.000263 | 0.002712 |
| module.model.da3_metric.backbone.pretrained.blocks.13.mlp | 1 | 0.000260 | 0.000260 | 0.002712 |
| module.model.da3_metric.backbone.pretrained.blocks.1.mlp | 1 | 0.000258 | 0.000258 | 0.002706 |
| module.model.da3_metric.backbone.pretrained.blocks.2.mlp | 1 | 0.000257 | 0.000257 | 0.002702 |
| module.model.da3_metric.backbone.pretrained.blocks.21.mlp | 1 | 0.000255 | 0.000255 | 0.002710 |
| module.model.da3_metric.backbone.pretrained.blocks.19.mlp | 1 | 0.000254 | 0.000254 | 0.002712 |
| module.model.da3_metric.backbone.pretrained.blocks.11.mlp | 1 | 0.000254 | 0.000254 | 0.002701 |
| module.model.da3_metric.backbone.pretrained.blocks.18.mlp | 1 | 0.000252 | 0.000252 | 0.002714 |
| module.model.da3_metric.backbone.pretrained.blocks.22.mlp | 1 | 0.000251 | 0.000251 | 0.002711 |
| module.model.da3_metric.backbone.pretrained.blocks.16.mlp | 1 | 0.000251 | 0.000251 | 0.002711 |
| module.model.da3_metric.backbone.pretrained.blocks.23.mlp | 1 | 0.000249 | 0.000249 | 0.002711 |
| module.model.da3_metric.backbone.pretrained.blocks.6.mlp | 1 | 0.000247 | 0.000247 | 0.002707 |
| module.model.da3_metric.backbone.pretrained.blocks.8.mlp | 1 | 0.000245 | 0.000245 | 0.002700 |
| module.model.da3_metric.backbone.pretrained.blocks.5.mlp | 1 | 0.000245 | 0.000245 | 0.002704 |
| module.model.da3_metric.backbone.pretrained.blocks.4.mlp | 1 | 0.000245 | 0.000245 | 0.002702 |
| module.model.da3_metric.backbone.pretrained.blocks.9.mlp | 1 | 0.000242 | 0.000242 | 0.002703 |
| module.model.da3_metric.backbone.pretrained.blocks.14.mlp | 1 | 0.000242 | 0.000242 | 0.002711 |
| module.model.da3_metric.backbone.pretrained.blocks.10.mlp | 1 | 0.000241 | 0.000241 | 0.002712 |
| module.model.da3_metric.backbone.pretrained.blocks.7.mlp | 1 | 0.000240 | 0.000240 | 0.002703 |
| request.da3._apply_metric_scaling | 1 | 0.000172 | 0.000172 | 0.000222 |

## da3: instrumented_points_export

| 区间 | 次数 | inclusive wall / s | exclusive wall / s | CUDA stream / s |
| --- | --- | --- | --- | --- |
| line.scanner_io.export_prediction:125 scene.export(str(output_glb)) | 1 | 0.470666 | 0.470666 | — |
| line.scanner_io.export_prediction:107 points, colors = backproject( | 1 | 0.204039 | 0.002928 | — |
| line.scanner_io.export_prediction:119 points, colors = points[indices], colors[indices] | 1 | 0.097443 | 0.097443 | — |
| line.scanner_io.export_prediction:138 return { | 1 | 0.073577 | 0.073577 | — |
| line.scanner_io.export_prediction:105 mask, threshold = prediction_mask(prediction) | 1 | 0.055759 | 0.000046 | — |
| points.validity_and_confidence_mask | 1 | 0.055713 | 0.055713 | — |
| line.scanner_io.backproject:88 colors.append(rgb[y, x]) | 20 | 0.050640 | 0.050640 | — |
| line.scanner_io.backproject:86 camera = (pixels @ np.linalg.inv(k).T) * d[y, x, None] | 20 | 0.044374 | 0.044374 | — |
| line.scanner_io.export_prediction:122 cloud = trimesh.points.PointCloud(points.astype(np.float32), colors=colors) | 1 | 0.035284 | 0.035284 | — |
| line.scanner_io.export_prediction:118 indices = np.random.default_rng(seed).choice(before, max_points, replace=False) | 1 | 0.031943 | 0.031943 | — |
| line.scanner_io.backproject:85 pixels = np.column_stack((x, y, np.ones(len(x)))).astype(np.float32) | 20 | 0.030013 | 0.030013 | — |
| line.scanner_io.backproject:84 y, x = np.nonzero(mask & np.isfinite(d) & (d > 0)) | 20 | 0.028204 | 0.028204 | — |
| line.scanner_io.backproject:87 points.append(camera @ pose[:3, :3].T + pose[:3, 3]) | 20 | 0.025901 | 0.025901 | — |
| line.scanner_io.export_prediction:127 loaded = trimesh.load(str(output_glb), force="scene", process=False) | 1 | 0.023516 | 0.023516 | — |
| line.scanner_io.backproject:89 return np.concatenate(points).astype(np.float32), np.concatenate(colors) | 1 | 0.021974 | 0.021974 | — |
| line.scanner_io.export_prediction:121 points = points @ display[:3, :3].T + display[:3, 3] | 1 | 0.010978 | 0.010978 | — |
| line.scanner_io.export_prediction:124 scene.add_geometry(cloud, geom_name=output_glb.stem, node_name=output_glb.stem) | 1 | 0.000229 | 0.000229 | — |
| line.scanner_io.export_prediction:123 scene = trimesh.Scene() | 1 | 0.000105 | 0.000105 | — |
| line.scanner_io.export_prediction:120 display = np.diag([1, -1, -1, 1]).astype(np.float32) @ sensor["w2c"][0] | 1 | 0.000098 | 0.000098 | — |
| line.scanner_io.export_prediction:128 geometries = list(loaded.geometry.values()) | 1 | 0.000008 | 0.000008 | — |
| line.scanner_io.export_prediction:114 before = len(points) | 1 | 0.000008 | 0.000008 | — |
| line.scanner_io.backproject:80 points, colors = [], [] | 1 | 0.000006 | 0.000006 | — |
| line.scanner_io.export_prediction:131 restored = geometries[0] | 1 | 0.000003 | 0.000003 | — |

## promptda: model_load

| 区间 | 次数 | inclusive wall / s | exclusive wall / s | CUDA stream / s |
| --- | --- | --- | --- | --- |
| load.cpu_network_construct_and_checkpoint | 1 | 8.542126 | 0.000919 | — |
| load.dinov2_network_construct | 1 | 7.376584 | 7.376584 | — |
| load.promptda_checkpoint | 1 | 0.894954 | 0.016545 | — |
| load.torch_load_cpu_checkpoint | 1 | 0.778435 | 0.778435 | — |
| load.module_to | 1 | 0.517447 | 0.517447 | — |
| load.dpt_head_construct | 1 | 0.269669 | 0.269669 | — |
| load.state_assignment | 1 | 0.099974 | 0.099974 | — |

## promptda: instrumented_request

| 区间 | 次数 | inclusive wall / s | exclusive wall / s | CUDA stream / s |
| --- | --- | --- | --- | --- |
| line.candidates.predict_promptda:182 with torch.no_grad(), autocast: | 10 | 1.100150 | 0.005230 | — |
| line.candidates.predict_promptda:148 with Image.open(image_path) as source: | 20 | 1.030387 | 0.007748 | — |
| line.candidates.predict_promptda:153 rgb = np.asarray(source.convert("RGB")) | 20 | 1.022639 | 1.022639 | — |
| line.candidates.predict_promptda:191 depths.extend(depth[:, 0, :height, :width].float().cpu().numpy().copy()) | 10 | 0.548086 | 0.015319 | — |
| line.candidates.predict_promptda:183 depth = model.predict(image.to(device), depth_input.to(device)) | 10 | 0.546834 | 0.004905 | — |
| transfer.d2h.depth | 10 | 0.532768 | 0.532768 | 0.016220 |
| request.promptda.predict | 10 | 0.493959 | 0.005917 | 1.012598 |
| request.promptda.rgb_encoder | 10 | 0.390708 | 0.038311 | 0.719334 |
| line.candidates.predict_promptda:154 pixels = rgb.astype(np.float32) / 255 | 20 | 0.189045 | 0.189045 | — |
| line.candidates.predict_promptda:159 colors.append(np.rint(pixels * 255).clip(0, 255).astype(np.uint8)) | 20 | 0.150983 | 0.150983 | — |
| module.depth_head | 10 | 0.094364 | 0.026841 | 0.288133 |
| line.candidates.predict_promptda:156 pixels = cv2.resize( | 20 | 0.066317 | 0.066317 | — |
| transfer.h2d.image | 10 | 0.046013 | 0.046013 | 0.047168 |
| line.candidates.predict_promptda:192 return { | 1 | 0.042575 | 0.042575 | — |
| line.candidates.predict_promptda:160 padded = np.pad( | 20 | 0.039891 | 0.039891 | — |
| line.candidates.predict_promptda:175 image = torch.stack(tensors) | 10 | 0.021511 | 0.021511 | — |
| module.pretrained.blocks.0 | 10 | 0.019115 | 0.006017 | 0.028014 |
| module.pretrained.blocks.1 | 10 | 0.015275 | 0.005071 | 0.028005 |
| module.pretrained.blocks.2 | 10 | 0.014848 | 0.005121 | 0.027981 |
| module.pretrained.blocks.3 | 10 | 0.014763 | 0.005065 | 0.028004 |
| module.pretrained.blocks.7 | 10 | 0.014531 | 0.004980 | 0.027937 |
| module.pretrained.blocks.13 | 10 | 0.014514 | 0.004975 | 0.027871 |
| module.pretrained.blocks.10 | 10 | 0.014511 | 0.005036 | 0.027935 |
| module.pretrained.blocks.20 | 10 | 0.014469 | 0.004972 | 0.027953 |
| module.pretrained.blocks.4 | 10 | 0.014467 | 0.004963 | 0.027950 |
| module.pretrained.blocks.14 | 10 | 0.014467 | 0.004958 | 0.027985 |
| module.pretrained.blocks.23 | 10 | 0.014453 | 0.005022 | 0.027986 |
| module.pretrained.blocks.6 | 10 | 0.014451 | 0.004974 | 0.027961 |
| module.pretrained.blocks.22 | 10 | 0.014441 | 0.005013 | 0.027930 |
| module.pretrained.blocks.9 | 10 | 0.014441 | 0.004927 | 0.027948 |
| module.pretrained.blocks.19 | 10 | 0.014440 | 0.004952 | 0.027975 |
| module.pretrained.blocks.5 | 10 | 0.014436 | 0.004982 | 0.027954 |
| module.pretrained.blocks.16 | 10 | 0.014427 | 0.004942 | 0.027950 |
| module.pretrained.blocks.21 | 10 | 0.014422 | 0.004945 | 0.027959 |
| module.pretrained.blocks.11 | 10 | 0.014374 | 0.004962 | 0.027960 |
| module.pretrained.blocks.12 | 10 | 0.014363 | 0.004907 | 0.027982 |
| module.pretrained.blocks.17 | 10 | 0.014353 | 0.004902 | 0.027962 |
| module.pretrained.blocks.18 | 10 | 0.014348 | 0.004954 | 0.027961 |
| module.pretrained.blocks.15 | 10 | 0.014291 | 0.004886 | 0.027963 |
| module.pretrained.blocks.8 | 10 | 0.014196 | 0.004866 | 0.027967 |
| module.depth_head.scratch.refinenet2 | 10 | 0.012475 | 0.012475 | 0.033516 |
| module.depth_head.scratch.refinenet3 | 10 | 0.012450 | 0.012450 | 0.010446 |
| module.depth_head.scratch.refinenet1 | 10 | 0.012321 | 0.012321 | 0.128599 |
| module.depth_head.scratch.refinenet4 | 10 | 0.010287 | 0.010287 | 0.003905 |
| module.pretrained.blocks.0.attn | 10 | 0.010056 | 0.010056 | 0.014110 |
| module.pretrained.blocks.1.attn | 10 | 0.007389 | 0.007389 | 0.014139 |
| module.pretrained.blocks.2.attn | 10 | 0.006899 | 0.006899 | 0.014137 |
| module.pretrained.blocks.3.attn | 10 | 0.006889 | 0.006889 | 0.014123 |
| module.pretrained.blocks.13.attn | 10 | 0.006803 | 0.006803 | 0.014148 |
| module.pretrained.blocks.7.attn | 10 | 0.006794 | 0.006794 | 0.014148 |
| module.pretrained.blocks.4.attn | 10 | 0.006771 | 0.006771 | 0.014128 |
| module.pretrained.blocks.21.attn | 10 | 0.006765 | 0.006765 | 0.014141 |
| module.pretrained.blocks.19.attn | 10 | 0.006760 | 0.006760 | 0.014143 |
| module.pretrained.blocks.10.attn | 10 | 0.006742 | 0.006742 | 0.014130 |
| module.pretrained.blocks.17.attn | 10 | 0.006736 | 0.006736 | 0.014133 |
| module.pretrained.blocks.9.attn | 10 | 0.006733 | 0.006733 | 0.014132 |
| module.pretrained.blocks.6.attn | 10 | 0.006730 | 0.006730 | 0.014142 |
| module.pretrained.blocks.22.attn | 10 | 0.006712 | 0.006712 | 0.014129 |
| module.pretrained.blocks.14.attn | 10 | 0.006712 | 0.006712 | 0.014158 |
| module.pretrained.blocks.20.attn | 10 | 0.006708 | 0.006708 | 0.014142 |
| module.pretrained.blocks.23.attn | 10 | 0.006707 | 0.006707 | 0.014142 |
| module.pretrained.blocks.5.attn | 10 | 0.006699 | 0.006699 | 0.014146 |
| module.pretrained.blocks.12.attn | 10 | 0.006699 | 0.006699 | 0.014130 |
| module.pretrained.blocks.16.attn | 10 | 0.006697 | 0.006697 | 0.014150 |
| module.pretrained.blocks.11.attn | 10 | 0.006697 | 0.006697 | 0.014128 |
| module.pretrained.blocks.15.attn | 10 | 0.006685 | 0.006685 | 0.014140 |
| module.pretrained.blocks.8.attn | 10 | 0.006630 | 0.006630 | 0.014137 |
| module.pretrained.blocks.18.attn | 10 | 0.006614 | 0.006614 | 0.014132 |
| line.candidates.predict_promptda:128 prompt = np.where( | 1 | 0.003627 | 0.003627 | — |
| module.pretrained.blocks.0.mlp | 10 | 0.003042 | 0.003042 | 0.011175 |
| module.depth_head.scratch.output_conv2 | 10 | 0.003026 | 0.003026 | 0.027159 |
| module.pretrained.blocks.2.mlp | 10 | 0.002828 | 0.002828 | 0.011152 |
| module.pretrained.blocks.1.mlp | 10 | 0.002816 | 0.002816 | 0.011186 |
| module.pretrained.blocks.3.mlp | 10 | 0.002809 | 0.002809 | 0.011179 |
| module.pretrained.blocks.14.mlp | 10 | 0.002797 | 0.002797 | 0.011174 |
| module.pretrained.blocks.20.mlp | 10 | 0.002789 | 0.002789 | 0.011183 |
| module.pretrained.blocks.16.mlp | 10 | 0.002787 | 0.002787 | 0.011163 |
| module.pretrained.blocks.9.mlp | 10 | 0.002781 | 0.002781 | 0.011177 |
| module.pretrained.blocks.18.mlp | 10 | 0.002780 | 0.002780 | 0.011181 |
| module.pretrained.blocks.12.mlp | 10 | 0.002758 | 0.002758 | 0.011181 |
| module.pretrained.blocks.7.mlp | 10 | 0.002757 | 0.002757 | 0.011130 |
| module.pretrained.blocks.5.mlp | 10 | 0.002755 | 0.002755 | 0.011172 |
| module.pretrained.blocks.6.mlp | 10 | 0.002748 | 0.002748 | 0.011172 |
| module.pretrained.blocks.13.mlp | 10 | 0.002736 | 0.002736 | 0.011050 |
| module.pretrained.blocks.10.mlp | 10 | 0.002733 | 0.002733 | 0.011160 |
| module.pretrained.blocks.4.mlp | 10 | 0.002733 | 0.002733 | 0.011172 |
| module.pretrained.blocks.19.mlp | 10 | 0.002728 | 0.002728 | 0.011177 |
| module.pretrained.blocks.23.mlp | 10 | 0.002724 | 0.002724 | 0.011174 |
| module.pretrained.blocks.15.mlp | 10 | 0.002720 | 0.002720 | 0.011159 |
| module.pretrained.blocks.22.mlp | 10 | 0.002716 | 0.002716 | 0.011175 |
| module.pretrained.blocks.11.mlp | 10 | 0.002715 | 0.002715 | 0.011172 |
| module.pretrained.blocks.17.mlp | 10 | 0.002714 | 0.002714 | 0.011193 |
| module.pretrained.blocks.21.mlp | 10 | 0.002711 | 0.002711 | 0.011172 |
| module.pretrained.blocks.8.mlp | 10 | 0.002700 | 0.002700 | 0.011178 |
| request.promptda.lidar_minmax_normalize | 10 | 0.002343 | 0.002343 | 0.002845 |
| module.depth_head.projects.0 | 10 | 0.002085 | 0.002085 | 0.001120 |
| transfer.h2d.depth_input | 10 | 0.001958 | 0.001958 | 0.002410 |
| module.depth_head.resize_layers.0 | 10 | 0.001922 | 0.001922 | 0.002502 |
| line.candidates.predict_promptda:163 tensors.append(torch.from_numpy(padded).permute(2, 0, 1)) | 20 | 0.001467 | 0.001467 | — |
| module.depth_head.resize_layers.3 | 10 | 0.001465 | 0.001465 | 0.004056 |
| module.depth_head.projects.2 | 10 | 0.001451 | 0.001451 | 0.002026 |
| module.depth_head.projects.3 | 10 | 0.001418 | 0.001418 | 0.001931 |
| module.depth_head.resize_layers.1 | 10 | 0.001411 | 0.001411 | 0.001657 |
| module.depth_head.projects.1 | 10 | 0.001398 | 0.001398 | 0.001524 |
| module.depth_head.scratch.output_conv1 | 10 | 0.001276 | 0.001276 | 0.022558 |
| module.depth_head.scratch.layer4_rn | 10 | 0.001152 | 0.001152 | 0.001466 |
| module.depth_head.scratch.layer3_rn | 10 | 0.001146 | 0.001146 | 0.003780 |
| line.candidates.predict_promptda:176 depth_input = torch.cat(prompts) | 10 | 0.001115 | 0.001115 | — |
| module.depth_head.scratch.layer1_rn | 10 | 0.001097 | 0.001097 | 0.010435 |
| module.depth_head.scratch.layer2_rn | 10 | 0.001039 | 0.001039 | 0.006951 |
| line.candidates.predict_promptda:177 autocast = ( | 10 | 0.000948 | 0.000948 | — |
| request.promptda.metric_denormalize | 10 | 0.000628 | 0.000628 | 0.000164 |
| line.candidates.predict_promptda:142 tensors, prompts = [], [] | 10 | 0.000251 | 0.000251 | — |
| line.candidates.predict_promptda:164 depth_input = torch.from_numpy(native_depth[None, None]) | 20 | 0.000229 | 0.000229 | — |
| module.depth_head.resize_layers.2 | 10 | 0.000103 | 0.000103 | 0.000016 |
| line.candidates.predict_promptda:174 prompts.append(depth_input) | 20 | 0.000074 | 0.000074 | — |
| line.candidates.predict_promptda:138 width, height = promptda_grid_size(source_size, max_size) | 1 | 0.000035 | 0.000035 | — |
| line.candidates.predict_promptda:137 source_size = tuple(map(int, sensor["image_size"])) | 1 | 0.000031 | 0.000031 | — |
| line.candidates.predict_promptda:139 pad_width, pad_height = -width % 14, -height % 14 | 1 | 0.000006 | 0.000006 | — |
| line.candidates.predict_promptda:140 depths, colors = [], [] | 1 | 0.000005 | 0.000005 | — |

## promptda: instrumented_points_export

| 区间 | 次数 | inclusive wall / s | exclusive wall / s | CUDA stream / s |
| --- | --- | --- | --- | --- |
| line.scanner_io.export_prediction:107 points, colors = backproject( | 1 | 1.311909 | 0.004522 | — |
| line.scanner_io.export_prediction:125 scene.export(str(output_glb)) | 1 | 0.415129 | 0.415129 | — |
| line.scanner_io.backproject:88 colors.append(rgb[y, x]) | 20 | 0.337411 | 0.337411 | — |
| line.scanner_io.backproject:86 camera = (pixels @ np.linalg.inv(k).T) * d[y, x, None] | 20 | 0.274712 | 0.274712 | — |
| line.scanner_io.backproject:85 pixels = np.column_stack((x, y, np.ones(len(x)))).astype(np.float32) | 20 | 0.264032 | 0.264032 | — |
| line.scanner_io.backproject:87 points.append(camera @ pose[:3, :3].T + pose[:3, 3]) | 20 | 0.158395 | 0.158395 | — |
| line.scanner_io.backproject:89 return np.concatenate(points).astype(np.float32), np.concatenate(colors) | 1 | 0.156758 | 0.156758 | — |
| line.scanner_io.export_prediction:119 points, colors = points[indices], colors[indices] | 1 | 0.125446 | 0.125446 | — |
| line.scanner_io.backproject:84 y, x = np.nonzero(mask & np.isfinite(d) & (d > 0)) | 20 | 0.116073 | 0.116073 | — |
| line.scanner_io.export_prediction:138 return { | 1 | 0.063489 | 0.063489 | — |
| line.scanner_io.export_prediction:118 indices = np.random.default_rng(seed).choice(before, max_points, replace=False) | 1 | 0.053941 | 0.053941 | — |
| line.scanner_io.export_prediction:122 cloud = trimesh.points.PointCloud(points.astype(np.float32), colors=colors) | 1 | 0.029507 | 0.029507 | — |
| line.scanner_io.export_prediction:105 mask, threshold = prediction_mask(prediction) | 1 | 0.019597 | 0.000029 | — |
| points.validity_and_confidence_mask | 1 | 0.019568 | 0.019568 | — |
| line.scanner_io.export_prediction:127 loaded = trimesh.load(str(output_glb), force="scene", process=False) | 1 | 0.014738 | 0.014738 | — |
| line.scanner_io.export_prediction:121 points = points @ display[:3, :3].T + display[:3, 3] | 1 | 0.011956 | 0.011956 | — |
| line.scanner_io.export_prediction:124 scene.add_geometry(cloud, geom_name=output_glb.stem, node_name=output_glb.stem) | 1 | 0.000169 | 0.000169 | — |
| line.scanner_io.export_prediction:120 display = np.diag([1, -1, -1, 1]).astype(np.float32) @ sensor["w2c"][0] | 1 | 0.000110 | 0.000110 | — |
| line.scanner_io.export_prediction:123 scene = trimesh.Scene() | 1 | 0.000046 | 0.000046 | — |
| line.scanner_io.export_prediction:128 geometries = list(loaded.geometry.values()) | 1 | 0.000007 | 0.000007 | — |
| line.scanner_io.export_prediction:114 before = len(points) | 1 | 0.000006 | 0.000006 | — |
| line.scanner_io.backproject:80 points, colors = [], [] | 1 | 0.000004 | 0.000004 | — |
| line.scanner_io.export_prediction:131 restored = geometries[0] | 1 | 0.000003 | 0.000003 | — |
