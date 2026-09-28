# One Sensor, Whole Body

*3D Body Pose from a Single Consumer Earbud IMU*

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg?style=flat-square)](#citation)

**[Zhilin Guo](https://zhilinguo.github.io/)¹, [Boqiao Zhang](https://boqiaoz00.github.io/boqiao_steven_zhang.github.io/)¹, Oszkár Urbán¹, [Josef Bengtson](https://www.chalmers.se/en/persons/bjosef/)², [Hakan Aktas](https://scholar.google.com/citations?user=RxjN5w4AAAAJ&hl=en)¹, [Wenzhao Li](https://wenzhao-cam.github.io/)¹, [Siyu Hong](https://www.linkedin.com/in/siyuhong/)¹, [Kyle Fogarty](https://kyle-fogarty.github.io/)¹, [Chenliang Zhou](https://chenliang-zhou.github.io/)¹, Ali Senguel¹, [Cengiz Oztireli](https://sites.google.com/view/cengiz-oztireli-intro/home)¹**

¹ University of Cambridge &nbsp;&nbsp;·&nbsp;&nbsp; ² Chalmers University of Technology

> Consumer earbuds already stream inertial motion data from the head, one of the most widely worn sensor locations on the body. We ask how much of the 3D body pose a single such head IMU can recover, and whether adding more consumer sensors actually helps. We build a multimodal capture pipeline that records four-view RGB-D video together with an AirPods head IMU and two Striv insole IMUs, synchronize the streams post-hoc, and generate pseudo-ground-truth with SAM 3D Body, yielding a 35-take single-subject benchmark spanning gait, turning, vertical, everyday, and clinically inspired motions. Adapting two recurrent model families (IMUPoser and MobilePoser), we show that one head IMU recovers lower-body pose at 79.0 mm rigid-MPJPE and per-foot ground contact at 0.809 macro-F1, and that a causal variant retains most of this accuracy at streaming latency. In paired per-take significance tests across both families, adding the consumer foot IMUs never significantly improves pose and significantly degrades it in two of four model–split combinations; a mounting-bias probe and feet-only ablation identify insole orientation quality, not foot placement, as the mechanism. Extending the output to a 20-joint full-body skeleton maps the boundary: gross distal-arm motion is partially recoverable from the head alone, proximal upper-body pose is not, and staged fine-tuning recovers the leg accuracy that naive joint training sacrifices to multi-task dilution. For learned pose from consumer wearables, sensor reliability, not sensor count, is the binding constraint here. For the devices tested, the earbud is its sweet spot.

![Adding consumer foot IMUs never significantly improves pose](images/teaser.png)

**Adding consumer foot IMUs never significantly improves pose — and often significantly degrades it.** *Paired per-take head-only (x) vs. head+feet (y) rigid-MPJPE across both model families and evaluation splits; points above the diagonal favour head-only.*

## Code coming soon

We are preparing the public code release. Watch this repository for updates.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{guo2026onesensor,
  title     = {One Sensor, Whole Body --- 3D Body Pose from a Single Consumer Earbud IMU},
  author    = {Guo, Zhilin and Zhang, Boqiao and Urb{\'a}n, Oszk{\'a}r and Bengtson, Josef and Aktas, Hakan and Li, Wenzhao and Hong, Siyu and Fogarty, Kyle and Zhou, Chenliang and Senguel, Ali and Oztireli, Cengiz},
  booktitle = {The 6th International Workshop on Human-centric Multimedia Analysis (HUMA '26), ACM Multimedia},
  year      = {2026},
  doi       = {10.1145/3841192.3841753}
}
```

## License

This project is licensed under the Apache License 2.0, as found in the [LICENSE](LICENSE) file.
