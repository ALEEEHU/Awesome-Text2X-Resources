# Awesome Text2X Resources
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FALEEEHU%2FAwesome-Text2X-Resources%2F&count_bg=%23EAA8EA&title_bg=%233D2549&icon=react.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=true)](https://hits.seeyoufarm.com)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-pink.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-pink) ![Stars](https://img.shields.io/github/stars/ALEEEHU/Awesome-Text2X-Resources)

This is an open collection of state-of-the-art (SOTA), novel **Text to X (X can be everything)** methods (papers, codes and datasets), intended to keep pace with the anticipated surge of research in the coming months. 

⭐ If you find this repository useful to your research or work, it is really appreciated to star this repository. 

💗 Continual improvements are being made to this repository. If you come across any relevant papers that should be included, please don't hesitate to submit a pull request (PR) or open an issue. Additional resources like blog posts, videos, etc. are also welcome. 

✉️ Any additions or suggestions, feel free to contribute and contact hyqale1024@gmail.com. 

## 🔥 News
* `2024.10.30` adjusted the layout of the `Text to 4D` section.
* `2024.04.05` adjusted the layout and added accepted lists and ArXiv lists to each section.


<div><div align="center">
	<img width="500" height="350" src="media/logo.svg" alt="Awesome"></div>

## Table of Contents

- [Text to 4D](#text-to-4d)
  * [Accepted Papers](#-4d-accepted-papers)
  * [ArXiv Papers](#-4d-arxiv-papers)
  * [Additional Info](#other-4d-additional-info)
- [Text to Scene](#text-to-scene)
  * [Accepted Papers](#-scene-accepted-papers)
  * [ArXiv Papers](#-scene-arxiv-papers)
- [Text to Human Motion](#text-to-human-motion)
  * [Accepted Papers](#-motion-accepted-papers)
  * [ArXiv Papers](#-motion-arxiv-papers)
  * [Datasets](#datasets)
- [Text to 3D Human](#text-to-3d-human)
  * [Accepted Papers](#-human-accepted-papers)
  * [ArXiv Papers](#-human-arxiv-papers)
- [Text to Video](#text-to-video)
  * [Accepted Papers](#-video-accepted-papers)
  * [ArXiv Papers](#-video-arxiv-papers)
  * [Additional Info](#other-additional-info)
- [Related Resources](#related-resources)
  * [Text to Other Tasks](#text-to-other-tasks)
  * [Survey and Awesome Repos](#survey-and-awesome-repos)

## Update Logs
<details span>
<summary><b>Update Logs:</b></summary>
<br>
	
* `2024.09.26` - update several papers status "NeurIPS 2024" to accepted papers, congrats to all 🎉
* `2024.09.03` - add one new section 'text to model'.
* `2024.06.30` - add one new section 'text to video'.	
* `2024.07.02` - update several papers status "ECCV 2024" to accepted papers, congrats to all 🎉
* `2024.06.21` - add one hot Topic about _AIGC 4D Generation_ on the section of __Suvery and Awesome Repos__.
* `2024.06.17` - an awesome repo for CVPR2024 [Link](https://github.com/52CV/CVPR-2024-Papers) 👍🏻
* `2024.04.05` - an awesome repo for CVPR2024 on 3DGS and NeRF [Link](https://github.com/Yubel426/NeRF-3DGS-at-CVPR-2024) 👍🏻
* `2024.03.25` - add one new survey paper of 3D GS into the section of "Survey and Awesome Repos--Topic 1: 3D Gaussian Splatting".
* `2024.03.12` - add a new section "Dynamic Gaussian Splatting", including Neural Deformable 3D Gaussians, 4D Gaussians, Dynamic 3D Gaussians.
* `2024.03.11` - CVPR 2024 Accpeted Papers [Link](https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers) 
* update some papers accepted by CVPR 2024! Congratulations🎉
  
</details>
<br>

## Text to 4D
(Also, Image to 4D)

### 🎉 4D Accepted Papers



| Task | Year | Title             | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ---- | ----------------- | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :-------------------------------: |
|<img src="https://img.shields.io/badge/4D Scene-FFFF93" /> | 2023 | **Text-To-4D Dynamic Scene Generation**  | ICML 2023 |          [Link](https://arxiv.org/abs/2301.11280)          | --  | [Link](https://make-a-video3d.github.io/)  |
|<img src="https://img.shields.io/badge/4D Scene-FFFF93" /> | 2023 |  **4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2311.17984)          | [Link](https://github.com/sherwinbahmani/4dfy)  | [Link](https://sherwinbahmani.github.io/4dfy/)  |
|<img src="https://img.shields.io/badge/4D Scene-FFFF93" /> | 2023 |  **Dream-in-4D: A Unified Approach for Text- and Image-guided 4D Scene Generation**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2311.16854)          | [Link](https://github.com/NVlabs/dream-in-4d)  | [Link](https://research.nvidia.com/labs/nxp/dream-in-4d/)  |
|<img src="https://img.shields.io/badge/4D Scene-FFFF93" /> | 2023 |  **Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2312.13763)          | -- | [Link](https://research.nvidia.com/labs/toronto-ai/AlignYourGaussians/)  |
|<img src="https://img.shields.io/badge/4D Scene-FFFF93" /> | 2024 | **TC4D: Trajectory-Conditioned Text-to-4D Generation**  | ECCV 2024 |         [Link](https://arxiv.org/abs/2403.17920)         | [Link](https://github.com/sherwinbahmani/tc4d)   | [Link](https://sherwinbahmani.github.io/tc4d/) |
|<img src="https://img.shields.io/badge/4D Scene-FFFF93" /> | 2024 | **4Real: Towards Photorealistic 4D Scene Generation via Video Diffusion Models**  | NeurIPS 2024 |          [Link](https://arxiv.org/abs/2406.07472)          | -- | [Link](https://snap-research.github.io/4Real/) |
|<img src="https://img.shields.io/badge/4D Scene-FFFF93" /> | 2024 | **Compositional 3D-aware Video Generation with LLM Director**  | NeurIPS 2024  | [Link](https://arxiv.org/abs/2409.00558) |     --   | [Link](https://www.microsoft.com/en-us/research/project/compositional-3d-aware-video-generation/) |
|<img src="https://img.shields.io/badge/Video%20to%204D-CCFF80" />| 2023 | **Consistent4D: Consistent 360° Dynamic Object Generation from Monocular Video**  | ICLR 2024 |          [Link](https://arxiv.org/abs/2311.02848)          | [Link](https://github.com/yanqinJiang/Consistent4D)  | [Link](https://consistent4d.github.io/)  |
|<img src="https://img.shields.io/badge/Video%20to%204D-CCFF80" />| 2023 | **SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer**  | ECCV 2024  |          [Link](https://arxiv.org/abs/2404.03736)          | [Link](https://github.com/JarrentWu1031/SC4D)  |[Link](https://sc4d.github.io/) |
|<img src="https://img.shields.io/badge/Video%20to%204D-CCFF80" />| 2023 | **STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians**  | ECCV 2024  |           [Link](https://arxiv.org/abs/2403.14939) | [Link](https://github.com/zeng-yifei/STAG4D)  |[Link](https://nju-3dv.github.io/projects/STAG4D/) |
|<img src="https://img.shields.io/badge/Video%20to%204D-CCFF80" />| 2024 | **Animate3D: Animating Any 3D Model with Multi-view Video Diffusion**  | NeurIPS 2024 |          [Link](https://arxiv.org/abs/2407.11398)          | [Link](https://github.com/yanqinJiang/Animate3D) | [Link](https://animate3d.github.io/) |
|<img src="https://img.shields.io/badge/Video%20to%204D-CCFF80" />| 2024 | **Vidu4D: Single Generated Video to High-Fidelity 4D Reconstruction with Dynamic Gaussian Surfels**  | NeurIPS 2024 |       [Link](https://arxiv.org/abs/2405.16822)        | [Link](https://github.com/yikaiw/vidu4d)  | [Link](https://vidu4d-dgs.github.io/)  |
|<img src="https://img.shields.io/badge/Video%20to%204D-CCFF80" />| 2024 | **4Diffusion: Multi-view Video Diffusion Model for 4D Generation**  | NeurIPS 2024 |       [Link](https://arxiv.org/abs/2405.20674)        | [Link](https://github.com/aejion/4Diffusion)  | [Link](https://aejion.github.io/4diffusion/)  |
|<img src="https://img.shields.io/badge/Video%20to%204D-CCFF80" />| 2024 | **DreamMesh4D: Video-to-4D Generation with Sparse-Controlled Gaussian-Mesh Hybrid Representation**  | NeurIPS 2024  | [Link](https://arxiv.org/abs/2410.06756) |         [Link](https://github.com/WU-CVGL/DreamMesh4D)         | [Link](https://lizhiqi49.github.io/DreamMesh4D/) |
|<img src="https://img.shields.io/badge/Video%20to%204D-CCFF80" />| 2024 | **DreamScene4D: Dynamic Multi-Object Scene Generation from Monocular Videos**  | NeurIPS 2024 |          [Link](https://arxiv.org/abs/2405.02280)          | [Link](https://github.com/dreamscene4d/dreamscene4d)  |[Link](https://dreamscene4d.github.io/) |
|<img src="https://img.shields.io/badge/4D Human-FFC78E" /> | 2023 |  **Control4D: Efficient 4D Portrait Editing with Text**  | CVPR 2024 |    [Link](https://arxiv.org/abs/2305.20082)          | --  | [Link](https://control4darxiv.github.io./)  |


<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

%4d scene
@article{singer2023text4d,
  author = {Singer, Uriel and Sheynin, Shelly and Polyak, Adam and Ashual, Oron and
           Makarov, Iurii and Kokkinos, Filippos and Goyal, Naman and Vedaldi, Andrea and
           Parikh, Devi and Johnson, Justin and Taigman, Yaniv},
  title = {Text-To-4D Dynamic Scene Generation},
  journal = {arXiv:2301.11280},
  year = {2023},
}

@article{bah20244dfy,
  author = {Bahmani, Sherwin and Skorokhodov, Ivan and Rong, Victor and Wetzstein, Gordon and Guibas, Leonidas and Wonka, Peter and Tulyakov, Sergey and Park, Jeong Joon and Tagliasacchi, Andrea and Lindell, David B.},
  title = {4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling},
  journal = {IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  year = {2024},
}

@InProceedings{zheng2024unified,
    title     = {A Unified Approach for Text- and Image-guided 4D Scene Generation},
    author    = {Yufeng Zheng and Xueting Li and Koki Nagano and Sifei Liu and Otmar Hilliges and Shalini De Mello},
    booktitle = {CVPR},
    year      = {2024}
}

@article{ling2023alignyourgaussians,
    title={Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models},
    author={Ling, Huan and Kim, Seung Wook and Torralba, Antonio and Fidler, Sanja and Kreis, Karsten},
    title={arXiv preprint arXiv:2312.13763},
    year={2023}
}

@article{bah2024tc4d,
  author = {Bahmani, Sherwin and Liu, Xian and Yifan, Wang and Skorokhodov, Ivan and Rong, Victor and Liu, Ziwei and Liu, Xihui and Park, Jeong Joon and Tulyakov, Sergey and Wetzstein, Gordon and Tagliasacchi, Andrea and Lindell, David B.},
  title = {TC4D: Trajectory-Conditioned Text-to-4D Generation},
  journal = {arXiv},
  year = {2024},
}

@misc{yu20244real,
      title={4Real: Towards Photorealistic 4D Scene Generation via Video Diffusion Models}, 
      author={Heng Yu and Chaoyang Wang and Peiye Zhuang and Willi Menapace and Aliaksandr Siarohin and Junli Cao and Laszlo A Jeni and Sergey Tulyakov and Hsin-Ying Lee},
      year={2024},
      eprint={2406.07472},
      archivePrefix={arXiv},
      primaryClass={id='cs.CV' full_name='Computer Vision and Pattern Recognition' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers image processing, computer vision, pattern recognition, and scene understanding. Roughly includes material in ACM Subject Classes I.2.10, I.4, and I.5.'}
}

@misc{zhu2024compositional3dawarevideogeneration,
      title={Compositional 3D-aware Video Generation with LLM Director}, 
      author={Hanxin Zhu and Tianyu He and Anni Tang and Junliang Guo and Zhibo Chen and Jiang Bian},
      year={2024},
      eprint={2409.00558},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.00558}, 
}

% video to 4d
@inproceedings{
jiang2024consistentd,
title={Consistent4D: Consistent 360{\textdegree} Dynamic Object Generation from Monocular Video},
author={Yanqin Jiang and Li Zhang and Jin Gao and Weiming Hu and Yao Yao},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=sPUrdFGepF}
}

@article{wu2024sc4d,
    author = {Wu, Zijie and Yu, Chaohui and Jiang, Yanqin and Cao, Chenjie and Wang Fan and Bai, Xiang.},
    title  = {SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer},
    journal = {arxiv:2404.03736},
    year   = {2024}
}

@article{zeng2024stag4d,
  title={Stag4d: Spatial-temporal anchored generative 4d gaussians},
  author={Zeng, Yifei and Jiang, Yanqin and Zhu, Siyu and Lu, Yuanxun and Lin, Youtian and Zhu, Hao and Hu, Weiming and Cao, Xun and Yao, Yao},
  journal={arXiv preprint arXiv:2403.14939},
  year={2024}
}

@article{
jiang2024animate3d,
title={Animate3D: Animating Any 3D Model with Multi-view Video Diffusion},
author={Yanqin Jiang and Chaohui Yu and Chenjie Cao and Fan Wang and Weiming Hu and Jin Gao},
booktitle={arXiv},
year={2024},
}

@article{wang2024vidu4d,
  title={Vidu4D: Single Generated Video to High-Fidelity 4D Reconstruction with Dynamic Gaussian Surfels},
  author={Yikai Wang and Xinzhou Wang and Zilong Chen and Zhengyi Wang and Fuchun Sun and Jun Zhu},
  journal={arXiv preprint arXiv},
  year={2024}
}

@article{zhang20244diffusion,
    title={4Diffusion: Multi-view Video Diffusion Model for 4D Generation}, 
    author={Haiyu Zhang and Xinyuan Chen and Yaohui Wang and Xihui Liu and Yunhong Wang and Yu Qiao},
    year={2024}
}

@inproceedings{li2024dreammesh4d,
    title={DreamMesh4D: Video-to-4D Generation with Sparse-Controlled Gaussian-Mesh Hybrid Representation},
    author={Zhiqi Li and Yiming Chen and Peidong Liu},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2024}
}

@inproceedings{dreamscene4d,
  title={DreamScene4D: Dynamic Multi-Object Scene Generation from Monocular Videos},
  author={Chu, Wen-Hsuan and Ke, Lei and Fragkiadaki, Katerina},
  booktitle={NeurIPS},
  year={2024}
}

% 4d human
@article{shao2023control4d,
title = {Control4D: Efficient 4D Portrait Editing with Text},
author = {Shao, Ruizhi and Sun, Jingxiang and Peng, Cheng and Zheng, Zerong and Zhou, Boyao and Zhang, Hongwen and Liu, Yebin},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
year = {2024}
}
```
</details>

---

### 💡 4D ArXiv Papers

#### 1. Animate124: Animating One Image to 4D Dynamic Scene
Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, Gim Hee Lee

(National University of Singapore, Huawei Noah's Ark Lab)
<details span>
<summary><b>Abstract</b></summary>
We introduce Animate124 (Animate-one-image-to-4D), the first work to animate a single in-the-wild image into 3D video through textual motion descriptions, an underexplored problem with significant applications. Our 4D generation leverages an advanced 4D grid dynamic Neural Radiance Field (NeRF) model, optimized in three distinct stages using multiple diffusion priors. Initially, a static model is optimized using the reference image, guided by 2D and 3D diffusion priors, which serves as the initialization for the dynamic NeRF. Subsequently, a video diffusion model is employed to learn the motion specific to the subject. However, the object in the 3D videos tends to drift away from the reference image over time. This drift is mainly due to the misalignment between the text prompt and the reference image in the video diffusion model. In the final stage, a personalized diffusion prior is therefore utilized to address the semantic drift. As the pioneering image-text-to-4D generation framework, our method demonstrates significant advancements over existing baselines, evidenced by comprehensive quantitative and qualitative assessments.
</details>

#### 2. 4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency
Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, Yunchao Wei

(Beijing Jiaotong University, University of Texas at Austin)
<details span>
<summary><b>Abstract</b></summary>
Aided by text-to-image and text-to-video diffusion models, existing 4D content creation pipelines utilize score distillation sampling to optimize the entire dynamic 3D scene. However, as these pipelines generate 4D content from text or image inputs, they incur significant time and effort in prompt engineering through trial and error. This work introduces 4DGen, a novel, holistic framework for grounded 4D content creation that decomposes the 4D generation task into multiple stages. We identify static 3D assets and monocular video sequences as key components in constructing the 4D content. Our pipeline facilitates conditional 4D generation, enabling users to specify geometry (3D assets) and motion (monocular videos), thus offering superior control over content creation. Furthermore, we construct our 4D representation using dynamic 3D Gaussians, which permits efficient, high-resolution supervision through rendering during training, thereby facilitating high-quality 4D generation. Additionally, we employ spatial-temporal pseudo labels on anchor frames, along with seamless consistency priors implemented through 3D-aware score distillation sampling and smoothness regularizations. Compared to existing baselines, our approach yields competitive results in faithfully reconstructing input signals and realistically inferring renderings from novel viewpoints and timesteps. Most importantly, our method supports grounded generation, offering users enhanced control, a feature difficult to achieve with previous methods.
</details>

#### 3. DreamGaussian4D: Generative 4D Gaussian Splatting
Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, Ziwei Liu

(S-Lab, Nanyang Technological University, Shanghai AI Laboratory, Peking University, University of Michigan)
<details span>
<summary><b>Abstract</b></summary>
Remarkable progress has been made in 4D content generation recently. However, existing methods suffer from long optimization time, lack of motion controllability, and a low level of detail. In this paper, we introduce DreamGaussian4D, an efficient 4D generation framework that builds on 4D Gaussian Splatting representation. Our key insight is that the explicit modeling of spatial transformations in Gaussian Splatting makes it more suitable for the 4D generation setting compared with implicit representations. DreamGaussian4D reduces the optimization time from several hours to just a few minutes, allows flexible control of the generated 3D motion, and produces animated meshes that can be efficiently rendered in 3D engines.
</details>

#### 4. Fast Dynamic 3D Object Generation from a Single-view Video
Zijie Pan, Zeyu Yang, Xiatian Zhu, Li Zhang (Fudan University, University of Surrey)
<details span>
<summary><b>Abstract</b></summary>
Generating dynamic three-dimensional (3D) object from a single-view video is challenging due to the lack of 4D labeled data. Existing methods extend text-to-3D pipelines by transferring off-the-shelf image generation models such as score distillation sampling, but they are slow and expensive to scale (e.g., 150 minutes per object) due to the need for back-propagating the information-limited supervision signals through a large pretrained model. To address this limitation, we propose an efficient video-to-4D object generation framework called Efficient4D. It generates high-quality spacetime-consistent images under different camera views, and then uses them as labeled data to directly train a novel 4D Gaussian splatting model with explicit point cloud geometry, enabling real-time rendering under continuous camera trajectories. Extensive experiments on synthetic and real videos show that Efficient4D offers a remarkable 10-fold increase in speed when compared to prior art alternatives while preserving the same level of innovative view synthesis quality. For example, Efficient4D takes only 14 minutes to model a dynamic object.
</details>

#### 5. GaussianFlow: Splatting Gaussian Dynamics for 4D Content Creation
Quankai Gao, Qiangeng Xu, Zhe Cao, Ben Mildenhall, Wenchao Ma, Le Chen, Danhang Tang, Ulrich Neumann

(University of Southern California, Google, Pennsylvania State University, Max Planck Institute for Intelligent Systems)
<details span>
<summary><b>Abstract</b></summary>
Creating 4D fields of Gaussian Splatting from images or videos is a challenging task due to its under-constrained nature. While the optimization can draw photometric reference from the input videos or be regulated by generative models, directly supervising Gaussian motions remains underexplored. In this paper, we introduce a novel concept, Gaussian flow, which connects the dynamics of 3D Gaussians and pixel velocities between consecutive frames. The Gaussian flow can be efficiently obtained by splatting Gaussian dynamics into the image space. This differentiable process enables direct dynamic supervision from optical flow. Our method significantly benefits 4D dynamic content generation and 4D novel view synthesis with Gaussian Splatting, especially for contents with rich motions that are hard to be handled by existing methods. The common color drifting issue that happens in 4D generation is also resolved with improved Guassian dynamics. Superior visual quality on extensive experiments demonstrates our method's effectiveness. Quantitative and qualitative evaluations show that our method achieves state-of-the-art results on both tasks of 4D generation and 4D novel view synthesis.
</details>

#### 6. Comp4D: LLM-Guided Compositional 4D Scene Generation
Dejia Xu, Hanwen Liang, Neel P. Bhatt, Hezhen Hu, Hanxue Liang, Konstantinos N. Plataniotis, Zhangyang Wang

(University of Texas at Austin, University of Toronto, University of Cambridge)
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in diffusion models for 2D and 3D content creation have sparked a surge of interest in generating 4D content. However, the scarcity of 3D scene datasets constrains current methodologies to primarily object-centric generation. To overcome this limitation, we present Comp4D, a novel framework for Compositional 4D Generation. Unlike conventional methods that generate a singular 4D representation of the entire scene, Comp4D innovatively constructs each 4D object within the scene separately. Utilizing Large Language Models (LLMs), the framework begins by decomposing an input text prompt into distinct entities and maps out their trajectories. It then constructs the compositional 4D scene by accurately positioning these objects along their designated paths. To refine the scene, our method employs a compositional score distillation technique guided by the pre-defined trajectories, utilizing pre-trained diffusion models across text-to-image, text-to-video, and text-to-3D domains. Extensive experiments demonstrate our outstanding 4D content creation capability compared to prior arts, showcasing superior visual quality, motion fidelity, and enhanced object interactions.
</details>

#### 7. Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models
Hanwen Liang, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N. Plataniotis, Yao Zhao, Yunchao Wei

(University of Toronto, Beijing Jiaotong University, University of Texas at Austin, University of Cambridge)
<details span>
<summary><b>Abstract</b></summary>
The availability of large-scale multimodal datasets and advancements in diffusion models have significantly accelerated progress in 4D content generation. Most prior approaches rely on multiple image or video diffusion models, utilizing score distillation sampling for optimization or generating pseudo novel views for direct supervision. However, these methods are hindered by slow optimization speeds and multi-view inconsistency issues. Spatial and temporal consistency in 4D geometry has been extensively explored respectively in 3D-aware diffusion models and traditional monocular video diffusion models. Building on this foundation, we propose a strategy to migrate the temporal consistency in video diffusion models to the spatial-temporal consistency required for 4D generation. Specifically, we present a novel framework, Diffusion4D, for efficient and scalable 4D content generation. Leveraging a meticulously curated dynamic 3D dataset, we develop a 4D-aware video diffusion model capable of synthesizing orbital views of dynamic 3D assets. To control the dynamic strength of these assets, we introduce a 3D-to-4D motion magnitude metric as guidance. Additionally, we propose a novel motion magnitude reconstruction loss and 3D-aware classifier-free guidance to refine the learning and generation of motion dynamics. After obtaining orbital views of the 4D asset, we perform explicit 4D construction with Gaussian splatting in a coarse-to-fine manner. The synthesized multi-view consistent 4D image set enables us to swiftly generate high-fidelity and diverse 4D assets within just several minutes. Extensive experiments demonstrate that our method surpasses prior state-of-the-art techniques in terms of generation efficiency and 4D geometry consistency across various prompt modalities.
</details>

#### 8. EG4D: Explicit Generation of 4D Object without Score Distillation
Qi Sun, Zhiyang Guo, Ziyu Wan, Jing Nathan Yan, Shengming Yin, Wengang Zhou, Jing Liao, Houqiang Li

(USTC, City University of Hong Kong, Cornell University)
<details span>
<summary><b>Abstract</b></summary>
In recent years, the increasing demand for dynamic 3D assets in design and gaming applications has given rise to powerful generative pipelines capable of synthesizing high-quality 4D objects. Previous methods generally rely on score distillation sampling (SDS) algorithm to infer the unseen views and motion of 4D objects, thus leading to unsatisfactory results with defects like over-saturation and Janus problem. Therefore, inspired by recent progress of video diffusion models, we propose to optimize a 4D representation by explicitly generating multi-view videos from one input image. However, it is far from trivial to handle practical challenges faced by such a pipeline, including dramatic temporal inconsistency, inter-frame geometry and texture diversity, and semantic defects brought by video generation results. To address these issues, we propose DG4D, a novel multi-stage framework that generates high-quality and consistent 4D assets without score distillation. Specifically, collaborative techniques and solutions are developed, including an attention injection strategy to synthesize temporal-consistent multi-view videos, a robust and efficient dynamic reconstruction method based on Gaussian Splatting, and a refinement stage with diffusion prior for semantic restoration. The qualitative results and user preference study demonstrate that our framework outperforms the baselines in generation quality by a considerable margin.
</details>

#### 9. PLA4D: Pixel-Level Alignments for Text-to-4D Gaussian Splatting
Qiaowei Miao, Yawei Luo, Yi Yang (Zhejiang University)
<details span>
<summary><b>Abstract</b></summary>
As text-conditioned diffusion models (DMs) achieve breakthroughs in image, video, and 3D generation, the research community's focus has shifted to the more challenging task of text-to-4D synthesis, which introduces a temporal dimension to generate dynamic 3D objects. In this context, we identify Score Distillation Sampling (SDS), a widely used technique for text-to-3D synthesis, as a significant hindrance to text-to-4D performance due to its Janus-faced and texture-unrealistic problems coupled with high computational costs. In this paper, we propose Pixel-Level Alignments for Text-to-4D Gaussian Splatting (PLA4D), a novel method that utilizes text-to-video frames as explicit pixel alignment targets to generate static 3D objects and inject motion into them. Specifically, we introduce Focal Alignment to calibrate camera poses for rendering and GS-Mesh Contrastive Learning to distill geometry priors from rendered image contrasts at the pixel level. Additionally, we develop Motion Alignment using a deformation network to drive changes in Gaussians and implement Reference Refinement for smooth 4D object surfaces. These techniques enable 4D Gaussian Splatting to align geometry, texture, and motion with generated videos at the pixel level. Compared to previous methods, PLA4D produces synthesized outputs with better texture details in less time and effectively mitigates the Janus-faced problem. PLA4D is fully implemented using open-source models, offering an accessible, user-friendly, and promising direction for 4D digital content creation.
</details>

#### 10. STAR: Skeleton-aware Text-based 4D Avatar Generation with In-Network Motion Retargeting
Zenghao Chai, Chen Tang, Yongkang Wong, Mohan Kankanhalli

(National University of Singapore, Tsinghua University)
<details span>
<summary><b>Abstract</b></summary>
The creation of 4D avatars (i.e., animated 3D avatars) from text description typically uses text-to-image (T2I) diffusion models to synthesize 3D avatars in the canonical space and subsequently applies animation with target motions. However, such an optimization-by-animation paradigm has several drawbacks. (1) For pose-agnostic optimization, the rendered images in canonical pose for naive Score Distillation Sampling (SDS) exhibit domain gap and cannot preserve view-consistency using only T2I priors, and (2) For post hoc animation, simply applying the source motions to target 3D avatars yields translation artifacts and misalignment. To address these issues, we propose Skeleton-aware Text-based 4D Avatar generation with in-network motion Retargeting (STAR). STAR considers the geometry and skeleton differences between the template mesh and target avatar, and corrects the mismatched source motion by resorting to the pretrained motion retargeting techniques. With the informatively retargeted and occlusion-aware skeleton, we embrace the skeleton-conditioned T2I and text-to-video (T2V) priors, and propose a hybrid SDS module to coherently provide multi-view and frame-consistent supervision signals. Hence, STAR can progressively optimize the geometry, texture, and motion in an end-to-end manner. The quantitative and qualitative experiments demonstrate our proposed STAR can synthesize high-quality 4D avatars with vivid animations that align well with the text description. Additional ablation studies shows the contributions of each component in STAR.
</details>

#### 11. L4GM: Large 4D Gaussian Reconstruction Model
Jiawei Ren, Kevin Xie, Ashkan Mirzaei, Hanxue Liang, Xiaohui Zeng, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, Huan Ling

(NVIDIA, University of Toronto, University of Cambridge, MIT, Nanyang Technological University)
<details span>
<summary><b>Abstract</b></summary>
We present L4GM, the first 4D Large Reconstruction Model that produces animated objects from a single-view video input -- in a single feed-forward pass that takes only a second. Key to our success is a novel dataset of multiview videos containing curated, rendered animated objects from Objaverse. This dataset depicts 44K diverse objects with 110K animations rendered in 48 viewpoints, resulting in 12M videos with a total of 300M frames. We keep our L4GM simple for scalability and build directly on top of LGM, a pretrained 3D Large Reconstruction Model that outputs 3D Gaussian ellipsoids from multiview image input. L4GM outputs a per-frame 3D Gaussian Splatting representation from video frames sampled at a low fps and then upsamples the representation to a higher fps to achieve temporal smoothness. We add temporal self-attention layers to the base LGM to help it learn consistency across time, and utilize a per-timestep multiview rendering loss to train the model. The representation is upsampled to a higher framerate by training an interpolation model which produces intermediate 3D Gaussian representations. We showcase that L4GM that is only trained on synthetic data generalizes extremely well on in-the-wild videos, producing high quality animated 3D assets.
</details>

#### 12. 4K4DGen: Panoramic 4D Generation at 4K Resolution
Renjie Li, Panwang Pan, Bangbang Yang, Dejia Xu, Shijie Zhou, Xuanyang Zhang, Zeming Li, Achuta Kadambi, Zhangyang Wang, Zhiwen Fan

(Pico, UT Austin, UCLA)
<details span>
<summary><b>Abstract</b></summary>
The blooming of virtual reality and augmented reality (VR/AR) technologies has driven an increasing demand for the creation of high-quality, immersive, and dynamic environments. However, existing generative techniques either focus solely on dynamic objects or perform outpainting from a single perspective image, failing to meet the needs of VR/AR applications. In this work, we tackle the challenging task of elevating a single panorama to an immersive 4D experience. For the first time, we demonstrate the capability to generate omnidirectional dynamic scenes with 360-degree views at 4K resolution, thereby providing an immersive user experience. Our method introduces a pipeline that facilitates natural scene animations and optimizes a set of 4D Gaussians using efficient splatting techniques for real-time exploration. To overcome the lack of scene-scale annotated 4D data and models, especially in panoramic formats, we propose a novel Panoramic Denoiser that adapts generic 2D diffusion priors to animate consistently in 360-degree images, transforming them into panoramic videos with dynamic scenes at targeted regions. Subsequently, we elevate the panoramic video into a 4D immersive environment while preserving spatial and temporal consistency. By transferring prior knowledge from 2D models in the perspective domain to the panoramic domain and the 4D lifting with spatial appearance and geometry regularization, we achieve high-quality Panorama-to-4D generation at a resolution of (4096 × 2048) for the first time.
</details>

#### 13. Shape of Motion: 4D Reconstruction from a Single Video
Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, Angjoo Kanazawa

(UC Berkeley, Google Research)
<details span>
<summary><b>Abstract</b></summary>
Monocular dynamic reconstruction is a challenging and long-standing vision problem due to the highly ill-posed nature of the task. Existing approaches are limited in that they either depend on templates, are effective only in quasi-static scenes, or fail to model 3D motion explicitly. In this work, we introduce a method capable of reconstructing generic dynamic scenes, featuring explicit, full-sequence-long 3D motion, from casually captured monocular videos. We tackle the under-constrained nature of the problem with two key insights: First, we exploit the low-dimensional structure of 3D motion by representing scene motion with a compact set of SE3 motion bases. Each point's motion is expressed as a linear combination of these bases, facilitating soft decomposition of the scene into multiple rigidly-moving groups. Second, we utilize a comprehensive set of data-driven priors, including monocular depth maps and long-range 2D tracks, and devise a method to effectively consolidate these noisy supervisory signals, resulting in a globally consistent representation of the dynamic scene. Experiments show that our method achieves state-of-the-art performance for both long-range 3D/2D motion estimation and novel view synthesis on dynamic scenes.
</details>

#### 14. 4Dynamic: Text-to-4D Generation with Hybrid Priors
Yu-Jie Yuan, Leif Kobbelt, Jiwen Liu, Yuan Zhang, Pengfei Wan, Yu-Kun Lai, Lin Gao

<details span>
<summary><b>Abstract</b></summary>
Due to the fascinating generative performance of text-to-image diffusion models, growing text-to-3D generation works explore distilling the 2D generative priors into 3D, using the score distillation sampling (SDS) loss, to bypass the data scarcity problem. The existing text-to-3D methods have achieved promising results in realism and 3D consistency, but text-to-4D generation still faces challenges, including lack of realism and insufficient dynamic motions. In this paper, we propose a novel method for text-to-4D generation, which ensures the dynamic amplitude and authenticity through direct supervision provided by a video prior. Specifically, we adopt a text-to-video diffusion model to generate a reference video and divide 4D generation into two stages: static generation and dynamic generation. The static 3D generation is achieved under the guidance of the input text and the first frame of the reference video, while in the dynamic generation stage, we introduce a customized SDS loss to ensure multi-view consistency, a video-based SDS loss to improve temporal consistency, and most importantly, direct priors from the reference video to ensure the quality of geometry and texture. Moreover, we design a prior-switching training strategy to avoid conflicts between different priors and fully leverage the benefits of each prior. In addition, to enrich the generated motion, we further introduce a dynamic modeling representation composed of a deformation network and a topology network, which ensures dynamic continuity while modeling topological changes. Our method not only supports text-to-4D generation but also enables 4D generation from monocular videos. The comparison experiments demonstrate the superiority of our method compared to existing methods.
</details>

#### 15. CT4D: Consistent Text-to-4D Generation with Animatable Meshes
Ce Chen, Shaoli Huang, Xuelin Chen, Guangyi Chen, Xiaoguang Han, Kun Zhang, Mingming Gong

(Mohamed bin Zayed University of Artificial Intelligence, Tencent AI Lab, Carnegie Mellon University, FNii CUHKSZ, SSE CUHKSZ, University of Melbourne)
<details span>
<summary><b>Abstract</b></summary>
Text-to-4D generation has recently been demonstrated viable by integrating a 2D image diffusion model with a video diffusion model. However, existing models tend to produce results with inconsistent motions and geometric structures over time. To this end, we present a novel framework, coined CT4D, which directly operates on animatable meshes for generating consistent 4D content from arbitrary user-supplied prompts. The primary challenges of our mesh-based framework involve stably generating a mesh with details that align with the text prompt while directly driving it and maintaining surface continuity. Our CT4D framework incorporates a unique Generate-Refine-Animate (GRA) algorithm to enhance the creation of text-aligned meshes. To improve surface continuity, we divide a mesh into several smaller regions and implement a uniform driving function within each area. Additionally, we constrain the animating stage with a rigidity regulation to ensure cross-region continuity. Our experimental results, both qualitative and quantitative, demonstrate that our CT4D framework surpasses existing text-to-4D techniques in maintaining interframe consistency and preserving global geometry. Furthermore, we showcase that this enhanced representation inherently possesses the capability for combinational 4D generation and texture editing.
</details>


#### 16. Disco4D: Disentangled 4D Human Generation and Animation from a Single Image
Hui En Pang, Shuai Liu, Zhongang Cai, Lei Yang, Tianwei Zhang, Ziwei Liu

(Nanyang Technological University, Sensetime Research, Shanghai AI Lab)
<details span>
<summary><b>Abstract</b></summary>
We present Disco4D, a novel Gaussian Splatting framework for 4D human genera- tion and animation from a single image. Different from existing methods, Disco4D distinctively disentangles clothings (with Gaussian models) from the human body (with SMPL-X model), significantly enhancing the generation details and flexibility. It has the following technical innovations. 1) Disco4D learns to efficiently fit the clothing Gaussians over the SMPL-X Gaussians. 2) It adopts diffusion models to enhance the 3D generation process, e.g., modeling occluded parts not visible in the input image. 3) It learns an identity encoding for each clothing Gaussian to facilitate the separation and extraction of clothing assets. Furthermore, Disco4D naturally supports 4D human animation with vivid dynamics. Extensive experiments demonstrate the superiority of Disco4D on 4D human generation and animation tasks.
</details>

#### 17. MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion
Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, Ming-Hsuan Yang

(UC Berkeley, Google DeepMind, Stability AI, UC Merced)
<details span>
<summary><b>Abstract</b></summary>
Estimating geometry from dynamic scenes, where objects move and deform over time, remains a core challenge in computer vision. Current approaches often rely on multi-stage pipelines or global optimizations that decompose the problem into subtasks, like depth and flow, leading to complex systems prone to errors. In this paper, we present Motion DUSt3R (MonST3R), a novel geometry-first approach that directly estimates per-timestep geometry from dynamic scenes. Our key insight is that by simply estimating a pointmap for each timestep, we can effectively adapt DUST3R's representation, previously only used for static scenes, to dynamic scenes. However, this approach presents a significant challenge: the scarcity of suitable training data, namely dynamic, posed videos with depth labels. Despite this, we show that by posing the problem as a fine-tuning task, identifying several suitable datasets, and strategically training the model on this limited data, we can surprisingly enable the model to handle dynamics, even without an explicit motion representation. Based on this, we introduce new optimizations for several downstream video-specific tasks and demonstrate strong performance on video depth and camera pose estimation, outperforming prior work in terms of robustness and efficiency. Moreover, MonST3R shows promising results for primarily feed-forward 4D reconstruction.
</details>

#### 18. AvatarGO: Zero-shot 4D Human-Object Interaction Generation and Animation
Yukang Cao, Liang Pan, Kai Han, Kwan-Yee K. Wong, Ziwei Liu

(Nanyang Technological University, Shanghai AI Lab, The University of Hong Kong)
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in diffusion models have led to significant improvements in the generation and animation of 4D full-body human-object interactions (HOI). Nevertheless, existing methods primarily focus on SMPL-based motion generation, which is limited by the scarcity of realistic large-scale interaction data. This constraint affects their ability to create everyday HOI scenes. This paper addresses this challenge using a zero-shot approach with a pre-trained diffusion model.Despite this potential, achieving our goals is difficult due to the diffusion model's lack of understanding of "where" and "how" objects interact with the human body. To tackle these issues, we introduce AvatarGO, a novel framework designed to generate animatable 4D HOI scenes directly from textual inputs. Specifically, 1) for the "where" challenge, we propose LLM-guided contact retargeting, which employs Lang-SAM to identify the contact body part from text prompts, ensuring precise representation of human-object spatial relations. 2) For the "how" challenge, we introduce correspondence-aware motion optimization that constructs motion fields for both human and object models using the linear blend skinning function from SMPL-X. Our framework not only generates coherent compositional motions, but also exhibits greater robustness in handling penetration issues. Extensive experiments with existing methods validate AvatarGO's superior generation and animation capabilities on a variety of human-object pairs and diverse poses. As the first attempt to synthesize 4D avatars with object interactions, we hope AvatarGO could open new doors for human-centric 4D content creation.
</details>

#### 19. 4-LEGS: 4D Language Embedded Gaussian Splatting
Gal Fiebelman, Tamir Cohen, Ayellet Morgenstern, Peter Hedman, Hadar Averbuch-Elor

(Tel Aviv University, Google Research)
<details span>
<summary><b>Abstract</b></summary>
The emergence of neural representations has revolutionized our means for digitally viewing a wide range of 3D scenes, enabling the synthesis of photorealistic images rendered from novel views. Recently, several techniques have been proposed for connecting these low-level representations with the high-level semantics understanding embodied within the scene. These methods elevate the rich semantic understanding from 2D imagery to 3D representations, distilling high-dimensional spatial features onto 3D space. In our work, we are interested in connecting language with a dynamic modeling of the world. We show how to lift spatio-temporal features to a 4D representation based on 3D Gaussian Splatting. This enables an interactive interface where the user can spatiotemporally localize events in the video from text prompts. We demonstrate our system on public 3D video datasets of people and animals performing various actions.
</details>

#### 20. GenXD: Generating Any 3D and 4D Scenes
Yuyang Zhao, Chung-Ching Lin, Kevin Lin, Zhiwen Yan, Linjie Li, Zhengyuan Yang, Jianfeng Wang, Gim Hee Lee, Lijuan Wang

(National University of Singapore, Microsoft)
<details span>
<summary><b>Abstract</b></summary>
Recent developments in 2D visual generation have been remarkably successful. However, 3D and 4D generation remain challenging in real-world applications due to the lack of large-scale 4D data and effective model design. In this paper, we propose to jointly investigate general 3D and 4D generation by leveraging camera and object movements commonly observed in daily life. Due to the lack of real-world 4D data in the community, we first propose a data curation pipeline to obtain camera poses and object motion strength from videos. Based on this pipeline, we introduce a large-scale real-world 4D scene dataset: CamVid-30K. By leveraging all the 3D and 4D data, we develop our framework, GenXD, which allows us to produce any 3D or 4D scene. We propose multiview-temporal modules, which disentangle camera and object movements, to seamlessly learn from both 3D and 4D data. Additionally, GenXD employs masked latent conditions to support a variety of conditioning views. GenXD can generate videos that follow the camera trajectory as well as consistent 3D views that can be lifted into 3D representations. We perform extensive evaluations across various real-world and synthetic datasets, demonstrating GenXD's effectiveness and versatility compared to previous methods in 3D and 4D generation.
</details>

#### 21. DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion
Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, Yikai Wang

(HKUST, Tsinghua University, ShengShu)
<details span>
<summary><b>Abstract</b></summary>
In this paper, we introduce \textbf{DimensionX}, a framework designed to generate photorealistic 3D and 4D scenes from just a single image with video diffusion. Our approach begins with the insight that both the spatial structure of a 3D scene and the temporal evolution of a 4D scene can be effectively represented through sequences of video frames. While recent video diffusion models have shown remarkable success in producing vivid visuals, they face limitations in directly recovering 3D/4D scenes due to limited spatial and temporal controllability during generation. To overcome this, we propose ST-Director, which decouples spatial and temporal factors in video diffusion by learning dimension-aware LoRAs from dimension-variant data. This controllable video diffusion approach enables precise manipulation of spatial structure and temporal dynamics, allowing us to reconstruct both 3D and 4D representations from sequential frames with the combination of spatial and temporal dimensions. Additionally, to bridge the gap between generated videos and real-world scenes, we introduce a trajectory-aware mechanism for 3D generation and an identity-preserving denoising strategy for 4D generation. Extensive experiments on various real-world and synthetic datasets demonstrate that DimensionX achieves superior results in controllable video generation, as well as in 3D and 4D scene generation, compared with previous methods.
</details>

#### 22. CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models
Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T. Barron, Aleksander Holynski

(Google DeepMind, Columbia University, UC San Diego)
<details span>
<summary><b>Abstract</b></summary>
We present CAT4D, a method for creating 4D (dynamic 3D) scenes from monocular video. CAT4D leverages a multi-view video diffusion model trained on a diverse combination of datasets to enable novel view synthesis at any specified camera poses and timestamps. Combined with a novel sampling approach, this model can transform a single monocular video into a multi-view video, enabling robust 4D reconstruction via optimization of a deformable 3D Gaussian representation. We demonstrate competitive performance on novel view synthesis and dynamic scene reconstruction benchmarks, and highlight the creative capabilities for 4D scene generation from real or generated videos. 
</details>

#### 23. PaintScene4D: Consistent 4D Scene Generation from Text Prompts
Vinayak Gupta, Yunze Man, Yu-Xiong Wang

(Indian Institute of Technology Madras, University of Illinois Urbana-Champaign)
<details span>
<summary><b>Abstract</b></summary>
Recent advances in diffusion models have revolutionized 2D and 3D content creation, yet generating photorealistic dynamic 4D scenes remains a significant challenge. Existing dynamic 4D generation methods typically rely on distilling knowledge from pre-trained 3D generative models, often fine-tuned on synthetic object datasets. Consequently, the resulting scenes tend to be object-centric and lack photorealism. While text-to-video models can generate more realistic scenes with motion, they often struggle with spatial understanding and provide limited control over camera viewpoints during rendering. To address these limitations, we present PaintScene4D, a novel text-to-4D scene generation framework that departs from conventional multi-view generative models in favor of a streamlined architecture that harnesses video generative models trained on diverse real-world datasets. Our method first generates a reference video using a video generation model, and then employs a strategic camera array selection for rendering. We apply a progressive warping and inpainting technique to ensure both spatial and temporal consistency across multiple viewpoints. Finally, we optimize multi-view images using a dynamic renderer, enabling flexible camera control based on user preferences. Adopting a training-free architecture, our PaintScene4D efficiently produces realistic 4D scenes that can be viewed from arbitrary trajectories. 
</details>

---

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **Animate124: Animating One Image to 4D Dynamic Scene**  | 24 Nov 2023 |          [Link](https://arxiv.org/abs/2311.14603)          | [Link](https://github.com/HeliosZhao/Animate124)  | [Link](https://animate124.github.io/)  |
| 2023 | **4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency**  | 28 Dec 2023 |          [Link](https://arxiv.org/abs/2312.17225)          | [Link](https://github.com/VITA-Group/4DGen)  | [Link](https://vita-group.github.io/4DGen/)  |
| 2023 | **DreamGaussian4D:Generative 4D Gaussian Splatting**  | 28 Dec 2023 |          [Link](https://arxiv.org/abs/2312.17142)          | [Link](https://github.com/jiawei-ren/dreamgaussian4d)  | [Link](https://jiawei-ren.github.io/projects/dreamgaussian4d/)  |
| 2024 | **Fast Dynamic 3D Object Generation from a Single-view Video**  | 16 Jan 2024 |          [Link](https://arxiv.org/abs/2401.08742)          | [Link](https://github.com/fudan-zvg/Efficient4D)  | [Link](https://fudan-zvg.github.io/Efficient4D/)  |
| 2024 | **GaussianFlow: Splatting Gaussian Dynamics for 4D Content Creation**  | 19 Mar 2024 |         [Link](https://arxiv.org/abs/2403.12365)         | [Link](https://github.com/Zerg-Overmind/GaussianFlow)   | [Link](https://zerg-overmind.github.io/GaussianFlow.github.io/) |
| 2024 | **Comp4D: LLM-Guided Compositional 4D Scene Generation**  |  25 Mar 2024 |          [Link](https://arxiv.org/abs/2403.16993)          | [Link](https://github.com/VITA-Group/Comp4D)  |[Link](https://vita-group.github.io/Comp4D/#) |
| 2024 | **Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models**  | 26 May 2024 |          [Link](https://arxiv.org/abs/2405.16645)          | [Link](https://github.com/VITA-Group/Diffusion4D)  |[Link](https://vita-group.github.io/Diffusion4D/) |
| 2024 | **EG4D: Explicit Generation of 4D Object without Score Distillation**  | 28 May 2024 |         [Link](https://arxiv.org/abs/2405.18132)          | [Link](https://github.com/jasongzy/EG4D)  | -- |
| 2024 | **PLA4D: Pixel-Level Alignments for Text-to-4D Gaussian Splatting**  | 4 Jun 2024 |          [Link](https://arxiv.org/abs/2405.19957)          | --  | [Link](https://github.com/MiaoQiaowei/PLA4D.github.io) |
| 2024 | **STAR: Skeleton-aware Text-based 4D Avatar Generation with In-Network Motion Retargeting**  | 7 Jun 2024 |          [Link](https://arxiv.org/abs/2406.04629)          | [Link](https://github.com/czh-98/STAR)      | [Link](https://star-avatar.github.io/) |
| 2024 | **L4GM: Large 4D Gaussian Reconstruction Model**  | 14 Jun 2024 |          [Link](https://arxiv.org/abs/2406.10324)          | -- | [Link](https://research.nvidia.com/labs/toronto-ai/l4gm/) |
| 2024 | **4K4DGen: Panoramic 4D Generation at 4K Resolution**  | 19 Jun 2024 |          [Link](https://arxiv.org/abs/2406.13527)          | -- | [Link](https://4k4dgen.github.io/index.html) |
| 2024 | **Shape of Motion: 4D Reconstruction from a Single Video**  | 18 Jul 2024 |          [Link](https://arxiv.org/abs/2407.13764)          | [Link](https://github.com/vye16/shape-of-motion/) | [Link](https://shape-of-motion.github.io/) |
| 2024 | **4Dynamic: Text-to-4D Generation with Hybrid Priors**  | 17 Jul 2024 |          [Link](https://arxiv.org/abs/2407.12684)          | -- | -- |
| 2024 | **CT4D: Consistent Text-to-4D Generation with Animatable Meshes**  | 15 Aug 2024 |          [Link](https://arxiv.org/abs/2408.08342)          | -- | -- |
| 2024 | **Disco4D: Disentangled 4D Human Generation and Animation from a Single Image**  | 25 Sep 2024 |          [Link](https://arxiv.org/abs/2409.17280)          | -- |  [Link](https://disco-4d.github.io/) |
| 2024 | **MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion**  | 4 Oct 2024 |          [Link](https://arxiv.org/abs/2410.03825)          | [Link](https://github.com/Junyi42/monst3r)     |  [Link](https://monst3r-project.github.io/) |
| 2024 | **AvatarGO: Zero-shot 4D Human-Object Interaction Generation and Animation**  | 9 Oct 2024 |          [Link](https://arxiv.org/abs/2410.07164)          | [Link](https://github.com/yukangcao/AvatarGO)     |  [Link](https://yukangcao.github.io/AvatarGO/) |
| 2024 | **4-LEGS: 4D Language Embedded Gaussian Splatting**  | 15 Oct 2024 |          [Link](https://arxiv.org/abs/2410.10719)          |  --   |  [Link](https://tau-vailab.github.io/4-LEGS/) |
| 2024 | **GenXD: Generating Any 3D and 4D Scenes**  | 5 Nov 2024 |          [Link](https://arxiv.org/abs/2411.02319)          |  [Link](https://github.com/HeliosZhao/GenXD)   |  [Link](https://gen-x-d.github.io/) |
| 2024 | **DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion**  | 7 Nov 2024 |          [Link](https://arxiv.org/abs/2411.04928)          |  [Link](https://github.com/wenqsun/DimensionX)   |  [Link](https://chenshuo20.github.io/DimensionX/) |
| 2024 | **CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models**  | 27 Nov 2024 |          [Link](https://arxiv.org/abs/2411.18613)          | --  |  [Link](https://cat-4d.github.io/) |
| 2024 | **PaintScene4D: Consistent 4D Scene Generation from Text Prompts**  | 5 Dec 2024 |          [Link](https://arxiv.org/abs/2412.04471)          | [Link](https://github.com/paintscene4d/paintscene4d.github.io)  |  [Link](https://paintscene4d.github.io/) |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@article{zhao2023animate124,
  author    = {Zhao, Yuyang and Yan, Zhiwen and Xie, Enze and Hong, Lanqing and Li, Zhenguo and Lee, Gim Hee},
  title     = {Animate124: Animating One Image to 4D Dynamic Scene},
  journal   = {arXiv preprint arXiv:2311.14603},
  year      = {2023},
}

@article{yin20234dgen,
  title={4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency},
  author={},
  journal={arXiv preprint: 2312.17225},
  year={2023}
}

@article{ren2023dreamgaussian4d,
  title={DreamGaussian4D: Generative 4D Gaussian Splatting},
  author={Ren, Jiawei and Pan, Liang and Tang, Jiaxiang and Zhang, Chi and Cao, Ang and Zeng, Gang and Liu, Ziwei},
  journal={arXiv preprint arXiv:xxxx.xxxx},
  year={2023}
}

@article{pan2024fast,
  title={Fast Dynamic 3D Object Generation from a Single-view Video},
  author={Pan, Zijie and Yang, Zeyu and Zhu, Xiatian and Zhang, Li},
  journal={arXiv preprint arXiv 2401.08742},
  year={2024}
}

@article{gao2024gaussianflow,
  title={GaussianFlow: Splatting Gaussian Dynamics for 4D Content Creation},
  author={Gao, Quankai and Xu, Qiangeng and Cao, Zhe and Mildenhall, Ben and Ma, Wenchao and Chen, Le and Tang, Danhang and Neumann, Ulrich},
  journal={arXiv preprint arXiv:2403.12365},
  year={2024}
}

@misc{xu2024comp4d,
      title={Comp4D: LLM-Guided Compositional 4D Scene Generation}, 
      author={Dejia Xu and Hanwen Liang and Neel P. Bhatt and Hezhen Hu and Hanxue Liang and Konstantinos N. Plataniotis and Zhangyang Wang},
      year={2024},
      eprint={2403.16993},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{liang2024diffusion4d,
      title={Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models}, 
      author={Hanwen Liang and Yuyang Yin and Dejia Xu and Hanxue Liang and Zhangyang Wang and Konstantinos N. Plataniotis and Yao Zhao and Yunchao Wei},
      year={2024},
      eprint={2405.16645},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{sun2024eg4d,
  title={EG4D: Explicit Generation of 4D Object without Score Distillation},
  author={Qi Sun and Zhiyang Guo and Ziyu Wan and Jing Nathan Yan and Shengming Yin and Wengang Zhou and Jing Liao and Houqiang Li},
  journal={arXiv preprint arXiv:2405.18132},
  year={2024}
}

@misc{miao2024pla4d,
      title={PLA4D: Pixel-Level Alignments for Text-to-4D Gaussian Splatting}, 
      author={Qiaowei Miao and Yawei Luo and Yi Yang},
      year={2024},
      eprint={2405.19957},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{chai2024star,
  author = {Chai, Zenghao and Tang, Chen and Wong, Yongkang and Kankanhalli, Mohan},
  title = {STAR: Skeleton-aware Text-based 4D Avatar Generation with In-Network Motion Retargeting},
  eprint={2406.04629},
  archivePrefix={arXiv},
  year={2024},
}

@article{ren2024l4gm,
    title={L4GM: Large 4D Gaussian Reconstruction Model},
    author={Ren, Jiawei and Xie, Kevin and Mirzaei, Ashkan and Liang, Hanxue and Zeng, Xiaohui and Kreis, Karsten and Liu, Ziwei and Torralba, Antonio and Fidler, Sanja and Kim, Seung Wook and Ling, Huan},
    title={arXiv preprint arXiv:2406.xxxxx},
    year={2024}
}

@misc{li20244k4dgen,
      title={4K4DGen: Panoramic 4D Generation at 4K Resolution}, 
      author={Renjie Li and Panwang Pan and Bangbang Yang and Dejia Xu and Shijie Zhou and Xuanyang Zhang and Zeming Li and Achuta Kadambi and Zhangyang Wang and Zhiwen Fan},
      year={2024},
      eprint={2406.13527},
      archivePrefix={arXiv},
      primaryClass={id='cs.CV' full_name='Computer Vision and Pattern Recognition' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers image processing, computer vision, pattern recognition, and scene understanding. Roughly includes material in ACM Subject Classes I.2.10, I.4, and I.5.'}
}

@inproceedings{som2024,
  title     = {Shape of Motion: 4D Reconstruction from a Single Video},
  author    = {Wang, Qianqian and Ye, Vickie and Gao, Hang and Austin, Jake and Li, Zhengqi and Kanazawa, Angjoo},
  journal   = {arXiv preprint arXiv:2407.13764},
  year      = {2024}
}

@misc{yuan20244dynamictextto4dgenerationhybrid,
      title={4Dynamic: Text-to-4D Generation with Hybrid Priors}, 
      author={Yu-Jie Yuan and Leif Kobbelt and Jiwen Liu and Yuan Zhang and Pengfei Wan and Yu-Kun Lai and Lin Gao},
      year={2024},
      eprint={2407.12684},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.12684}, 
}

@misc{chen2024ct4dconsistenttextto4dgeneration,
      title={CT4D: Consistent Text-to-4D Generation with Animatable Meshes}, 
      author={Ce Chen and Shaoli Huang and Xuelin Chen and Guangyi Chen and Xiaoguang Han and Kun Zhang and Mingming Gong},
      year={2024},
      eprint={2408.08342},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2408.08342}, 
}

@misc{pang2024disco4ddisentangled4dhuman,
      title={Disco4D: Disentangled 4D Human Generation and Animation from a Single Image}, 
      author={Hui En Pang and Shuai Liu and Zhongang Cai and Lei Yang and Tianwei Zhang and Ziwei Liu},
      year={2024},
      eprint={2409.17280},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.17280}, 
}

@article{zhang2024monst3r,
  title={MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion},
  author={Zhang, Junyi and Herrmann, Charles and Hur, Junhwa and Jampani, Varun and Darrell, Trevor and Cole, Forrester and Sun, Deqing and Yang, Ming-Hsuan},
  journal={arXiv preprint arxiv:2410.03825},
  year={2024}
}

@misc{cao2024avatargozeroshot4dhumanobject,
      title={AvatarGO: Zero-shot 4D Human-Object Interaction Generation and Animation}, 
      author={Yukang Cao and Liang Pan and Kai Han and Kwan-Yee K. Wong and Ziwei Liu},
      year={2024},
      eprint={2410.07164},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.07164}, 
}

@misc{fiebelman20244legs4dlanguageembedded,
    title={4-LEGS: 4D Language Embedded Gaussian Splatting},
    author={Gal Fiebelman and Tamir Cohen and Ayellet Morgenstern and Peter Hedman and Hadar Averbuch-Elor},
    year={2024},
    eprint={2410.10719},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@article{zhao2024genxd,
  author={Zhao, Yuyang and Lin, Chung-Ching and Lin, Kevin and Yan, Zhiwen and Li, Linjie and Yang, Zhengyuan and Wang, Jianfeng and Lee, Gim Hee and Wang, Lijuan},
  title={GenXD: Generating Any 3D and 4D Scenes},
  journal={arXiv preprint arXiv:2411.02319},
  year={2024}
}

@misc{sun2024dimensionxcreate3d4d,
    title={DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion}, 
    author={Wenqiang Sun and Shuo Chen and Fangfu Liu and Zilong Chen and Yueqi Duan and Jun Zhang and Yikai Wang},
    year={2024},
    eprint={2411.04928},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2411.04928}, 
}

@misc{wu2024cat4dcreate4dmultiview,
      title={CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models}, 
      author={Rundi Wu and Ruiqi Gao and Ben Poole and Alex Trevithick and Changxi Zheng and Jonathan T. Barron and Aleksander Holynski},
      year={2024},
      eprint={2411.18613},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.18613}, 
}

@article{gupta2024paintscene4d,
title={PaintScene4D: Consistent 4D Scene Generation from Text Prompts},
author={Gupta, Vinayak and Man, Yunze and Wang, Yuxiong},
journal={https://arxiv.org/abs/2412.04471},
year={2024}
}
```
</details>

---

### Other 4D Additional Info
```Video-to-4D diffusion model for novel-view video synthesis```

**1. SV4D: Dynamic 3D Content Generation with Multi-Frame and Multi-View Consistency**

Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, Varun Jampani

(Stability AI, Northeastern University)
<details span>
<summary><b>Abstract</b></summary>
We present Stable Video 4D (SV4D), a latent video diffusion model for multi-frame and multi-view consistent dynamic 3D content generation. Unlike previous methods that rely on separately trained generative models for video generation and novel view synthesis, we design a unified diffusion model to generate novel view videos of dynamic 3D objects. Specifically, given a monocular reference video, SV4D generates novel views for each video frame that are temporally consistent. We then use the generated novel view videos to optimize an implicit 4D representation (dynamic NeRF) efficiently, without the need for cumbersome SDS-based optimization used in most prior works. To train our unified novel view video generation model, we curated a dynamic 3D object dataset from the existing Objaverse dataset. Extensive experimental results on multiple datasets and user studies demonstrate SV4D's state-of-the-art performance on novel-view video synthesis as well as 4D generation compared to prior works.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2024 | **SV4D: Dynamic 3D Content Generation with Multi-Frame and Multi-View Consistency**  | 24 Jul 2024 |          [Link](https://arxiv.org/abs/2407.17470)          | [Link](https://github.com/Stability-AI/generative-models) | [Link](https://sv4d.github.io/) |

## Text to Scene

### 🎉 Scene Accepted Papers
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **SceneScape: Text-Driven Consistent Scene Generation**  | NeurIPS 2023 |          [Link](https://arxiv.org/abs/2302.01133)          | [Link](https://github.com/RafailFridman/SceneScape)  | [Link](https://scenescape.github.io/)  |
| 2023 | **Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models**  |  ICCV 2023 (Oral) |          [Link](https://arxiv.org/abs/2303.11989)          | [Link](https://github.com/lukasHoel/text2room)  | [Link](https://lukashoel.github.io/text-to-room/)  |
| 2023 | **SceneWiz3D: Towards Text-guided 3D Scene Composition**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2312.08885)          | [Link](https://github.com/zqh0253/SceneWiz3D)   | [Link](https://zqh0253.github.io/SceneWiz3D/)  |
| 2023 | **GraphDreamer: Compositional 3D Scene Synthesis from Scene Graphs**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2312.00093)          | [Link](https://github.com/GGGHSL/GraphDreamer)  | [Link](https://graphdreamer.github.io/)  |
| 2023 | **ControlRoom3D: Room Generation using Semantic Proxy Rooms**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2312.05208)          | --  | [Link](https://jonasschult.github.io/ControlRoom3D/)  |
| 2024 | **ART3D: 3D Gaussian Splatting for Text-Guided Artistic Scenes Generation**  | CVPR 2024 Workshop on AI3DG |          [Link](https://arxiv.org/abs/2405.10508)          | -- |--|
| 2024 | **GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guidedGenerative Gaussian Splatting**  | ICML 2024 |          [Link](https://arxiv.org/abs/2402.07207)          | [Link](https://github.com/VDIGPKU/GALA3D)  |[Link](https://gala3d.github.io/) |
| 2024 | **Disentangled 3D Scene Generation with Layout Learning**  | ICML 2024 |          [Link](https://arxiv.org/abs/2402.16936)          | --  |[Link](https://dave.ml/layoutlearning/) |
| 2024 | **DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting**  | ECCV 2024 |          [Link](https://arxiv.org/abs/2404.06903)          | [Link](https://github.com/ShijieZhou-UCLA/DreamScene360)  |[Link](https://dreamscene360.github.io/) |
| 2024 | **BeyondScene: Higher-Resolution Human-Centric Scene Generation With Pretrained Diffusion**  | ECCV 2024 |          [Link](https://arxiv.org/abs/2404.04544)          | [Link](https://github.com/BeyondScene/BeyondScene) |[Link](https://janeyeon.github.io/beyond-scene/) |
| 2024 | **DreamScene: 3D Gaussian-based Text-to-3D Scene Generation via Formation Pattern Sampling**  | ECCV 2024 |          [Link](https://arxiv.org/abs/2404.03575)          | [Link](https://github.com/DreamScene-Project/DreamScene)  |[Link](https://dreamscene-project.github.io/) |
| 2024 | **The Fabrication of Reality and Fantasy: Scene Generation with LLM-Assisted Prompt Interpretation**  | ECCV 2024 |          [Link](https://arxiv.org/abs/2407.12579)          | [Link](https://github.com/leo81005/Reality-and-Fantasy)  |[Link](https://leo81005.github.io/Reality-and-Fantasy/) |
| 2024 | **SceneTeller: Language-to-3D Scene Generation**  | ECCV 2024 |          [Link](https://arxiv.org/abs/2407.20727)          | [Link](https://github.com/sceneteller/SceneTeller)  |[Link](https://sceneteller.github.io/) |
| 2024 | **Director3D: Real-world Camera Trajectory and 3D Scene Generation from Text**  | NeurIPS 2024 |          [Link](https://arxiv.org/abs/2406.17601)          | [Link](https://github.com/imlixinyang/director3d)  |[Link](https://imlixinyang.github.io/director3d-page/) |
| 2024 | **ReplaceAnything3D:Text-Guided 3D Scene Editing with Compositional Neural Radiance Fields**  | NeurIPS 2024 |          [Link](https://arxiv.org/abs/2401.17895)          | --  |[Link](https://replaceanything3d.github.io/) |

<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

@article{SceneScape,
      author    = {Fridman, Rafail and Abecasis, Amit and Kasten, Yoni and Dekel, Tali},
      title     = {SceneScape: Text-Driven Consistent Scene Generation},
      journal   = {arXiv preprint arXiv:2302.01133},
      year      = {2023},
  }

@InProceedings{hoellein2023text2room,
    author    = {H\"ollein, Lukas and Cao, Ang and Owens, Andrew and Johnson, Justin and Nie{\ss}ner, Matthias},
    title     = {Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {7909-7920}
}

@inproceedings{zhang2023scenewiz3d,
              author = {Qihang Zhang and Chaoyang Wang and Aliaksandr Siarohin and Peiye Zhuang and Yinghao Xu and Ceyuan Yang and Dahua Lin and Bo Dai and Bolei Zhou and Sergey Tulyakov and Hsin-Ying Lee},
              title = {{SceneWiz3D}: Towards Text-guided {3D} Scene Composition},
              booktitle = {arXiv},
              year = {2023}
}

@Inproceedings{gao2024graphdreamer,
  author    = {Gao, Gege and Liu, Weiyang and Chen, Anpei and Geiger, Andreas and Schölkopf, Bernhard},
  title     = {GraphDreamer: Compositional 3D Scene Synthesis from Scene Graphs},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
}

@inproceedings{schult24controlroom3d,
  author    = {Schult, Jonas and Tsai, Sam and H\"ollein, Lukas and Wu, Bichen and Wang, Jialiang and Ma, Chih-Yao and Li, Kunpeng and Wang, Xiaofang and Wimbauer, Felix and He, Zijian and Zhang, Peizhao and Leibe, Bastian and Vajda, Peter and Hou, Ji},
  title     = {ControlRoom3D: Room Generation using Semantic Proxy Rooms},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
}

@misc{li2024art3d,
      title={ART3D: 3D Gaussian Splatting for Text-Guided Artistic Scenes Generation}, 
      author={Pengzhi Li and Chengshuai Tang and Qinxuan Huang and Zhiheng Li},
      year={2024},
      eprint={2405.10508},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{zhou2024gala3d,
      title={GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting}, 
      author={Xiaoyu Zhou and Xingjian Ran and Yajiao Xiong and Jinlin He and Zhiwei Lin and Yongtao Wang and Deqing Sun and Ming-Hsuan Yang},
      year={2024},
      eprint={2402.07207},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{epstein2024disentangled,
      title={Disentangled 3D Scene Generation with Layout Learning},
      author={Dave Epstein and Ben Poole and Ben Mildenhall and Alexei A. Efros and Aleksander Holynski},
      year={2024},
      eprint={2402.16936},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{zhou2024dreamscene360,
  author    = {Zhou, Shijie and Fan, Zhiwen and Xu, Dejia and Chang, Haoran and Chari, Pradyumna and Bharadwaj, Tejas You, Suya and Wang, Zhangyang and Kadambi, Achuta},
  title     = {DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting},
  journal   = {arXiv preprint arXiv:2404.06903},
  year      = {2024},
}

@misc{kim2024beyondscenehigherresolutionhumancentricscene,
      title={BeyondScene: Higher-Resolution Human-Centric Scene Generation With Pretrained Diffusion}, 
      author={Gwanghyun Kim and Hayeon Kim and Hoigi Seo and Dong Un Kang and Se Young Chun},
      year={2024},
      eprint={2404.04544},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.04544}, 
}

@article{li2024dreamscene,
  title={DreamScene: 3D Gaussian-based Text-to-3D Scene Generation via Formation Pattern Sampling},
  author={Li, Haoran and Shi, Haolin and Zhang, Wenli and Wu, Wenjun and Liao, Yong and Lin Wang and Lik-hang Lee and Zhou, Pengyuan},
  journal={arXiv preprint arXiv:2404.03575},
  year={2024}
}

@article{yao2024fabricationrealityfantasyscene,
    title          = {The Fabrication of Reality and Fantasy: Scene Generation with LLM-Assisted Prompt Interpretation}, 
    author         = {Yi Yao and Chan-Feng Hsu and Jhe-Hao Lin and Hongxia Xie and Terence Lin and Yi-Ning Huang and Hong-Han Shuai and Wen-Huang Cheng},
    year           = {2024},
    eprint         = {2407.12579},
    archivePrefix  = {arXiv},
    primaryClass   = {cs.CV},
    url            = {https://arxiv.org/abs/2407.12579}, 
}

@misc{öcal2024scenetellerlanguageto3dscenegeneration,
      title={SceneTeller: Language-to-3D Scene Generation}, 
      author={Başak Melis Öcal and Maxim Tatarchenko and Sezer Karaoglu and Theo Gevers},
      year={2024},
      eprint={2407.20727},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.20727}, 
}

@article{li2024director3d,
  author = {Xinyang Li and Zhangyu Lai and Linning Xu and Yansong Qu and Liujuan Cao and Shengchuan Zhang and Bo Dai and Rongrong Ji},
  title = {Director3D: Real-world Camera Trajectory and 3D Scene Generation from Text},
  journal = {arXiv:2406.17601},
  year = {2024},
}

@misc{bartrum2024replaceanything3dtextguided,
            title={ReplaceAnything3D:Text-Guided 3D Scene Editing
              with Compositional Neural Radiance Fields}, 
            author={Edward Bartrum and Thu Nguyen-Phuoc and
              Chris Xie and Zhengqin Li and Numair Khan and
              Armen Avetisyan and Douglas Lanman and Lei Xiao},
            year={2024},
            eprint={2401.17895},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
}	
```
</details>

---

### 💡 Scene ArXiv Papers

#### 1. Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints 
Chuan Fang, Xiaotao Hu, Kunming Luo, Ping Tan 

(Hong Kong University of Science and Technology, Light Illusions, Nankai University)
<details span>
<summary><b>Abstract</b></summary>
Text-driven 3D indoor scene generation could be useful for gaming, film industry, and AR/VR applications. However, existing methods cannot faithfully capture the room layout, nor do they allow flexible editing of individual objects in the room. To address these problems, we present Ctrl-Room, which is able to generate convincing 3D rooms with designer-style layouts and high-fidelity textures from just a text prompt. Moreover, Ctrl-Room enables versatile interactive editing operations such as resizing or moving individual furniture items. Our key insight is to separate the modeling of layouts and appearance. %how to model the room that takes into account both scene texture and geometry at the same time. To this end, Our proposed method consists of two stages, a `Layout Generation Stage' and an `Appearance Generation Stage'. The `Layout Generation Stage' trains a text-conditional diffusion model to learn the layout distribution with our holistic scene code parameterization. Next, the `Appearance Generation Stage' employs a fine-tuned ControlNet to produce a vivid panoramic image of the room guided by the 3D scene layout and text prompt. In this way, we achieve a high-quality 3D room with convincing layouts and lively textures. Benefiting from the scene code parameterization, we can easily edit the generated room model through our mask-guided editing module, without expensive editing-specific training. Extensive experiments on the Structured3D dataset demonstrate that our method outperforms existing methods in producing more reasonable, view-consistent, and editable 3D rooms from natural language prompts.
</details>

#### 2. Text2Immersion: Generative Immersive Scene with 3D Gaussians
Hao Ouyang, Kathryn Heal, Stephen Lombardi, Tiancheng Sun (HKUST, Google)
<details span>
<summary><b>Abstract</b></summary>
We introduce Text2Immersion, an elegant method for producing high-quality 3D immersive scenes from text prompts. Our proposed pipeline initiates by progressively generating a Gaussian cloud using pre-trained 2D diffusion and depth estimation models. This is followed by a refining stage on the Gaussian cloud, interpolating and refining it to enhance the details of the generated scene. Distinct from prevalent methods that focus on single object or indoor scenes, or employ zoom-out trajectories, our approach generates diverse scenes with various objects, even extending to the creation of imaginary scenes. Consequently, Text2Immersion can have wide-ranging implications for various applications such as virtual reality, game development, and automated content creation. Extensive evaluations demonstrate that our system surpasses other methods in rendering quality and diversity, further progressing towards text-driven 3D scene generation. 
</details>

#### 3. ShowRoom3D: Text to High-Quality 3D Room Generation Using 3D Priors
Weijia Mao, Yan-Pei Cao, Jia-Wei Liu, Zhongcong Xu, Mike Zheng Shou

(Show Lab National University of Singapore, ARC Lab Tencent PCG)
<details span>
<summary><b>Abstract</b></summary>
We introduce ShowRoom3D, a three-stage approach for generating high-quality 3D room-scale scenes from texts. Previous methods using 2D diffusion priors to optimize neural radiance fields for generating room-scale scenes have shown unsatisfactory quality. This is primarily attributed to the limitations of 2D priors lacking 3D awareness and constraints in the training methodology. In this paper, we utilize a 3D diffusion prior, MVDiffusion, to optimize the 3D room-scale scene. Our contributions are in two aspects. Firstly, we propose a progressive view selection process to optimize NeRF. This involves dividing the training process into three stages, gradually expanding the camera sampling scope. Secondly, we propose the pose transformation method in the second stage. It will ensure MVDiffusion provide the accurate view guidance. As a result, ShowRoom3D enables the generation of rooms with improved structural integrity, enhanced clarity from any view, reduced content repetition, and higher consistency across different perspectives. Extensive experiments demonstrate that our method, significantly outperforms state-of-the-art approaches by a large margin in terms of user study.
</details>

#### 4. Detailed Human-Centric Text Description-Driven Large Scene Synthesis
Gwanghyun Kim, Dong Un Kang, Hoigi Seo, Hayeon Kim, Se Young Chun

(Dept. of Electrical and Computer Engineering, INMC & IPAI, Seoul National University Republic of Korea)
<details span>
<summary><b>Abstract</b></summary>
Text-driven large scene image synthesis has made significant progress with diffusion models, but controlling it is challenging. While using additional spatial controls with corresponding texts has improved the controllability of large scene synthesis, it is still challenging to faithfully reflect detailed text descriptions without user-provided controls. Here, we propose DetText2Scene, a novel text-driven large-scale image synthesis with high faithfulness, controllability, and naturalness in a global context for the detailed human-centric text description. Our DetText2Scene consists of 1) hierarchical keypoint-box layout generation from the detailed description by leveraging large language model (LLM), 2) view-wise conditioned joint diffusion process to synthesize a large scene from the given detailed text with LLM-generated grounded keypoint-box layout and 3) pixel perturbation-based pyramidal interpolation to progressively refine the large scene for global coherence. Our DetText2Scene significantly outperforms prior arts in text-to-large scene synthesis qualitatively and quantitatively, demonstrating strong faithfulness with detailed descriptions, superior controllability, and excellent naturalness in a global context.
</details>

#### 5. 3D-SceneDreamer: Text-Driven 3D-Consistent Scene Generation
Frank Zhang, Yibo Zhang, Quan Zheng, Rui Ma, Wei Hua, Hujun Bao, Weiwei Xu, Changqing Zou

(Zhejiang University, Jilin University, Zhejiang Lab, Institute of Software Chinese Academy of Sciences)
<details span>
<summary><b>Abstract</b></summary>
Text-driven 3D scene generation techniques have made rapid progress in recent years. Their success is mainly attributed to using existing generative models to iteratively perform image warping and inpainting to generate 3D scenes. However, these methods heavily rely on the outputs of existing models, leading to error accumulation in geometry and appearance that prevent the models from being used in various scenarios (e.g., outdoor and unreal scenarios). To address this limitation, we generatively refine the newly generated local views by querying and aggregating global 3D information, and then progressively generate the 3D scene. Specifically, we employ a tri-plane features-based NeRF as a unified representation of the 3D scene to constrain global 3D consistency, and propose a generative refinement network to synthesize new contents with higher quality by exploiting the natural image prior from 2D diffusion model as well as the global 3D information of the current scene. Our extensive experiments demonstrate that, in comparison to previous methods, our approach supports wide variety of scene generation and arbitrary camera trajectories with improved visual quality and 3D consistency.
</details>

#### 6. RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion
Jaidev Shriram, Alex Trevithick, Lingjie Liu, Ravi Ramamoorthi (University of California San Diego, University of Pennsylvania)
<details span>
<summary><b>Abstract</b></summary>
We introduce RealmDreamer, a technique for generation of general forward-facing 3D scenes from text descriptions. Our technique optimizes a 3D Gaussian Splatting representation to match complex text prompts. We initialize these splats by utilizing the state-of-the-art text-to-image generators, lifting their samples into 3D, and computing the occlusion volume. We then optimize this representation across multiple views as a 3D inpainting task with image-conditional diffusion models. To learn correct geometric structure, we incorporate a depth diffusion model by conditioning on the samples from the inpainting model, giving rich geometric structure. Finally, we finetune the model using sharpened samples from image generators. Notably, our technique does not require video or multi-view data and can synthesize a variety of high-quality 3D scenes in different styles, consisting of multiple objects. Its generality additionally allows 3D synthesis from a single image.
</details>

#### 7. 3DitScene: Editing Any Scene via Language-guided Disentangled Gaussian Splatting
Qihang Zhang, Yinghao Xu, Chaoyang Wang, Hsin-Ying Lee, Gordon Wetzstein, Bolei Zhou, Ceyuan Yang

(The Chinese University of Hong Kong, Stanford University, Snap Inc., University of California Los Angeles, ByteDance)
<details span>
<summary><b>Abstract</b></summary>
Scene image editing is crucial for entertainment, photography, and advertising design. Existing methods solely focus on either 2D individual object or 3D global scene editing. This results in a lack of a unified approach to effectively control and manipulate scenes at the 3D level with different levels of granularity. In this work, we propose 3DitScene, a novel and unified scene editing framework leveraging language-guided disentangled Gaussian Splatting that enables seamless editing from 2D to 3D, allowing precise control over scene composition and individual objects. We first incorporate 3D Gaussians that are refined through generative priors and optimization techniques. Language features from CLIP then introduce semantics into 3D geometry for object disentanglement. With the disentangled Gaussians, 3DitScene allows for manipulation at both the global and individual levels, revolutionizing creative expression and empowering control over scenes and objects. Experimental results demonstrate the effectiveness and versatility of 3DitScene in scene image editing.
</details>

#### 8. HoloDreamer: Holistic 3D Panoramic World Generation from Text Descriptions
Haiyang Zhou, Xinhua Cheng, Wangbo Yu, Yonghong Tian, Li Yuan

(Peking University, Peng Cheng Laboratory)
<details span>
<summary><b>Abstract</b></summary>
3D scene generation is in high demand across various domains, including virtual reality, gaming, and the film industry. Owing to the powerful generative capabilities of text-to-image diffusion models that provide reliable priors, the creation of 3D scenes using only text prompts has become viable, thereby significantly advancing researches in text-driven 3D scene generation. In order to obtain multiple-view supervision from 2D diffusion models, prevailing methods typically employ the diffusion model to generate an initial local image, followed by iteratively outpainting the local image using diffusion models to gradually generate scenes. Nevertheless, these outpainting-based approaches prone to produce global inconsistent scene generation results without high degree of completeness, restricting their broader applications. To tackle these problems, we introduce HoloDreamer, a framework that first generates high-definition panorama as a holistic initialization of the full 3D scene, then leverage 3D Gaussian Splatting (3D-GS) to quickly reconstruct the 3D scene, thereby facilitating the creation of view-consistent and fully enclosed 3D scenes. Specifically, we propose Stylized Equirectangular Panorama Generation, a pipeline that combines multiple diffusion models to enable stylized and detailed equirectangular panorama generation from complex text prompts. Subsequently, Enhanced Two-Stage Panorama Reconstruction is introduced, conducting a two-stage optimization of 3D-GS to inpaint the missing region and enhance the integrity of the scene. Comprehensive experiments demonstrated that our method outperforms prior works in terms of overall visual consistency and harmony as well as reconstruction quality and rendering robustness when generating fully enclosed scenes.
</details>

#### 9. Scene123: One Prompt to 3D Scene Generation via Video-Assisted and Consistency-Enhanced MAE
Yiying Yang, Fukun Yin, Jiayuan Fan, Xin Chen, Wanzhang Li, Gang Yu

(Fudan University, Tencent PCG)
<details span>
<summary><b>Abstract</b></summary>
As Artificial Intelligence Generated Content (AIGC) advances, a variety of methods have been developed to generate text, images, videos, and 3D objects from single or multimodal inputs, contributing efforts to emulate human-like cognitive content creation. However, generating realistic large-scale scenes from a single input presents a challenge due to the complexities involved in ensuring consistency across extrapolated views generated by models. Benefiting from recent video generation models and implicit neural representations, we propose Scene123, a 3D scene generation model, that not only ensures realism and diversity through the video generation framework but also uses implicit neural fields combined with Masked Autoencoders (MAE) to effectively ensures the consistency of unseen areas across views. Specifically, we initially warp the input image (or an image generated from text) to simulate adjacent views, filling the invisible areas with the MAE model. However, these filled images usually fail to maintain view consistency, thus we utilize the produced views to optimize a neural radiance field, enhancing geometric consistency.
Moreover, to further enhance the details and texture fidelity of generated views, we employ a GAN-based Loss against images derived from the input image through the video generation model. Extensive experiments demonstrate that our method can generate realistic and consistent scenes from a single prompt. Both qualitative and quantitative results indicate that our approach surpasses existing state-of-the-art methods.
</details>

#### 10. LayerPano3D: Layered 3D Panorama for Hyper-Immersive Scene Generation
Shuai Yang, Jing Tan, Mengchen Zhang, Tong Wu, Yixuan Li, Gordon Wetzstein, Ziwei Liu, Dahua Lin

(Shanghai Jiao Tong University, The Chinese University of Hong Kong, Zhejiang University, Shanghai AI Laboratory, Stanford University, S-Lab, Nanyang Technological University)
<details span>
<summary><b>Abstract</b></summary>
3D immersive scene generation is a challenging yet critical task in computer vision and graphics. A desired virtual 3D scene should 1) exhibit omnidirectional view consistency, and 2) allow for free exploration in complex scene hierarchies. Existing methods either rely on successive scene expansion via inpainting or employ panorama representation to represent large FOV scene environments. However, the generated scene suffers from semantic drift during expansion and is unable to handle occlusion among scene hierarchies. To tackle these challenges, we introduce LayerPano3D, a novel framework for full-view, explorable panoramic 3D scene generation from a single text prompt. Our key insight is to decompose a reference 2D panorama into multiple layers at different depth levels, where each layer reveals the unseen space from the reference views via diffusion prior. LayerPano3D comprises multiple dedicated designs: 1) we introduce a novel text-guided anchor view synthesis pipeline for high-quality, consistent panorama generation. 2) We pioneer the Layered 3D Panorama as underlying representation to manage complex scene hierarchies and lift it into 3D Gaussians to splat detailed 360-degree omnidirectional scenes with unconstrained viewing paths. Extensive experiments demonstrate that our framework generates state-of-the-art 3D panoramic scene in both full view consistency and immersive exploratory experience. We believe that LayerPano3D holds promise for advancing 3D panoramic scene creation with numerous applications.
</details>

#### 11. SceneDreamer360: Text-Driven 3D-Consistent Scene Generation with Panoramic Gaussian Splatting
Wenrui Li, Yapeng Mi, Fucheng Cai, Zhe Yang, Wangmeng Zuo, Xingtao Wang, Xiaopeng Fan

(Harbin Institute of Technology, University of Electronic Science and Technology of China)
<details span>
<summary><b>Abstract</b></summary>
Text-driven 3D scene generation has seen significant advancements recently. However, most existing methods generate single-view images using generative models and then stitch them together in 3D space. This independent generation for each view often results in spatial inconsistency and implausibility in the 3D scenes. To address this challenge, we proposed a novel text-driven 3D-consistent scene generation model: SceneDreamer360. Our proposed method leverages a text-driven panoramic image generation model as a prior for 3D scene generation and employs 3D Gaussian Splatting (3DGS) to ensure consistency across multi-view panoramic images. Specifically, SceneDreamer360 enhances the fine-tuned Panfusion generator with a three-stage panoramic enhancement, enabling the generation of high-resolution, detail-rich panoramic images. During the 3D scene construction, a novel point cloud fusion initialization method is used, producing higher quality and spatially consistent point clouds. Our extensive experiments demonstrate that compared to other methods, SceneDreamer360 with its panoramic image generation and 3DGS can produce higher quality, spatially consistent, and visually appealing 3D scenes from any text prompt. 
</details>

#### 12. Semantic Score Distillation Sampling for Compositional Text-to-3D Generation
Ling Yang, Zixiang Zhang, Junlin Han, Bohan Zeng, Runjia Li, Philip Torr, Wentao Zhang

(Peking University, University of Oxford)
<details span>
<summary><b>Abstract</b></summary>
Generating high-quality 3D assets from textual descriptions remains a pivotal challenge in computer graphics and vision research. Due to the scarcity of 3D data, state-of-the-art approaches utilize pre-trained 2D diffusion priors, optimized through Score Distillation Sampling (SDS). Despite progress, crafting complex 3D scenes featuring multiple objects or intricate interactions is still difficult. To tackle this, recent methods have incorporated box or layout guidance. However, these layout-guided compositional methods often struggle to provide fine-grained control, as they are generally coarse and lack expressiveness. To overcome these challenges, we introduce a novel SDS approach, Semantic Score Distillation Sampling (SemanticSDS), designed to effectively improve the expressiveness and accuracy of compositional text-to-3D generation. Our approach integrates new semantic embeddings that maintain consistency across different rendering views and clearly differentiate between various objects and parts. These embeddings are transformed into a semantic map, which directs a region-specific SDS process, enabling precise optimization and compositional generation. By leveraging explicit semantic guidance, our method unlocks the compositional capabilities of existing pre-trained diffusion models, thereby achieving superior quality in 3D content generation, particularly for complex objects and scenes. Experimental results demonstrate that our SemanticSDS framework is highly effective for generating state-of-the-art complex 3D content.
</details>

#### 13. The Scene Language: Representing Scenes with Programs, Words, and Embeddings
Yunzhi Zhang, Zizhang Li, Matt Zhou, Shangzhe Wu, Jiajun Wu (Stanford University, UC Berkeley)
<details span>
<summary><b>Abstract</b></summary>
We introduce the Scene Language, a visual scene representation that concisely and precisely describes the structure, semantics, and identity of visual scenes. It represents a scene with three key components: a program that specifies the hierarchical and relational structure of entities in the scene, words in natural language that summarize the semantic class of each entity, and embeddings that capture the visual identity of each entity. This representation can be inferred from pre-trained language models via a training-free inference technique, given text or image inputs. The resulting scene can be rendered into images using traditional, neural, or hybrid graphics renderers. Together, this forms a robust, automated system for high-quality 3D and 4D scene generation. Compared with existing representations like scene graphs, our proposed Scene Language generates complex scenes with higher fidelity, while explicitly modeling the scene structures to enable precise control and editing.
</details>

---

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints**  | 5 Oct 2023 |          [Link](https://arxiv.org/abs/2310.03602)          | [Link](https://github.com/fangchuan/Ctrl-Room)    | [Link](https://fangchuan.github.io/ctrl-room.github.io/)  |
| 2023 | **Text2Immersion: Generative Immersive Scene with 3D Gaussians**  | 14 Dec 2023 |          [Link](https://arxiv.org/abs/2312.09242)          | --   | [Link](https://ken-ouyang.github.io/text2immersion/index.html)  |
| 2023 | **ShowRoom3D: Text to High-Quality 3D Room Generation Using 3D Priors**  | 20 Dec 2023 |          [Link](https://arxiv.org/abs/2312.13324)          | [Link](https://github.com/showlab/ShowRoom3D)  | [Link](https://showroom3d.github.io/)  |
| 2023 | **Detailed Human-Centric Text Description-Driven Large Scene Synthesis**  | 30 Nov 2023 |          [Link](https://arxiv.org/abs/2311.18654)          | --  |-- |
| 2024 | **3D-SceneDreamer: Text-Driven 3D-Consistent Scene Generation**  | 14 Mar 2024 |          [Link](https://arxiv.org/abs/2403.09439)          | --  | -- |
| 2024 | **RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion**  | 10 Apr 2024 |          [Link](https://arxiv.org/abs/2404.07199)          | [Link](https://github.com/jaidevshriram/realmdreamer)  |[Link](https://realmdreamer.github.io/) |
| 2024 | **3DitScene: Editing Any Scene via Language-guided Disentangled Gaussian Splatting**  | 28 May 2024 |          [Link](https://arxiv.org/abs/2405.18424)          | [Link](https://github.com/zqh0253/3DitScene)  |[Link](https://zqh0253.github.io/3DitScene/) |
| 2024 | **HoloDreamer: Holistic 3D Panoramic World Generation from Text Descriptions**  | 21 Jul 2024 |          [Link](https://arxiv.org/abs/2407.15187)          | [Link](https://github.com/zhouhyOcean/HoloDreamer)  |[Link](https://zhouhyocean.github.io/holodreamer/) |
| 2024 | **Scene123: One Prompt to 3D Scene Generation via Video-Assisted and Consistency-Enhanced MAE**  | 10 Aug 2024 |          [Link](https://www.arxiv.org/abs/2408.05477)          | [Link](https://github.com/YiyingYang12/Scene123)  |[Link](https://yiyingyang12.github.io/Scene123.github.io/) |
| 2024 | **LayerPano3D: Layered 3D Panorama for Hyper-Immersive Scene Generation**  | 23 Aug 2024 |          [Link](https://arxiv.org/abs/2408.13252)          | [Link](https://github.com/YS-IMTech/LayerPano3D)  |[Link](https://ys-imtech.github.io/projects/LayerPano3D/) |
| 2024 | **SceneDreamer360: Text-Driven 3D-Consistent Scene Generation with Panoramic Gaussian Splatting**  | 25 Aug 2024 |          [Link](https://arxiv.org/abs/5811784)          | [Link](https://github.com/liwrui/SceneDreamer360)  | [Link](https://scenedreamer360.github.io/)|
| 2024 | **Semantic Score Distillation Sampling for Compositional Text-to-3D Generation**  | 11 Oct 2024 |          [Link](https://arxiv.org/abs/2410.09009)          | [Link](https://github.com/YangLing0818/SemanticSDS-3D)  | -- |
| 2024 | **The Scene Language: Representing Scenes with Programs, Words, and Embeddings**  | 22 Oct 2024 |          [Link](https://arxiv.org/abs/2410.16770)          | [Link](https://github.com/zzyunzhi/scene-language)  | [Link](https://ai.stanford.edu/~yzzhang/projects/scene-language/)|

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@article{fang2023ctrl,
      title={Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints},
      author={Fang, Chuan and Hu, Xiaotao and Luo, Kunming and Tan, Ping},
      journal={arXiv preprint arXiv:2310.03602},
      year={2023}
}

@article{ouyang2023text,
  author    = {Ouyang, Hao and Sun, Tiancheng and Lombardi, Stephen and Heal, Kathryn},
  title     = {Text2Immersion: Generative Immersive Scene with 3D Gaussians},
  journal   = {Arxiv},
  year      = {2023},
}

@article{mao2023showroom3d,
  title={ShowRoom3D: Text to High-Quality 3D Room Generation Using 3D Priors},
  author={Mao, Weijia and Cao, Yan-Pei and Liu, Jia-Wei and Xu, Zhongcong and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2312.13324},
  year={2023}
}

@article{kim2023detailed,
  title={Detailed Human-Centric Text Description-Driven Large Scene Synthesis},
  author={Kim, Gwanghyun and Kang, Dong Un and Seo, Hoigi and Kim, Hayeon and Chun, Se Young},
  journal={arXiv preprint arXiv:2311.18654},
  year={2023}
}

@misc{zhang20243dscenedreamer,
      title={3D-SceneDreamer: Text-Driven 3D-Consistent Scene Generation}, 
      author={Frank Zhang and Yibo Zhang and Quan Zheng and Rui Ma and Wei Hua and Hujun Bao and Weiwei Xu and Changqing Zou},
      year={2024},
      eprint={2403.09439},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{shriram2024realmdreamer,
        title={RealmDreamer: Text-Driven 3D Scene Generation with 
                Inpainting and Depth Diffusion},
        author={Jaidev Shriram and Alex Trevithick and Lingjie Liu and Ravi Ramamoorthi},
        journal={arXiv},
        year={2024}
}

inproceedings{zhang20243DitScene,
  author = {Qihang Zhang and Yinghao Xu and Chaoyang Wang and Hsin-Ying Lee and Gordon Wetzstein and Bolei Zhou and Ceyuan Yang},
  title = {{3DitScene}: Editing Any Scene via Language-guided Disentangled Gaussian Splatting},
  booktitle = {arXiv},
  year = {2024}
}

@misc{zhou2024holodreamerholistic3dpanoramic,
      title={HoloDreamer: Holistic 3D Panoramic World Generation from Text Descriptions}, 
      author={Haiyang Zhou and Xinhua Cheng and Wangbo Yu and Yonghong Tian and Li Yuan},
      year={2024},
      eprint={2407.15187},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.15187}, 
}

@misc{yang2024scene123prompt3dscene,
      title={Scene123: One Prompt to 3D Scene Generation via Video-Assisted and Consistency-Enhanced MAE}, 
      author={Yiying Yang and Fukun Yin and Jiayuan Fan and Xin Chen and Wanzhang Li and Gang Yu},
      year={2024},
      eprint={2408.05477},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.05477}, 
}

@misc{yang2024layerpano3dlayered3dpanorama,
      title={LayerPano3D: Layered 3D Panorama for Hyper-Immersive Scene Generation}, 
      author={Shuai Yang and Jing Tan and Mengchen Zhang and Tong Wu and Yixuan Li and Gordon Wetzstein and Ziwei Liu and Dahua Lin},
      year={2024},
      eprint={2408.13252},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.13252}, 
}

@misc{li2024scenedreamer360textdriven3dconsistentscene,
      title={SceneDreamer360: Text-Driven 3D-Consistent Scene Generation with Panoramic Gaussian Splatting}, 
      author={Wenrui Li and Yapeng Mi and Fucheng Cai and Zhe Yang and Wangmeng Zuo and Xingtao Wang and Xiaopeng Fan},
      year={2024},
      eprint={2408.13711},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.13711}, 
}

@article{yang2024semanticsds,
  title={Semantic Score Distillation Sampling for Compositional Text-to-3D Generation},
  author={Yang, Ling and Zhang, Zixiang and Han, Junlin and Zeng, Bohan and Li, Runjia and Torr, Philip and Zhang, Wentao},
  journal={arXiv preprint arXiv:2410.09009},
  year={2024}
}

@misc{zhang2024scenelanguagerepresentingscenes,
      title={The Scene Language: Representing Scenes with Programs, Words, and Embeddings}, 
      author={Yunzhi Zhang and Zizhang Li and Matt Zhou and Shangzhe Wu and Jiajun Wu},
      year={2024},
      eprint={2410.16770},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.16770}, 
}
```
</details>

## Text to Human Motion

### 🎉 Motion Accepted Papers

| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **MDM: Human Motion Diffusion Model**  | ICLR2023 (Top-25%) |          [Link](https://arxiv.org/abs/2209.14916)          | [Link](https://github.com/GuyTevet/motion-diffusion-model)  | [Link](https://guytevet.github.io/mdm-page/)  |
| 2023 | **MotionGPT: Human Motion as a Foreign Language**  | NeurIPS 2023 |          [Link](https://arxiv.org/abs/2306.14795)          | [Link](https://github.com/OpenMotionLab/MotionGPT)  | [Link](https://motion-gpt.github.io/)  |
| 2023 | **MLD: Motion Latent Diffusion Models**  | CVPR 2023 |          [Link](https://arxiv.org/abs/2212.04048)          | [Link](https://github.com/ChenFengYe/motion-latent-diffusion)  | [Link](https://chenxin.tech/mld/)  |
| 2023 | **MotionMix: Weakly-Supervised Diffusion for Controllable Motion Generation**  | AAAI 2024 |          [Link](https://arxiv.org/abs/2401.11115)          | [Link](https://github.com/NhatHoang2002/MotionMix/tree/main)  | [Link](https://nhathoang2002.github.io/MotionMix-page/)  |
| 2023 | **SinMDM: Single Motion Diffusion**  | ICLR 2024 Spotlight |          [Link](https://arxiv.org/abs/2302.05905)          | [Link](https://github.com/SinMDM/SinMDM)  | [Link](https://sinmdm.github.io/SinMDM-page/)  |
| 2023 | **MoMask: Generative Masked Modeling of 3D Human Motions**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2312.00063)          | [Link](https://github.com/EricGuo5513/momask-codes) | [Link](https://ericguo5513.github.io/momask/)  |
| 2023 | **Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2403.18036)          | [Link](https://github.com/diffusion-motion-transfer/diffusion-motion-transfer) | [Link](https://diffusion-motion-transfer.github.io/)  |
| 2024 | **Multi-Track Timeline Control for Text-Driven 3D Human Motion Generation**  | CVPRW 2024 |          [Link](https://arxiv.org/abs/2401.08559)          | [Link](https://github.com/nv-tlabs/stmc)| [Link](https://mathis.petrovich.fr/stmc/)  |
| 2024 | **in2IN: Leveraging individual Information to Generate Human INteractions**  | HuMoGen CVPRW 2024 |          [Link](https://arxiv.org/abs/2404.09988)          | [Link](https://github.com/pabloruizponce/in2IN) | [Link](https://pabloruizponce.github.io/in2IN/)  |
| 2024 | **Exploring Text-to-Motion Generation with Human Preference**  | HuMoGen CVPRW 2024 |          [Link](https://arxiv.org/abs/2404.09445)          | [Link](https://github.com/THU-LYJ-Lab/InstructMotion) | --  |
| 2024 | **FlowMDM: Seamless Human Motion Composition with Blended Positional Encodings**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2402.15509)          | [Link](https://github.com/BarqueroGerman/FlowMDM) | [Link](https://barquerogerman.github.io/FlowMDM/)  |
| 2024 | **Move as You Say, Interact as You Can: Language-guided Human Motion Generation with Scene Affordance**  | CVPR 2024 (Highlight) |          [Link](https://arxiv.org/abs/2311.17009)          | [Link](https://github.com/afford-motion/afford-motion) | [Link](https://afford-motion.github.io/)  |
| 2024 | **Generating Human Motion in 3D Scenes from Text Descriptions**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2405.07784)          | [Link](https://github.com/zju3dv/text_scene_motion) | [Link](https://zju3dv.github.io/text_scene_motion/)  |
| 2024 | **OmniMotionGPT: Animal Motion Generation with Limited Data**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2311.18303)          | [Link](https://github.com/USRC-SEA/OmniMotionGPT) | [Link](https://zshyang.github.io/omgpt-website/)  |
| 2024 | **HumanTOMATO: Text-aligned Whole-body Motion Generation**  | ICML 2024 |          [Link](https://arxiv.org/abs/2310.12978)          | [Link](https://github.com/IDEA-Research/HumanTOMATO)  | [Link](https://lhchen.top/HumanTOMATO/)  |
| 2024 | **Self-Correcting Self-Consuming Loops for Generative Model Training**  | ICML 2024 |          [Link](https://arxiv.org/abs/2402.07087)          | [Link](https://github.com/nate-gillman/self-correcting-self-consuming) | [Link](https://cs.brown.edu/people/ngillman//sc-sc.html)  |
| 2024 | **Flexible Motion In-betweening with Diffusion Models**  |SIGGRAPH 2024|          [Link](https://arxiv.org/abs/2405.11126)          | [Link](https://github.com/setarehc/diffusion-motion-inbetweening) | [Link](https://setarehc.github.io/CondMDI/)  |
| 2024 | **Iterative Motion Editing with Natural Language**  |SIGGRAPH 2024|          [Link](https://arxiv.org/abs/2312.11538)          |[Link](https://github.com/purvigoel/iterative-editing-release)| [Link](https://purvigoel.github.io/iterative-motion-editing/)  |
| 2024 | **MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model**  | ECCV 2024  |          [Link](https://arxiv.org/abs/2404.19759)          | [Link](https://github.com/Dai-Wenxun/MotionLCM) | [Link](https://dai-wenxun.github.io/MotionLCM-page/)  |
| 2024 | **ParCo: Part-Coordinating Text-to-Motion Synthesis**  | ECCV 2024  |          [Link](https://arxiv.org/abs/2403.18512)          | [Link](https://github.com/qrzou/ParCo) | -- |
| 2024 | **CoMo: Controllable Motion Generation through Language Guided Pose Code Editing**  | ECCV 2024  |          [Link](https://arxiv.org/abs/2403.13900)          | [Link](https://github.com/yh2371/CoMo) | [Link](https://yh2371.github.io/como/) |
| 2024 | **SMooDi: Stylized Motion Diffusion Model**  | ECCV 2024  |          [Link](https://arxiv.org/abs/2407.12783)          | [Link](https://github.com/neu-vi/SMooDi)  | [Link](https://neu-vi.github.io/SMooDi/) |
| 2024 | **EMDM: Efficient Motion Diffusion Model for Fast, High-Quality Human Motion Generation**  | ECCV 2024  |          [Link](https://arxiv.org/abs/2312.02256)          | [Link](https://github.com/Frank-ZY-Dou/EMDM) | [Link](https://frank-zy-dou.github.io/projects/EMDM/index.html)  |
| 2024 | **Plan, Posture and Go: Towards Open-World Text-to-Motion Generation**  | ECCV 2024  |          [Link](https://arxiv.org/abs/2312.14828)          | [Link](https://github.com/moonsliu/Pro-Motion) | [Link](https://moonsliu.github.io/Pro-Motion/)  |
| 2024 | **Generating Human Interaction Motions in Scenes with Text Control**  | ECCV 2024  |       [Link](https://arxiv.org/abs/2404.10685)          | -- | [Link](https://research.nvidia.com/labs/toronto-ai/tesmo/)  |
| 2024 | **SATO: Stable Text-to-Motion Framework**  | ACM MULTIMEDIA 2024 |          [Link](https://arxiv.org/abs/2405.01461)          | [Link](https://github.com/sato-team/Stable-Text-to-motion-Framework) | [Link](https://sato-team.github.io/Stable-Text-to-Motion-Framework/)  |
| 2024 | **MotionFix: Text-Driven 3D Human Motion Editing**  | SIGGRAPH Asia 2024 |          [Link](https://arxiv.org/abs/2408.00712)          | [Link](https://github.com/athn-nik/motionfix) | [Link](https://motionfix.is.tue.mpg.de/)  |
| 2024 | **Autonomous Character-Scene Interaction Synthesis from Text Instruction**  | SIGGRAPH Asia 2024 |          [Link](https://arxiv.org/abs/2410.03187)          | -- | [Link](https://lingomotions.com/)  |

<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

@inproceedings{
tevet2023human,
title={Human Motion Diffusion Model},
author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=SJ1kSyO2jwu}
}

@article{jiang2024motiongpt,
    title={MotionGPT: Human Motion as a Foreign Language},
    author={Jiang, Biao and Chen, Xin and Liu, Wen and Yu, Jingyi and Yu, Gang and Chen, Tao},
    journal={Advances in Neural Information Processing Systems},
    volume={36},
    year={2024}
}

@inproceedings{chen2023executing,
  title     = {Executing your Commands via Motion Diffusion in Latent Space},
  author    = {Chen, Xin and Jiang, Biao and Liu, Wen and Huang, Zilong and Fu, Bin and Chen, Tao and Yu, Gang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {18000--18010},
  year      = {2023},
}

@misc{hoang2024motionmix,
  title={MotionMix: Weakly-Supervised Diffusion for Controllable Motion Generation}, 
  author={Nhat M. Hoang and Kehong Gong and Chuan Guo and Michael Bi Mi},
  year={2024},
  eprint={2401.11115},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@inproceedings{raab2024single,
            title={Single Motion Diffusion},
            author={Raab, Sigal and Leibovitch, Inbal and Tevet, Guy and Arar, Moab and Bermano, Amit H and Cohen-Or, Daniel},
            booktitle={The Twelfth International Conference on Learning Representations (ICLR)},             
            year={2024}
}

@article{guo2023momask,
      title={MoMask: Generative Masked Modeling of 3D Human Motions}, 
      author={Chuan Guo and Yuxuan Mu and Muhammad Gohar Javed and Sen Wang and Li Cheng},
      year={2023},
      eprint={2312.00063},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{yatim2023spacetime,
        title = {Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer},
        author = {Yatim, Danah and Fridman, Rafail and Bar-Tal, Omer and Kasten, Yoni and Dekel, Tali},
        journal={arXiv preprint arxiv:2311.17009},
        year={2023}
}

@inproceedings{petrovich24stmc,
    title     = {Multi-Track Timeline Control for Text-Driven 3D Human Motion Generation},
    author    = {Petrovich, Mathis and Litany, Or and Iqbal, Umar and Black, Michael J. and Varol, G{\"u}l and Peng, Xue Bin and Rempe, Davis},
    booktitle = {CVPR Workshop on Human Motion Generation},
    year      = {2024}
}

@misc{ponce2024in2in,
      title={in2IN: Leveraging individual Information to Generate Human INteractions}, 
      author={Pablo Ruiz Ponce and German Barquero and Cristina Palmero and Sergio Escalera and Jose Garcia-Rodriguez},
      year={2024},
      eprint={2404.09988},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{sheng2024exploring,
      title={Exploring Text-to-Motion Generation with Human Preference}, 
      author={Jenny Sheng and Matthieu Lin and Andrew Zhao and Kevin Pruvost and Yu-Hui Wen and Yangguang Li and Gao Huang and Yong-Jin Liu},
      year={2024},
      eprint={2404.09445},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@article{barquero2024seamless,
  title={Seamless Human Motion Composition with Blended Positional Encodings},
  author={Barquero, German and Escalera, Sergio and Palmero, Cristina},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

@inproceedings{wang2024move,
  title={Move as You Say, Interact as You Can: Language-guided Human Motion Generation with Scene Affordance},
  author={Wang, Zan and Chen, Yixin and Jia, Baoxiong and Li, Puhao and Zhang, Jinlu and Zhang, Jingze and Liu, Tengyu and Zhu, Yixin and Liang, Wei and Huang, Siyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

@inproceedings{cen2024text_scene_motion,
  title={Generating Human Motion in 3D Scenes from Text Descriptions},
  author={Cen, Zhi and Pi, Huaijin and Peng, Sida and Shen, Zehong and Yang, Minghui and Shuai, Zhu and Bao, Hujun and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2024}
}

@inproceedings{yang2024omnimotiongpt,
  title={OmniMotionGPT: Animal Motion Generation with Limited Data},
  author={Yang, Zhangsihao and Zhou, Mingyuan and Shan, Mengyi and Wen, Bingbing and Xuan, Ziwei and Hill, Mitch and Bai, Junjie and Qi, Guo-Jun and Wang, Yalin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1249--1259},
  year={2024}
}

@article{humantomato,
  title={HumanTOMATO: Text-aligned Whole-body Motion Generation},
  author={Lu, Shunlin and Chen, Ling-Hao and Zeng, Ailing and Lin, Jing and Zhang, Ruimao and Zhang, Lei and Shum, Heung-Yeung},
  journal={arxiv:2310.12978},
  year={2023}
}

@misc{gillman2024selfcorrecting,
  title={Self-Correcting Self-Consuming Loops for Generative Model Training}, 
  author={Nate Gillman and Michael Freeman and Daksh Aggarwal and Chia-Hong Hsu and Calvin Luo and Yonglong Tian and Chen Sun},
  year={2024},
  eprint={2402.07087},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{cohan2024flexible,
      title={Flexible Motion In-betweening with Diffusion Models}, 
      author={Setareh Cohan and Guy Tevet and Daniele Reda and Xue Bin Peng and Michiel van de Panne},
      year={2024},
      eprint={2405.11126},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{Goel_2024, series={SIGGRAPH ’24},
   title={Iterative Motion Editing with Natural Language},
   url={http://dx.doi.org/10.1145/3641519.3657447},
   DOI={10.1145/3641519.3657447},
   booktitle={Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers ’24},
   publisher={ACM},
   author={Goel, Purvi and Wang, Kuan-Chieh and Liu, C. Karen and Fatahalian, Kayvon},
   year={2024},
   month=jul, collection={SIGGRAPH ’24} }


@article{motionlcm,
      title={MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model},
      author={Wenxun Dai and Ling-Hao Chen and Jingbo Wang and Jinpeng Liu and Bo Dai and Yansong Tang},
      journal={arXiv preprint arXiv:2404.19759},
      year={2024}
}

@misc{zou2024parcopartcoordinatingtexttomotionsynthesis,
      title={ParCo: Part-Coordinating Text-to-Motion Synthesis}, 
      author={Qiran Zou and Shangyuan Yuan and Shian Du and Yu Wang and Chang Liu and Yi Xu and Jie Chen and Xiangyang Ji},
      year={2024},
      eprint={2403.18512},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.18512}, 
}

@misc{huang2024como,
      title={CoMo: Controllable Motion Generation through Language Guided Pose Code Editing}, 
      author={Yiming Huang and Weilin Wan and Yue Yang and Chris Callison-Burch and Mark Yatskar and Lingjie Liu},
      year={2024},
      eprint={2403.13900},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{zhong2024smoodistylizedmotiondiffusion,
      title={SMooDi: Stylized Motion Diffusion Model}, 
      author={Lei Zhong and Yiming Xie and Varun Jampani and Deqing Sun and Huaizu Jiang},
      year={2024},
      eprint={2407.12783},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.12783}, 
}

@article{zhou2023emdm,
  title={EMDM: Efficient Motion Diffusion Model for Fast, High-Quality Motion Generation},
  author={Zhou, Wenyang and Dou, Zhiyang and Cao, Zeyu and Liao, Zhouyingcheng and Wang, Jingbo and Wang, Wenjia and Liu, Yuan and Komura, Taku and Wang, Wenping and Liu, Lingjie},
  journal={arXiv preprint arXiv:2312.02256},
  year={2023}
}

@article{liu2023plan,
  title={Plan, Posture and Go: Towards Open-World Text-to-Motion Generation},
  author={Liu, Jinpeng and Dai, Wenxun and Wang, Chunyu and Cheng, Yiji and Tang, Yansong and Tong, Xin},
  journal={arXiv preprint arXiv:2312.14828},
  year={2023}
}

@article{yi2024tesmo,
    author={Yi, Hongwei and Thies, Justus and Black, Michael J. and Peng, Xue Bin and Rempe, Davis},
    title={Generating Human Interaction Motions in Scenes with Text Control},
    journal = {arXiv:2404.10685},
    year={2024}
}

@misc{chen2024sato,
      title={SATO: Stable Text-to-Motion Framework}, 
      author={Wenshuo Chen and Hongru Xiao and Erhang Zhang and Lijie Hu and Lei Wang and Mengyuan Liu and Chen Chen},
      year={2024},
      eprint={2405.01461},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{athanasiou2024motionfix,
  title = {{MotionFix}: Text-Driven 3D Human Motion Editing},
  author = {Athanasiou, Nikos and Ceske, Alpar and Diomataris, Markos and Black, Michael J. and Varol, G{\"u}l},
  booktitle = {SIGGRAPH Asia 2024 Conference Papers},
  year = {2024}
}

@misc{jiang2024autonomouscharactersceneinteractionsynthesis,
      title={Autonomous Character-Scene Interaction Synthesis from Text Instruction}, 
      author={Nan Jiang and Zimo He and Zi Wang and Hongjie Li and Yixin Chen and Siyuan Huang and Yixin Zhu},
      year={2024},
      eprint={2410.03187},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.03187}, 
}
```
</details>

---

### 💡 Motion ArXiv Papers

#### 1. Story-to-Motion: Synthesizing Infinite and Controllable Character Animation from Long Text  
Zhongfei Qing, Zhongang Cai, Zhitao Yang, Lei Yang (SenseTime)
<details span>
<summary><b>Abstract</b></summary>
Generating natural human motion from a story has the potential to transform the landscape of animation, gaming, and film industries. A new and challenging task, Story-to-Motion, arises when characters are required to move to various locations and perform specific motions based on a long text description. This task demands a fusion of low-level control (trajectories) and high-level control (motion semantics). Previous works in character control and text-to-motion have addressed related aspects, yet a comprehensive solution remains elusive: character control methods do not handle text description, whereas text-to-motion methods lack position constraints and often produce unstable motions. In light of these limitations, we propose a novel system that generates controllable, infinitely long motions and trajectories aligned with the input text. 1) we leverage contemporary Large Language Models to act as a text-driven motion scheduler to extract a series of (text, position) pairs from long text. 2) we develop a text-driven motion retrieval scheme that incorporates classic motion matching with motion semantic and trajectory constraints. 3) we design a progressive mask transformer that addresses common artifacts in the transition motion such as unnatural pose and foot sliding. Beyond its pioneering role as the first comprehensive solution for Story-to-Motion, our system undergoes evaluation across three distinct sub-tasks: trajectory following, temporal action composition, and motion blending, where it outperforms previous state-of-the-art (SOTA) motion synthesis methods across the board.
</details>

#### 2. Synthesizing Moving People with 3D Control  
Boyi Li, Jathushan Rajasegaran, Yossi Gandelsman, Alexei A. Efros, Jitendra Malik (UC Berkeley)
<details span>
<summary><b>Abstract</b></summary>
In this paper, we present a diffusion model-based framework for animating people from a single image for a given target 3D motion sequence. Our approach has two core components: a) learning priors about invisible parts of the human body and clothing, and b) rendering novel body poses with proper clothing and texture. For the first part, we learn an in-filling diffusion model to hallucinate unseen parts of a person given a single image. We train this model on texture map space, which makes it more sample-efficient since it is invariant to pose and viewpoint. Second, we develop a diffusion-based rendering pipeline, which is controlled by 3D human poses. This produces realistic renderings of novel poses of the person, including clothing, hair, and plausible in-filling of unseen regions. This disentangled approach allows our method to generate a sequence of images that are faithful to the target motion in the 3D pose and, to the input image in terms of visual similarity. In addition to that, the 3D control allows various synthetic camera trajectories to render a person. Our experiments show that our method is resilient in generating prolonged motions and varied challenging and complex poses compared to prior methods. 
</details>

#### 3. Large Motion Model for Unified Multi-Modal Motion Generation
Mingyuan Zhang, Daisheng Jin, Chenyang Gu, Fangzhou Hong, Zhongang Cai, Jingfang Huang, Chongzhi Zhang, Xinying Guo, Lei Yang, Ying He, Ziwei Liu 

(S-Lab, Nanyang Technological University, SenseTime China)
<details span>
<summary><b>Abstract</b></summary>
Human motion generation, a cornerstone technique in animation and video production, has widespread applications in various tasks like text-to-motion and music-to-dance. Previous works focus on developing specialist models tailored for each task without scalability. In this work, we present Large Motion Model (LMM), a motion-centric, multi-modal framework that unifies mainstream motion generation tasks into a generalist model. A unified motion model is appealing since it can leverage a wide range of motion data to achieve broad generalization beyond a single task. However, it is also challenging due to the heterogeneous nature of substantially different motion data and tasks. LMM tackles these challenges from three principled aspects: 1) Data: We consolidate datasets with different modalities, formats and tasks into a comprehensive yet unified motion generation dataset, MotionVerse, comprising 10 tasks, 16 datasets, a total of 320k sequences, and 100 million frames. 2) Architecture: We design an articulated attention mechanism ArtAttention that incorporates body part-aware modeling into Diffusion Transformer backbone. 3) Pre-Training: We propose a novel pre-training strategy for LMM, which employs variable frame rates and masking forms, to better exploit knowledge from diverse training data. Extensive experiments demonstrate that our generalist LMM achieves competitive performance across various standard motion generation tasks over state-of-the-art specialist models. Notably, LMM exhibits strong generalization capabilities and emerging properties across many unseen tasks. Additionally, our ablation studies reveal valuable insights about training and scaling up large motion models for future research.
</details>

#### 4. StableMoFusion: Towards Robust and Efficient Diffusion-based Motion Generation Framework
Yiheng Huang, Hui Yang, Chuanchen Luo, Yuxi Wang, Shibiao Xu, Zhaoxiang Zhang, Man Zhang, Junran Peng

(Beijing University of Posts and Telecommunications, CAIR/HKISI/CAS, Institute of Automation/Chinese Academy of Science)
<details span>
<summary><b>Abstract</b></summary>
Thanks to the powerful generative capacity of diffusion models, recent years have witnessed rapid progress in human motion generation. Existing diffusion-based methods employ disparate network architectures and training strategies. The effect of the design of each component is still unclear. In addition, the iterative denoising process consumes considerable computational overhead, which is prohibitive for real-time scenarios such as virtual characters and humanoid robots. For this reason, we first conduct a comprehensive investigation into network architectures, training strategies, and inference processs. Based on the profound analysis, we tailor each component for efficient high-quality human motion generation. Despite the promising performance, the tailored model still suffers from foot skating which is an ubiquitous issue in diffusion-based solutions. To eliminate footskate, we identify foot-ground contact and correct foot motions along the denoising process. By organically combining these well-designed components together, we present StableMoFusion, a robust and efficient framework for human motion generation. Extensive experimental results show that our StableMoFusion performs favorably against current state-of-the-art methods.
</details>

#### 5. CrowdMoGen: Zero-Shot Text-Driven Collective Motion Generation
Xinying Guo, Mingyuan Zhang, Haozhe Xie, Chenyang Gu, Ziwei Liu (S-Lab Nanyang Technological University)
<details span>
<summary><b>Abstract</b></summary>
Crowd Motion Generation is essential in entertainment industries such as animation and games as well as in strategic fields like urban simulation and planning. This new task requires an intricate integration of control and generation to realistically synthesize crowd dynamics under specific spatial and semantic constraints, whose challenges are yet to be fully explored. On the one hand, existing human motion generation models typically focus on individual behaviors, neglecting the complexities of collective behaviors. On the other hand, recent methods for multi-person motion generation depend heavily on pre-defined scenarios and are limited to a fixed, small number of inter-person interactions, thus hampering their practicality. To overcome these challenges, we introduce CrowdMoGen, a zero-shot text-driven framework that harnesses the power of Large Language Model (LLM) to incorporate the collective intelligence into the motion generation framework as guidance, thereby enabling generalizable planning and generation of crowd motions without paired training data. Our framework consists of two key components: 1) Crowd Scene Planner that learns to coordinate motions and dynamics according to specific scene contexts or introduced perturbations, and 2) Collective Motion Generator that efficiently synthesizes the required collective motions based on the holistic plans. Extensive quantitative and qualitative experiments have validated the effectiveness of our framework, which not only fills a critical gap by providing scalable and generalizable solutions for Crowd Motion Generation task but also achieves high levels of realism and flexibility.
</details>

#### 6. Infinite Motion: Extended Motion Generation via Long Text Instructions
Mengtian Li, Chengshuo Zhai, Shengxiang Yao, Zhifeng Xie, Keyu Chen Yu-Gang Jiang

(Shanghai University, Shanghai Engineering Research Center of Motion Picture Special Effects, Tavus Inc., Fudan University)
<details span>
<summary><b>Abstract</b></summary>
In the realm of motion generation, the creation of long-duration, high-quality motion sequences remains a significant challenge. This paper presents our groundbreaking work on "Infinite Motion", a novel approach that leverages long text to extended motion generation, effectively bridging the gap between short and long-duration motion synthesis. Our core insight is the strategic extension and reassembly of existing high-quality text-motion datasets, which has led to the creation of a novel benchmark dataset to facilitate the training of models for extended motion sequences. A key innovation of our model is its ability to accept arbitrary lengths of text as input, enabling the generation of motion sequences tailored to specific narratives or scenarios. Furthermore, we incorporate the timestamp design for text which allows precise editing of local segments within the generated sequences, offering unparalleled control and flexibility in motion synthesis. We further demonstrate the versatility and practical utility of "Infinite Motion" through three specific applications: natural language interactive editing, motion sequence editing within long sequences and splicing of independent motion sequences. Each application highlights the adaptability of our approach and broadens the spectrum of possibilities for research and development in motion generation. Through extensive experiments, we demonstrate the superior performance of our model in generating long sequence motions compared to existing methods.
</details>

#### 7. Adding Multi-modal Controls to Whole-body Human Motion Generation
Yuxuan Bian, Ailing Zeng, Xuan Ju, Xian Liu, Zhaoyang Zhang, Wei Liu, Qiang Xu

(The Chinese University of Hong Kong, Tencent)
<details span>
<summary><b>Abstract</b></summary>
Whole-body multi-modal motion generation, controlled by text, speech, or music, has numerous applications including video generation and character animation. However, employing a unified model to accomplish various generation tasks with different condition modalities presents two main challenges: motion distribution drifts across different generation scenarios and the complex optimization of mixed conditions with varying granularity. Furthermore, inconsistent motion formats in existing datasets further hinder effective multi-modal motion generation. In this paper, we propose ControlMM, a unified framework to Control whole-body Multi-modal Motion generation in a plug-and-play manner. To effectively learn and transfer motion knowledge across different motion distributions, we propose ControlMM-Attn, for parallel modeling of static and dynamic human topology graphs. To handle conditions with varying granularity, ControlMM employs a coarse-to-fine training strategy, including stage-1 text-to-motion pre-training for semantic generation and stage-2 multi-modal control adaptation for conditions of varying low-level granularity. To address existing benchmarks' varying motion format limitations, we introduce ControlMM-Bench, the first publicly available multi-modal whole-body human motion generation benchmark based on the unified whole-body SMPL-X format. Extensive experiments show that ControlMM achieves state-of-the-art performance across various standard motion generation tasks. 
</details>

#### 8. MoRAG -- Multi-Fusion Retrieval Augmented Generation for Human Motion
Kalakonda Sai Shashank, Shubh Maheshwari, Ravi Kiran Sarvadevabhatla

(IIIT Hyderabad, University of California San Diego)
<details span>
<summary><b>Abstract</b></summary>
We introduce MoRAG, a novel multi-part fusion based retrieval-augmented generation strategy for text-based human motion generation. The method enhances motion diffusion models by leveraging additional knowledge obtained through an improved motion retrieval process. By effectively prompting large language models (LLMs), we address spelling errors and rephrasing issues in motion retrieval. Our approach utilizes a multi-part retrieval strategy to improve the generalizability of motion retrieval across the language space. We create diverse samples through the spatial composition of the retrieved motions. Furthermore, by utilizing low-level, part-specific motion information, we can construct motion samples for unseen text descriptions. Our experiments demonstrate that our framework can serve as a plug-and-play module, improving the performance of motion diffusion models.
</details>

#### 9. T2M-X: Learning Expressive Text-to-Motion Generation from Partially Annotated Data
Mingdian Liu, Yilin Liu, Gurunandan Krishnan, Karl S Bayer, Bing Zhou

(Iowa State University, Pennsylvania State University, Snap Inc.)
<details span>
<summary><b>Abstract</b></summary>
The generation of humanoid animation from text prompts can profoundly impact animation production and AR/VR experiences. However, existing methods only generate body motion data, excluding facial expressions and hand movements. This limitation, primarily due to a lack of a comprehensive whole-body motion dataset, inhibits their readiness for production use. Recent attempts to create such a dataset have resulted in either motion inconsistency among different body parts in the artificially augmented data or lower quality in the data extracted from RGB videos. In this work, we propose T2M-X, a two-stage method that learns expressive text-to-motion generation from partially annotated data. T2M-X trains three separate Vector Quantized Variational AutoEncoders (VQ-VAEs) for body, hand, and face on respective high-quality data sources to ensure high-quality motion outputs, and a Multi-indexing Generative Pretrained Transformer (GPT) model with motion consistency loss for motion generation and coordination among different body parts. Our results show significant improvements over the baselines both quantitatively and qualitatively, demonstrating its robustness against the dataset limitations.
</details>

#### 10. UniMuMo: Unified Text, Music and Motion Generation
Han Yang, Kun Su, Yutong Zhang, Jiaben Chen, Kaizhi Qian, Gaowen Liu, Chuang Gan

(The Chinese University of Hong Kong, University of Washington, The University of British Columbia, UMass Amherst, MIT-IBM Watson AI Lab, Cisco Research)
<details span>
<summary><b>Abstract</b></summary>
We introduce UniMuMo, a unified multimodal model capable of taking arbitrary text, music, and motion data as input conditions to generate outputs across all three modalities. To address the lack of time-synchronized data, we align unpaired music and motion data based on rhythmic patterns to leverage existing large-scale music-only and motion-only datasets. By converting music, motion, and text into token-based representation, our model bridges these modalities through a unified encoder-decoder transformer architecture. To support multiple generation tasks within a single framework, we introduce several architectural improvements. We propose encoding motion with a music codebook, mapping motion into the same feature space as music. We introduce a music-motion parallel generation scheme that unifies all music and motion generation tasks into a single transformer decoder architecture with a single training task of music-motion joint generation. Moreover, the model is designed by fine-tuning existing pre-trained single-modality models, significantly reducing computational demands. Extensive experiments demonstrate that UniMuMo achieves competitive results on all unidirectional generation benchmarks across music, motion, and text modalities.
</details>

#### 11. DART: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control
Kaifeng Zhao, Gen Li, Siyu Tang (ETH Zürich)
<details span>
<summary><b>Abstract</b></summary>
Text-conditioned human motion generation, which allows for user interaction through natural language, has become increasingly popular. Existing methods typically generate short, isolated motions based on a single input sentence. However, human motions are continuous and can extend over long periods, carrying rich semantics. Creating long, complex motions that precisely respond to streams of text descriptions, particularly in an online and real-time setting, remains a significant challenge. Furthermore, incorporating spatial constraints into text-conditioned motion generation presents additional challenges, as it requires aligning the motion semantics specified by text descriptions with geometric information, such as goal locations and 3D scene geometry. To address these limitations, we propose DART, a Diffusion-based Autoregressive motion primitive model for Real-time Text-driven motion control. Our model, DART, effectively learns a compact motion primitive space jointly conditioned on motion history and text inputs using latent diffusion models. By autoregressively generating motion primitives based on the preceding history and current text input, DART enables real-time, sequential motion generation driven by natural language descriptions. Additionally, the learned motion primitive space allows for precise spatial motion control, which we formulate either as a latent noise optimization problem or as a Markov decision process addressed through reinforcement learning. We present effective algorithms for both approaches, demonstrating our model's versatility and superior performance in various motion synthesis tasks. Experiments show our method outperforms existing baselines in motion realism, efficiency, and controllability.
</details>

#### 12. ControlMM: Controllable Masked Motion Generation
Ekkasit Pinyoanuntapong, Muhammad Usama Saleem, Korrawe Karunratanakul, Pu Wang, Hongfei Xue, Chen Chen, Chuan Guo, Junli Cao, Jian Ren, Sergey Tulyakov

(University of North Carolina at Charlotte, ETH Zurich, University of Central Florida, Snap Inc.)
<details span>
<summary><b>Abstract</b></summary>
Recent advances in motion diffusion models have enabled spatially controllable text-to-motion generation. However, despite achieving acceptable control precision, these models suffer from generation speed and fidelity limitations. To address these challenges, we propose ControlMM, a novel approach incorporating spatial control signals into the generative masked motion model. ControlMM achieves real-time, high-fidelity, and high-precision controllable motion generation simultaneously. Our approach introduces two key innovations. First, we propose masked consistency modeling, which ensures high-fidelity motion generation via random masking and reconstruction, while minimizing the inconsistency between the input control signals and the extracted control signals from the generated motion. To further enhance control precision, we introduce inference-time logit editing, which manipulates the predicted conditional motion distribution so that the generated motion, sampled from the adjusted distribution, closely adheres to the input control signals. During inference, ControlMM enables parallel and iterative decoding of multiple motion tokens, allowing for high-speed motion generation. Extensive experiments show that, compared to the state of the art, ControlMM delivers superior results in motion quality, with better FID scores (0.061 vs 0.271), and higher control precision (average error 0.0091 vs 0.0108). ControlMM generates motions 20 times faster than diffusion-based methods. Additionally, ControlMM unlocks diverse applications such as any joint any frame control, body part timeline control, and obstacle avoidance.
</details>

#### 13. MotionCLR: Motion Generation and Training-free Editing via Understanding Attention Mechanisms
Ling-Hao Chen, Wenxun Dai, Xuan Ju, Shunlin Lu, Lei Zhang

(Tsinghua University, International Digital Economy Academy (IDEA), The Chinese University of Hong Kong, The Chinese University of Hong Kong Shenzhen)
<details span>
<summary><b>Abstract</b></summary>
This research delves into the problem of interactive editing of human motion generation. Previous motion diffusion models lack explicit modeling of the word-level text-motion correspondence and good explainability, hence restricting their fine-grained editing ability. To address this issue, we propose an attention-based motion diffusion model, namely MotionCLR, with CLeaR modeling of attention mechanisms. Technically, MotionCLR models the in-modality and cross-modality interactions with self-attention and cross-attention, respectively. More specifically, the self-attention mechanism aims to measure the sequential similarity between frames and impacts the order of motion features. By contrast, the cross-attention mechanism works to find the fine-grained word-sequence correspondence and activate the corresponding timesteps in the motion sequence. Based on these key properties, we develop a versatile set of simple yet effective motion editing methods via manipulating attention maps, such as motion (de-)emphasizing, in-place motion replacement, and example-based motion generation, etc. For further verification of the explainability of the attention mechanism, we additionally explore the potential of action-counting and grounded motion generation ability via attention maps. Our experimental results show that our method enjoys good generation and editing ability with good explainability.
</details>

#### 14. KMM: Key Frame Mask Mamba for Extended Motion Generation
Zeyu Zhang, Hang Gao, Akide Liu, Qi Chen, Feng Chen, Yiran Wang, Danning Li, Hao Tang

(Peking University, The Australian National University, Monash University, The University of Adelaide, The University of Sydney, McGill University)
<details span>
<summary><b>Abstract</b></summary>
Human motion generation is a cut-edge area of research in generative computer vision, with promising applications in video creation, game development, and robotic manipulation. The recent Mamba architecture shows promising results in efficiently modeling long and complex sequences, yet two significant challenges remain: Firstly, directly applying Mamba to extended motion generation is ineffective, as the limited capacity of the implicit memory leads to memory decay. Secondly, Mamba struggles with multimodal fusion compared to Transformers, and lack alignment with textual queries, often confusing directions (left or right) or omitting parts of longer text queries. To address these challenges, our paper presents three key contributions: Firstly, we introduce KMM, a novel architecture featuring Key frame Masking Modeling, designed to enhance Mamba's focus on key actions in motion segments. This approach addresses the memory decay problem and represents a pioneering method in customizing strategic frame-level masking in SSMs. Additionally, we designed a contrastive learning paradigm for addressing the multimodal fusion problem in Mamba and improving the motion-text alignment. Finally, we conducted extensive experiments on the go-to dataset, BABEL, achieving state-of-the-art performance with a reduction of more than 57% in FID and 70% parameters compared to previous state-of-the-art methods. 
</details>

#### 15. KinMo: Kinematic-aware Human Motion Understanding and Generation
Pengfei Zhang, Pinxin Liu, Hyeongwoo Kim, Pablo Garrido, Bindita Chaudhuri

(University of California Irvine, University of Rochester, Imperial College London, FlawlessAI)
<details span>
<summary><b>Abstract</b></summary>
Controlling human motion based on text presents an important challenge in computer vision. Traditional approaches often rely on holistic action descriptions for motion synthesis, which struggle to capture subtle movements of local body parts. This limitation restricts the ability to isolate and manipulate specific movements. To address this, we propose a novel motion representation that decomposes motion into distinct body joint group movements and interactions from a kinematic perspective. We design an automatic dataset collection pipeline that enhances the existing text-motion benchmark by incorporating fine-grained local joint-group motion and interaction descriptions. To bridge the gap between text and motion domains, we introduce a hierarchical motion semantics approach that progressively fuses joint-level interaction information into the global action-level semantics for modality alignment. With this hierarchy, we introduce a coarse-to-fine motion synthesis procedure for various generation and editing downstream applications. Our quantitative and qualitative experiments demonstrate that the proposed formulation enhances text-motion retrieval by improving joint-spatial understanding, and enables more precise joint-motion generation and control.
</details>

---

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **Story-to-Motion: Synthesizing Infinite and Controllable Character Animation from Long Text**  | 13 Nov 2023 |          [Link](https://arxiv.org/abs/2311.07446)          | -- | [Link](https://story2motion.github.io/)  |
| 2024 | **Synthesizing Moving People with 3D Control**  | 19 Jan 2024 |          [Link](https://arxiv.org/abs/2401.10889)          | [Link](https://github.com/Boyiliee/3DHM)   | [Link](https://boyiliee.github.io/3DHM.github.io/)  |
| 2024 | **Large Motion Model for Unified Multi-Modal Motion Generation**  | 1 Apr 2024 |          [Link](https://arxiv.org/abs/2404.01284)          | [Link](https://github.com/mingyuan-zhang/LMM) | [Link](https://mingyuan-zhang.github.io/projects/LMM.html)  |
| 2024 | **StableMoFusion: Towards Robust and Efficient Diffusion-based Motion Generation Framework**  | 9 May 2024 |          [Link](https://arxiv.org/abs/2405.05691)          | [Link](https://github.com/h-y1heng/StableMoFusion) | [Link](https://h-y1heng.github.io/StableMoFusion-page/)  |
| 2024 | **CrowdMoGen: Zero-Shot Text-Driven Collective Motion Generation**  | 8 Jul 2024 |          [Link](https://arxiv.org/abs/2407.06188)          | [Link](https://github.com/gxyes/CrowdMoGen) | [Link](https://gxyes.github.io/projects/CrowdMoGen.html)  |
| 2024 | **Infinite Motion: Extended Motion Generation via Long Text Instructions**  | 11 Jul 2024 |          [Link](https://arxiv.org/abs/2407.08443)          | [Link](https://github.com/shuochengzhai/Infinite-Motion) | [Link](https://shuochengzhai.github.io/Infinite-motion.github.io/)  |
| 2024 | **Adding Multi-modal Controls to Whole-body Human Motion Generation**  | 30 Jul 2024 |          [Link](https://arxiv.org/abs/2407.21136)          | [Link](https://github.com/yxbian23/ControlMM) | [Link](https://yxbian23.github.io/ControlMM/)  |
| 2024 | **MoRAG -- Multi-Fusion Retrieval Augmented Generation for Human Motion**  | 18 Sep 2024 |          [Link](https://arxiv.org/abs/2409.12140)          | [Link](https://github.com/Motion-RAG/MoRAG) | [Link](https://motion-rag.github.io/)  |
| 2024 | **T2M-X: Learning Expressive Text-to-Motion Generation from Partially Annotated Data**  | 20 Sep 2024 |          [Link](https://arxiv.org/abs/2409.13251)          | -- | -- |
| 2024 | **UniMuMo: Unified Text, Music, and Motion Generation**  | 6 Oct 2024 |          [Link](https://arxiv.org/abs/2410.04534)          | [Link](https://github.com/hanyangclarence/UniMuMo) | [Link](https://hanyangclarence.github.io/unimumo_demo/)  |
| 2024 | **DART: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control**  | 7 Oct 2024 |          [Link](https://arxiv.org/abs/2410.05260)          | -- | [Link](https://zkf1997.github.io/DART/)  |
| 2024 | **ControlMM: Controllable Masked Motion Generation**  | 14 Oct 2024 |          [Link](https://arxiv.org/abs/2410.10780)          | [Link](https://github.com/exitudio/ControlMM/) | [Link](https://exitudio.github.io/ControlMM-page/)  |
| 2024 | **MotionCLR: Motion Generation and Training-free Editing via Understanding Attention Mechanisms**  | 24 Oct 2024 |          [Link](https://arxiv.org/abs/2410.18977)          | [Link](https://github.com/IDEA-Research/MotionCLR) | [Link](https://lhchen.top/MotionCLR/)  |
| 2024 | **KMM: Key Frame Mask Mamba for Extended Motion Generation**  | 10 Nov 2024 |          [Link](https://arxiv.org/abs/2411.06481)          | [Link](https://github.com/steve-zeyu-zhang/KMM) | [Link](https://steve-zeyu-zhang.github.io/KMM/)  |
| 2024 | **KinMo: Kinematic-aware Human Motion Understanding and Generation**  | 23 Nov 2024 |          [Link](https://arxiv.org/abs/2411.15472)          | -- | [Link](https://andypinxinliu.github.io/KinMo/)  |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@misc{qing2023storytomotion,
        title={Story-to-Motion: Synthesizing Infinite and Controllable Character Animation from Long Text}, 
        author={Zhongfei Qing and Zhongang Cai and Zhitao Yang and Lei Yang},
        year={2023},
        eprint={2311.07446},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
}

@article{li20243dhm,
    author = {Li, Boyi and Rajasegaran, Jathushan and Gandelsman, Yossi and Efros, Alexei A. and Malik, Jitendra},
    title = {Synthesizing Moving People with 3D Control},
    journal = {Arxiv},
    year = {2024},
}

@article{zhang2024large,
      title   =   {Large Motion Model for Unified Multi-Modal Motion Generation}, 
      author  =   {Zhang, Mingyuan and
                   Jin, Daisheng around
                   Gu, Chenyang,
                   Hong, Fangzhou and
                   Cai, Zhongang and
                   Huang, Jingfang and
                   Zhang, Chongzhi and
                   Guo, Xinying and
                   Yang, Lei and,
                   He, Ying and,
                   Liu, Ziwei},
      year    =   {2024},
      journal =   {arXiv preprint arXiv:2404.01284},
}

@article{huang2024stablemofusion,
        title={StableMoFusion: Towards Robust and Efficient Diffusion-based Motion Generation Framework},
        author = {Huang, Yiheng and Hui, Yang and Luo, Chuanchen and Wang, Yuxi and Xu, Shibiao and Zhang, Zhaoxiang and Zhang, Man and Peng, Junran},
        journal = {arXiv preprint arXiv: 2405.05691},
        year = {2024}
}

@misc{guo2024crowdmogenzeroshottextdrivencollective,
      title={CrowdMoGen: Zero-Shot Text-Driven Collective Motion Generation}, 
      author={Xinying Guo and Mingyuan Zhang and Haozhe Xie and Chenyang Gu and Ziwei Liu},
      year={2024},
      eprint={2407.06188},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.06188}, 
}

@misc{li2024infinitemotionextendedmotion,
      title={Infinite Motion: Extended Motion Generation via Long Text Instructions}, 
      author={Mengtian Li and Chengshuo Zhai and Shengxiang Yao and Zhifeng Xie and Keyu Chen Yu-Gang Jiang},
      year={2024},
      eprint={2407.08443},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.08443}, 
}

@article{controlmm,
  title={Adding Multimodal Controls to Whole-body Human Motion Generation},
  author={Bian, Yuxuan, Zeng Ailing, Ju Xuan, Liu Xian, Zhang Zhaoyang, Liu Wei, and Xu Qiang},
  journal={arxiv},
  year={2024}
}

@InProceedings{MoRAG,
  author    = {Kalakonda, Sai Shashank and Maheshwari, Shubh and Sarvadevabhatla, Ravi Kiran},
  title     = {MoRAG - Multi-Fusion Retrieval Augmented Generation for Human Motion},
  booktitle   = {arXiv preprint},
  year      = {2024},
}

@misc{liu2024t2mxlearningexpressivetexttomotion,
      title={T2M-X: Learning Expressive Text-to-Motion Generation from Partially Annotated Data}, 
      author={Mingdian Liu and Yilin Liu and Gurunandan Krishnan and Karl S Bayer and Bing Zhou},
      year={2024},
      eprint={2409.13251},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.13251}, 
}

@misc{yang2024unimumounifiedtextmusic,
      title={UniMuMo: Unified Text, Music and Motion Generation}, 
      author={Han Yang and Kun Su and Yutong Zhang and Jiaben Chen and Kaizhi Qian and Gaowen Liu and Chuang Gan},
      year={2024},
      eprint={2410.04534},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2410.04534}, 
}

@inproceedings{Zhao:DART:2024,
   title = {A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control},
   author = {Zhao, Kaifeng and Li, Gen and Tang, Siyu},
   year = {2024}
}

@misc{pinyoanuntapong2024controlmmcontrollablemaskedmotion,
      title={ControlMM: Controllable Masked Motion Generation}, 
      author={Ekkasit Pinyoanuntapong and Muhammad Usama Saleem and Korrawe Karunratanakul and Pu Wang and Hongfei Xue and Chen Chen and Chuan Guo and Junli Cao and Jian Ren and Sergey Tulyakov},
      year={2024},
      eprint={2410.10780},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.10780}, 
}

@article{motionclr,
  title={MotionCLR: Motion Generation and Training-free Editing via Understanding Attention Mechanisms},
  author={Chen, Ling-Hao and Dai, Wenxun and Ju, Xuan and Lu, Shunlin and Zhang, Lei},
  journal={arxiv:2410.18977},
  year={2024}
}

@misc{zhang2024kmmkeyframemask,
      title={KMM: Key Frame Mask Mamba for Extended Motion Generation}, 
      author={Zeyu Zhang and Hang Gao and Akide Liu and Qi Chen and Feng Chen and Yiran Wang and Danning Li and Hao Tang},
      year={2024},
      eprint={2411.06481},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.06481}, 
}

@misc{zhang2024kinmokinematicawarehumanmotion,
      title={KinMo: Kinematic-aware Human Motion Understanding and Generation}, 
      author={Pengfei Zhang and Pinxin Liu and Hyeongwoo Kim and Pablo Garrido and Bindita Chaudhuri},
      year={2024},
      eprint={2411.15472},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.15472}, 
}
```
</details>

### Survey
- Survey: [Human Motion Generation: A Survey](https://arxiv.org/abs/2307.10894), ArXiv 2023 Nov

### Datasets
   | Motion | Info |                              URL                              |               Others                            | 
   | :-----: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
   |  AIST |  AIST Dance Motion Dataset  | [Link](https://aistdancedb.ongaaccel.jp/) |--|
   |  AIST++  |  AIST++ Dance Motion Dataset | [Link](https://google.github.io/aistplusplus_dataset/) | [dance video database with SMPL annotations](https://google.github.io/aistplusplus_dataset/download.html) |
   |  AMASS  |  optical marker-based motion capture datasets  | [Link](https://amass.is.tue.mpg.de/) |--|

#### Additional Info
<details>
<summary>AMASS</summary>

AMASS is a large database of human motion unifying different optical marker-based motion capture datasets by representing them within a common framework and parameterization. AMASS is readily useful for animation, visualization, and generating training data for deep learning.
  
</details>



## Text to 3D Human

### 🎉 Human Accepted Papers

| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2022 | **AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars**  | SIGGRAPH 2022 (Journal Track)  |          [Link](https://arxiv.org/abs/2205.08535)          | [Link](https://github.com/hongfz16/AvatarCLIP)  | [Link](https://hongfz16.github.io/projects/AvatarCLIP.html)  |
| 2023 | **AvatarCraft: Transforming Text into Neural Human Avatars with Parameterized Shape and Pose Control**  | ICCV 2023 |          [Link](https://arxiv.org/abs/2303.17606)          |  [Link](https://github.com/songrise/avatarcraft)   | [Link](https://avatar-craft.github.io/)  |
| 2023 | **DreamWaltz: Make a Scene with Complex 3D Animatable Avatars**  | NeurIPS 2023  |          [Link](https://arxiv.org/abs/2305.12529)          | [Link](https://github.com/IDEA-Research/DreamWaltz)  | [Link](https://idea-research.github.io/DreamWaltz/)  |
| 2023 | **DreamHuman: Animatable 3D Avatars from Text**  | NeurIPS 2023 (Spotlight)  |          [Link](https://arxiv.org/abs/2306.09329)          |  --  | [Link](https://dream-human.github.io/)  |
| 2023 | **TeCH: Text-guided Reconstruction of Lifelike Clothed Humans**  | 3DV 2024  |          [Link](https://arxiv.org/abs/2308.08545)          | [Link](https://github.com/huangyangyi/TeCH)  | [Link](https://huangyangyi.github.io/TeCH/)  |
| 2023 | **TADA! Text to Animatable Digital Avatars**  | 3DV 2024  |          [Link](https://arxiv.org/abs/2308.10899)          | [Link](https://github.com/TingtingLiao/TADA)  | [Link](https://tada.is.tue.mpg.de/)  |
| 2023 | **AvatarVerse: High-quality & Stable 3D Avatar Creation from Text and Pose**  | AAAI2024  |          [Link](https://arxiv.org/abs/2308.03610)          |  [Link](https://github.com/bytedance/AvatarVerse)  | [Link](https://avatarverse3d.github.io/)  |
| 2023 | **HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting**  | CVPR 2024  |          [Link](https://arxiv.org/abs/2311.17061)          | [Link](https://github.com/alvinliu0/HumanGaussian)  | [Link](https://alvinliu0.github.io/projects/HumanGaussian)  | 
| 2023 | **HumanNorm: Learning Normal Diffusion Model for High-quality and Realistic 3D Human Generation**  | CVPR 2024  |          [Link](https://arxiv.org/abs/2310.01406)          | [Link](https://github.com/xhuangcv/humannorm)  | [Link](https://humannorm.github.io/)  |
| 2024 | **En3D: An Enhanced Generative Model for Sculpting 3D Humans from 2D Synthetic Data**  | CVPR 2024  |          [Link](https://arxiv.org/abs/2401.01173)          |  [Link](https://github.com/menyifang/En3D)  | [Link](https://menyifang.github.io/projects/En3D/index.html)  |
| 2024 | **HeadArtist: Text-conditioned 3D Head Generation with Self Score Distillation**  | SIGGRAPH 2024  |          [Link](https://arxiv.org/abs/2312.07539)          |  [Link](https://github.com/KumapowerLIU/HeadArtist)  | [Link](https://kumapowerliu.github.io/HeadArtist/)  |
| 2024 | **HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting**  | ECCV 2024  |          [Link](https://arxiv.org/abs/2402.06149)          |  [Link](https://github.com/ZhenglinZhou/HeadStudio/)  | [Link](https://zhenglinzhou.github.io/HeadStudio-ProjectPage/)  |
| 2024 | **Instant 3D Human Avatar Generation using Image Diffusion Models**  | ECCV 2024  |          [Link](https://arxiv.org/abs/2406.07516)          | -- | [Link](https://www.nikoskolot.com/avatarpopup/)  |
| 2024 | **Disentangled Clothed Avatar Generation from Text Descriptions**  | ECCV 2024  |          [Link](https://arxiv.org/abs/2312.05295)          | [Link](https://github.com/shanemankiw/SO-SMPL) | [Link](https://shanemankiw.github.io/SO-SMPL/)  |

<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

@article{hong2022avatarclip,
    title={AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars},
    author={Hong, Fangzhou and Zhang, Mingyuan and Pan, Liang and Cai, Zhongang and Yang, Lei and Liu, Ziwei},
    journal={ACM Transactions on Graphics (TOG)},
    volume={41},
    number={4},
    pages={1--19},
    year={2022},
    publisher={ACM New York, NY, USA}
}

@article{jiang2023avatarcraft,
  title={AvatarCraft: Transforming Text into Neural Human Avatars with Parameterized Shape and Pose Control},
  author={Jiang, Ruixiang and Wang, Can and Zhang, Jingbo and Chai, Menglei and He, Mingming and Chen, Dongdong and Liao, Jing},
  journal={arXiv preprint arXiv:2303.17606},
  year={2023}
}

@inproceedings{huang2023dreamwaltz,
  title={{DreamWaltz: Make a Scene with Complex 3D Animatable Avatars}},
  author={Yukun Huang and Jianan Wang and Ailing Zeng and He Cao and Xianbiao Qi and Yukai Shi and Zheng-Jun Zha and Lei Zhang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

@article{kolotouros2023dreamhuman,
  title={DreamHuman: Animatable 3D Avatars from Text},
  author={Kolotouros, Nikos and Alldieck, Thiemo and Zanfir, Andrei and Bazavan, Eduard Gabriel and Fieraru, Mihai and Sminchisescu, Cristian},
  booktitle={NeurIPS},
  year={2023}
}

@inproceedings{huang2024tech,
  title={{TeCH: Text-guided Reconstruction of Lifelike Clothed Humans}},
  author={Huang, Yangyi and Yi, Hongwei and Xiu, Yuliang and Liao, Tingting and Tang, Jiaxiang and Cai, Deng and Thies, Justus},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2024}
}

@article{liao2023tada,
title={TADA! Text to Animatable Digital Avatars},
author={Liao, Tingting and Yi, Hongwei and Xiu, Yuliang and Tang, Jiaxiang and Huang, Yangyi and Thies, Justus and Black, Michael J},
journal={ArXiv},
month={Aug}, 
year={2023} 
}

@article{zhang2023avatarverse,
  title={Avatarverse: High-quality \& stable 3d avatar creation from text and pose},
  author={Zhang, Huichao and Chen, Bowen and Yang, Hao and Qu, Liao and Wang, Xu and Chen, Li and Long, Chao and Zhu, Feida and Du, Kang and Zheng, Min},
  journal={arXiv preprint arXiv:2308.03610},
  year={2023}
}

@article{liu2023humangaussian,
    title={HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting},
    author={Liu, Xian and Zhan, Xiaohang and Tang, Jiaxiang and Shan, Ying and Zeng, Gang and Lin, Dahua and Liu, Xihui and Liu, Ziwei},
    journal={arXiv preprint arXiv:2311.17061},
    year={2023}
}

@misc{huang2023humannorm,
title={Humannorm: Learning normal diffusion model for high-quality and realistic 3d human generation},
author={Huang, Xin and Shao, Ruizhi and Zhang, Qi and Zhang, Hongwen and Feng, Ying and Liu, Yebin and Wang, Qing},
booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
year={2024}
}

@inproceedings{men2024en3d,
  title={En3D: An Enhanced Generative Model for Sculpting 3D Humans from 2D Synthetic Data},
  author={Men, Yifang and Lei, Biwen and Yao, Yuan and Cui, Miaomiao and Lian, Zhouhui and Xie, Xuansong},
  journal={arXiv preprint arXiv:2401.01173},
  website={https://menyifang.github.io/projects/En3D/index.html},
  year={2024}
}

@article{liu2023HeadArtist,
  author = {Hongyu Liu, Xuan Wang, Ziyu Wan, Yujun Shen, Yibing Song, Jing Liao, Qifeng Chen},
  title = {HeadArtist: Text-conditioned 3D Head Generation with Self Score Distillation},
  journal = {arXiv:2312.07539},
  year = {2023},
}

@article{zhou2024headstudio,
  author = {Zhenglin Zhou and Fan Ma and Hehe Fan and Yi Yang},
  title = {HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting},
  journal={arXiv preprint arXiv:2402.06149},
  year={2024}
}

@inproceedings{kolotouros2024avatarpopup,
  author    = {Kolotouros, Nikos and Alldieck, Thiemo and Corona, Enric and Bazavan, Eduard Gabriel and Sminchisescu, Cristian},
  title     = {Instant 3D Human Avatar Generation using Image Diffusion Models},
  booktitle   = {European Conference on Computer Vision (ECCV)},
  year      = {2024},
}

@misc{wang2023disentangled,
      title={Disentangled Clothed Avatar Generation from Text Descriptions}, 
      author={Jionghao Wang and Yuan Liu and Zhiyang Dou and Zhengming Yu and Yongqing Liang and Xin Li and Wenping Wang and Rong Xie and Li Song},
      year={2023},
      eprint={2312.05295},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
</details>

---

### 💡 Human ArXiv Papers

#### 1. Make-A-Character: High Quality Text-to-3D Character Generation within Minutes
Jianqiang Ren, Chao He, Lin Liu, Jiahao Chen, Yutong Wang, Yafei Song, Jianfang Li, Tangli Xue, Siqi Hu, Tao Chen, Kunkun Zheng, Jianjing Xiang, Liefeng Bo

(Institute for Intelligent Computing, Alibaba Group)
<details span>
<summary><b>Abstract</b></summary>
There is a growing demand for customized and expressive 3D characters with the emergence of AI agents and Metaverse, but creating 3D characters using traditional computer graphics tools is a complex and time-consuming task. To address these challenges, we propose a user-friendly framework named Make-A-Character (Mach) to create lifelike 3D avatars from text descriptions. The framework leverages the power of large language and vision models for textual intention understanding and intermediate image generation, followed by a series of human-oriented visual perception and 3D generation modules. Our system offers an intuitive approach for users to craft controllable, realistic, fully-realized 3D characters that meet their expectations within 2 minutes, while also enabling easy integration with existing CG pipeline for dynamic expressiveness. 
</details>

#### 2. MagicMirror: Fast and High-Quality Avatar Generation with a Constrained Search Space
Armand Comas-Massagué, Di Qiu, Menglei Chai, Marcel Bühler, Amit Raj, Ruiqi Gao, Qiangeng Xu, Mark Matthews, Paulo Gotardo, Octavia Camps, Sergio Orts-Escolano, Thabo Beeler

(Google, Northeastern Univeristy, ETH Zurich, Google DeepMind)
<details span>
<summary><b>Abstract</b></summary>
We introduce a novel framework for 3D human avatar generation and personalization, leveraging text prompts to enhance user engagement and customization. Central to our approach are key innovations aimed at overcoming the challenges in photo-realistic avatar synthesis. Firstly, we utilize a conditional Neural Radiance Fields (NeRF) model, trained on a large-scale unannotated multi-view dataset, to create a versatile initial solution space that accelerates and diversifies avatar generation. Secondly, we develop a geometric prior, leveraging the capabilities of Text-to-Image Diffusion Models, to ensure superior view invariance and enable direct optimization of avatar geometry. These foundational ideas are complemented by our optimization pipeline built on Variational Score Distillation (VSD), which mitigates texture loss and over-saturation issues. As supported by our extensive experiments, these strategies collectively enable the creation of custom avatars with unparalleled visual quality and better adherence to input text prompts. 
</details>

#### 3. InstructHumans: Editing Animated 3D Human Textures with Instructions (text to 3d human texture editing)
Jiayin Zhu, Linlin Yang, Angela Yao

(National University of Singapore, Communication University of China)
<details span>
<summary><b>Abstract</b></summary>
We present InstructHumans, a novel framework for instruction-driven 3D human texture editing. Existing text-based editing methods use Score Distillation Sampling (SDS) to distill guidance from generative models. This work shows that naively using such scores is harmful to editing as they destroy consistency with the source avatar. Instead, we propose an alternate SDS for Editing (SDS-E) that selectively incorporates subterms of SDS across diffusion timesteps. We further enhance SDS-E with spatial smoothness regularization and gradient-based viewpoint sampling to achieve high-quality edits with sharp and high-fidelity detailing. InstructHumans significantly outperforms existing 3D editing methods, consistent with the initial avatar while faithful to the textual instructions.
</details>

#### 4. HumanCoser: Layered 3D Human Generation via Semantic-Aware Diffusion Model
Yi Wang, Jian Ma, Ruizhi Shao, Qiao Feng, Yu-kun Lai, Kun Li

(Tianjin University, Changzhou Institute of Technology, Cardiff University)
<details span>
<summary><b>Abstract</b></summary>
This paper aims to generate physically-layered 3D humans from text prompts. Existing methods either generate 3D clothed humans as a whole or support only tight and simple clothing generation, which limits their applications to virtual try-on and part-level editing. To achieve physically-layered 3D human generation with reusable and complex clothing, we propose a novel layer-wise dressed human representation based on a physically-decoupled diffusion model. Specifically, to achieve layer-wise clothing generation, we propose a dual-representation decoupling framework for generating clothing decoupled from the human body, in conjunction with an innovative multi-layer fusion volume rendering method. To match the clothing with different body shapes, we propose an SMPL-driven implicit field deformation network that enables the free transfer and reuse of clothing. Extensive experiments demonstrate that our approach not only achieves state-of-the-art layered 3D human generation with complex clothing but also supports virtual try-on and layered human animation.
</details>

#### 5. DreamHOI: Subject-Driven Generation of 3D Human-Object Interactions with Diffusion Priors
Thomas Hanwen Zhu, Ruining Li, Tomas Jakab

(University of Oxford, Carnegie Mellon University)
<details span>
<summary><b>Abstract</b></summary>
We present DreamHOI, a novel method for zero-shot synthesis of human-object interactions (HOIs), enabling a 3D human model to realistically interact with any given object based on a textual description. This task is complicated by the varying categories and geometries of real-world objects and the scarcity of datasets encompassing diverse HOIs. To circumvent the need for extensive data, we leverage text-to-image diffusion models trained on billions of image-caption pairs. We optimize the articulation of a skinned human mesh using Score Distillation Sampling (SDS) gradients obtained from these models, which predict image-space edits. However, directly backpropagating image-space gradients into complex articulation parameters is ineffective due to the local nature of such gradients. To overcome this, we introduce a dual implicit-explicit representation of a skinned mesh, combining (implicit) neural radiance fields (NeRFs) with (explicit) skeleton-driven mesh articulation. During optimization, we transition between implicit and explicit forms, grounding the NeRF generation while refining the mesh articulation. We validate our approach through extensive experiments, demonstrating its effectiveness in generating realistic HOIs.
</details>

#### 6. AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction
Lingteng Qiu, Shenhao Zhu, Qi Zuo, Xiaodong Gu, Yuan Dong, Junfei Zhang, Chao Xu, Zhe Li, Weihao Yuan, Liefeng Bo, Guanying Chen, Zilong Dong

(Alibaba Group, Sun Yat-sen University, Nanjing University, Huazhong University of Science and Technology)
<details span>
<summary><b>Abstract</b></summary>
Generating animatable human avatars from a single image is essential for various digital human modeling applications. Existing 3D reconstruction methods often struggle to capture fine details in animatable models, while generative approaches for controllable animation, though avoiding explicit 3D modeling, suffer from viewpoint inconsistencies in extreme poses and computational inefficiencies. In this paper, we address these challenges by leveraging the power of generative models to produce detailed multi-view canonical pose images, which help resolve ambiguities in animatable human reconstruction. We then propose a robust method for 3D reconstruction of inconsistent images, enabling real-time rendering during inference. Specifically, we adapt a transformer-based video generation model to generate multi-view canonical pose images and normal maps, pretraining on a large-scale video dataset to improve generalization. To handle view inconsistencies, we recast the reconstruction problem as a 4D task and introduce an efficient 3D modeling approach using 4D Gaussian Splatting. Experiments demonstrate that our method achieves photorealistic, real-time animation of 3D human avatars from in-the-wild images, showcasing its effectiveness and generalization capability.
</details>

---

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **Make-A-Character: High Quality Text-to-3D Character Generation within Minutes**  | 24 Dec 2023  |          [Link](https://arxiv.org/abs/2312.15430)          |  [Link](https://github.com/Human3DAIGC/Make-A-Character)  | [Link](https://human3daigc.github.io/MACH/)  |
| 2024 | **MagicMirror: Fast and High-Quality Avatar Generation with a Constrained Search Space**  | 1 Apr 2024  |          [Link](https://arxiv.org/abs/2404.01296)      | -- | [Link](https://syntec-research.github.io/MagicMirror/)  |
| 2024 | **InstructHumans: Editing Animated 3D Human Textures with Instructions**  | 5 Apr 2024  |          [Link](https://arxiv.org/abs/2404.04037)          | [Link](https://github.com/viridityzhu/InstructHumans)  | [Link](https://jyzhu.top/instruct-humans/)  |
| 2024 | **HumanCoser: Layered 3D Human Generation via Semantic-Aware Diffusion Model**  | 21 Aug 2024  |          [Link](https://arxiv.org/abs/2408.11357)          | -- | -- |
| 2024 | **DreamHOI: Subject-Driven Generation of 3D Human-Object Interactions with Diffusion Priors**  | 12 Sep 2024  |          [Link](https://arxiv.org/abs/2409.08278)          | [Link](https://github.com/hanwenzhu/dreamhoi)| [Link](https://dreamhoi.github.io/) |
| 2024 | **AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction**  | 3 Dec 2024  |         --        | [Link](https://github.com/aigc3d/AniGS)| [Link](https://lingtengqiu.github.io/2024/AniGS/) |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@article{ren2023makeacharacter,
      title={Make-A-Character: High Quality Text-to-3D Character Generation within Minutes},
      author={Jianqiang Ren and Chao He and Lin Liu and Jiahao Chen and Yutong Wang and Yafei Song and Jianfang Li and Tangli Xue and Siqi Hu and Tao Chen and Kunkun Zheng and Jianjing Xiang and Liefeng Bo},
      year={2023},
      journal = {arXiv preprint arXiv:2312.15430}
}

@article{comas2024magicmirror,
  title={MagicMirror: Fast and High-Quality Avatar Generation with a Constrained Search Space},
  author={Comas-Massagu{\'e}, Armand and Qiu, Di and Chai, Menglei and B{\"u}hler, Marcel and Raj, Amit and Gao, Ruiqi and Xu, Qiangeng and Matthews, Mark and Gotardo, Paulo and Camps, Octavia and others},
  journal={arXiv preprint arXiv:2404.01296},
  year={2024}
}

@article{zhu2024InstructHumans,
         author={Zhu, Jiayin and Yang, Linlin and Yao, Angela},
         title={InstructHumans: Editing Animated 3D Human Textures with Instructions},
         journal={arXiv preprint arXiv:2404.04037},
         year={2024}
}

@misc{wang2024humancoserlayered3dhuman,
      title={HumanCoser: Layered 3D Human Generation via Semantic-Aware Diffusion Model}, 
      author={Yi Wang and Jian Ma and Ruizhi Shao and Qiao Feng and Yu-kun Lai and Kun Li},
      year={2024},
      eprint={2408.11357},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.11357}, 
}

@article{zhu2024dreamhoi,
  title   = {{DreamHOI}: Subject-Driven Generation of 3D Human-Object Interactions with Diffusion Priors},
  author  = {Thomas Hanwen Zhu and Ruining Li and Tomas Jakab},
  journal = {arXiv preprint arXiv:2409.08278},
  year    = {2024}
}

@article{qiu2024AniGS,
  title={AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction},
  author={Qiu, Lingteng},
  year={2024}
}
```
</details>

### Additional Info
<details close>
<summary>Survey and Awesome Repos</summary>
 
#### Survey
- [PROGRESS AND PROSPECTS IN 3D GENERATIVE AI: A TECHNICAL OVERVIEW INCLUDING 3D HUMAN](https://arxiv.org/pdf/2401.02620.pdf), ArXiv 2024
  
#### Awesome Repos
- Resource1: [Awesome Digital Human](https://github.com/weihaox/awesome-digital-human)
</details>

<details close>
<summary>Pretrained Models</summary>

   | Pretrained Models (human body) | Info |                              URL                              |
   | :-----: | :-----: | :----------------------------------------------------------: |
   |  SMPL  |  smpl model (smpl weights) | [Link](https://smpl.is.tue.mpg.de/) |
   |  SMPL-X  |  smpl model (smpl weights)  | [Link](https://smpl-x.is.tue.mpg.de/) |
   |  human_body_prior  |  vposer model (smpl weights)  | [Link](https://github.com/nghorbani/human_body_prior) |
<details>
<summary>SMPL</summary>

SMPL is an easy-to-use, realistic, model of the of the human body that is useful for animation and computer vision.

- version 1.0.0 for Python 2.7 (female/male, 10 shape PCs)
- version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)
- UV map in OBJ format
  
</details>

<details>
<summary>SMPL-X</summary>

SMPL-X, that extends SMPL with fully articulated hands and facial expressions (55 joints, 10475 vertices)

</details>
</details>

## Text to Video

### 🎉 Video Accepted Papers

| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2024 | **MicroCinema:A Divide-and-Conquer Approach for Text-to-Video Generation**  | CVPR 2024 (Highlight) |          [Link](https://arxiv.org/abs/2311.18829)          | -- | [Link](https://wangyanhui666.github.io/MicroCinema.github.io/)  |
| 2024 | **LivePhoto: Real Image Animation with Text-guided Motion Control**  | ECCV 2024 |          [Link](https://arxiv.org/abs/2312.02928)          | [Link](https://github.com/XavierCHEN34/LivePhoto)  | [Link](https://xavierchen34.github.io/LivePhoto-Page/)  |
| 2024 | **xGen-VideoSyn-1: High-fidelity Text-to-Video Synthesis with Compressed Representations**  |  ECCV 2024 AI4VA |          [Link](https://arxiv.org/abs/2408.12590)          | [Link](https://github.com/SalesforceAIResearch/xgen-videosyn)  | -- |
| 2024 | **MotionBooth: Motion-Aware Customized Text-to-Video Generation**  | NeurIPS 2024 Spotlight  | [Link](https://arxiv.org/abs/2406.17758) |         [Link](https://github.com/jianzongwu/MotionBooth)         | [Link](https://jianzongwu.github.io/projects/motionbooth/) |
| 2024 | **Vivid-ZOO: Multi-View Video Generation with Diffusion Model**  | NeurIPS 2024  | [Link](https://arxiv.org/abs/2406.08659) |         [Link](https://github.com/hi-zhengcheng/vividzoo)         | [Link](https://hi-zhengcheng.github.io/vividzoo/) |
| 2024 | **Enhancing Motion in Text-to-Video Generation with Decomposed Encoding and Conditioning**  | NeurIPS 2024  | [Link](https://arxiv.org/abs/2410.24219) |         [Link](https://github.com/PR-Ryan/DEMO)    | [Link](https://pr-ryan.github.io/DEMO-project/) |
| 2024 | **VideoDirectorGPT: Consistent Multi-scene Video Generation via LLM-Guided Planning**  | COLM 2024  | [Link](https://arxiv.org/abs/2309.15091) |         [Link](https://github.com/HL-hanlin/VideoDirectorGPT)         | [Link](https://videodirectorgpt.github.io/) |

<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

@inproceedings{wang2024microcinema,
  title={Microcinema: A divide-and-conquer approach for text-to-video generation},
  author={Wang, Yanhui and Bao, Jianmin and Weng, Wenming and Feng, Ruoyu and Yin, Dacheng and Yang, Tao and Zhang, Jingxu and Dai, Qi and Zhao, Zhiyuan and Wang, Chunyu and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8414--8424},
  year={2024}
}

@article{chen2023livephoto,
    title={LivePhoto: Real Image Animation with Text-guided Motion Control},
    author={Chen, Xi and Liu, Zhiheng and Chen, Mengting and Feng, Yutong and Liu, Yu and Shen, Yujun and Zhao, Hengshuang},
    journal={arXiv preprint arXiv:2312.02928},
    year={2023}
}

@misc{qin2024xgenvideosyn1highfidelitytexttovideosynthesis,
      title={xGen-VideoSyn-1: High-fidelity Text-to-Video Synthesis with Compressed Representations}, 
      author={Can Qin and Congying Xia and Krithika Ramakrishnan and Michael Ryoo and Lifu Tu and Yihao Feng and Manli Shu and Honglu Zhou and Anas Awadalla and Jun Wang and Senthil Purushwalkam and Le Xue and Yingbo Zhou and Huan Wang and Silvio Savarese and Juan Carlos Niebles and Zeyuan Chen and Ran Xu and Caiming Xiong},
      year={2024},
      eprint={2408.12590},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.12590}, 
}

@article{wu2024motionbooth,
  title={MotionBooth: Motion-Aware Customized Text-to-Video Generation},
  author={Jianzong Wu and Xiangtai Li and Yanhong Zeng and Jiangning Zhang and Qianyu Zhou and Yining Li and Yunhai Tong and Kai Chen},
  journal={arXiv pre-print arXiv:2406.17758},
  year={2024},
}

@misc{li2024vividzoo,
  title={Vivid-ZOO: Multi-View Video Generation with Diffusion Model}, 
  author={Bing Li and Cheng Zheng and Wenxuan Zhu and Jinjie Mai and Biao Zhang and Peter Wonka and Bernard Ghanem},
  year={2024},
  eprint={2406.08659},
  archivePrefix={arXiv},
}

@misc{ruan2024enhancingmotiontexttovideogeneration,
      title={Enhancing Motion in Text-to-Video Generation with Decomposed Encoding and Conditioning}, 
      author={Penghui Ruan and Pichao Wang and Divya Saxena and Jiannong Cao and Yuhui Shi},
      year={2024},
      eprint={2410.24219},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.24219}, 
}

@article{Lin2023VideoDirectorGPT,
        author = {Han Lin and Abhay Zala and Jaemin Cho and Mohit Bansal},
        title = {VideoDirectorGPT: Consistent Multi-Scene Video Generation via LLM-Guided Planning},
        year = {2023},
}
```
</details>

---

### 💡 Video ArXiv Papers

#### 1. StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text  
Roberto Henschel, Levon Khachatryan, Daniil Hayrapetyan, Hayk Poghosyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, Humphrey Shi

(Picsart AI Resarch (PAIR), UT Austin, SHI Labs @ Georgia Tech Oregon & UIUC)
<details span>
<summary><b>Abstract</b></summary>
Text-to-video diffusion models enable the generation of high-quality videos that follow text instructions, making it easy to create diverse and individual content. However, existing approaches mostly focus on high-quality short video generation (typically 16 or 24 frames), ending up with hard-cuts when naively extended to the case of long video synthesis. To overcome these limitations, we introduce StreamingT2V, an autoregressive approach for long video generation of 80, 240, 600, 1200 or more frames with smooth transitions. The key components are:(i) a short-term memory block called conditional attention module (CAM), which conditions the current generation on the features extracted from the previous chunk via an attentional mechanism, leading to consistent chunk transitions, (ii) a long-term memory block called appearance preservation module, which extracts high-level scene and object features from the first video chunk to prevent the model from forgetting the initial scene, and (iii) a randomized blending approach that enables to apply a video enhancer autoregressively for infinitely long videos without inconsistencies between chunks. Experiments show that StreamingT2V generates high motion amount. In contrast, all competing image-to-video methods are prone to video stagnation when applied naively in an autoregressive manner. Thus, we propose with StreamingT2V a high-quality seamless text-to-long video generator that outperforms competitors with consistency and motion.
</details>

#### 2. Text-Animator: Controllable Visual Text Video Generation
Lin Liu, Quande Liu, Shengju Qian, Yuan Zhou, Wengang Zhou, Houqiang Li, Lingxi Xie, Qi Tian

(University of Science and Technology of China, Tencent, Nanyang Technical University, Huawei Tech)
<details span>
<summary><b>Abstract</b></summary>
Video generation is a challenging yet pivotal task in various industries, such as gaming, e-commerce, and advertising. One significant unresolved aspect within T2V is the effective visualization of text within generated videos. Despite the progress achieved in Text-to-Video~(T2V) generation, current methods still cannot effectively visualize texts in videos directly, as they mainly focus on summarizing semantic scene information, understanding, and depicting actions. While recent advances in image-level visual text generation show promise, transitioning these techniques into the video domain faces problems, notably in preserving textual fidelity and motion coherence. In this paper, we propose an innovative approach termed Text-Animator for visual text video generation. Text-Animator contains a text embedding injection module to precisely depict the structures of visual text in generated videos. Besides, we develop a camera control module and a text refinement module to improve the stability of generated visual text by controlling the camera movement as well as the motion of visualized text. Quantitative and qualitative experimental results demonstrate the superiority of our approach to the accuracy of generated visual text over state-of-the-art video generation methods. 
</details>

#### 3. Still-Moving: Customized Video Generation without Customized Video Data
Hila Chefer, Shiran Zada, Roni Paiss, Ariel Ephrat, Omer Tov, Michael Rubinstein, Lior Wolf, Tali Dekel, Tomer Michaeli, Inbar Mosseri

(Google DeepMind, Tel Aviv University, Weizmann Institute of Science, Technion)
<details span>
<summary><b>Abstract</b></summary>
Customizing text-to-image (T2I) models has seen tremendous progress recently, particularly in areas such as personalization, stylization, and conditional generation. However, expanding this progress to video generation is still in its infancy, primarily due to the lack of customized video data. In this work, we introduce Still-Moving, a novel generic framework for customizing a text-to-video (T2V) model, without requiring any customized video data. The framework applies to the prominent T2V design where the video model is built over a text-to-image (T2I) model (e.g., via inflation). We assume access to a customized version of the T2I model, trained only on still image data (e.g., using DreamBooth or StyleDrop). Naively plugging in the weights of the customized T2I model into the T2V model often leads to significant artifacts or insufficient adherence to the customization data. To overcome this issue, we train lightweight Spatial Adapters that adjust the features produced by the injected T2I layers. Importantly, our adapters are trained on "frozen videos" (i.e., repeated images), constructed from image samples generated by the customized T2I model. This training is facilitated by a novel Motion Adapter module, which allows us to train on such static videos while preserving the motion prior of the video model. At test time, we remove the Motion Adapter modules and leave in only the trained Spatial Adapters. This restores the motion prior of the T2V model while adhering to the spatial prior of the customized T2I model. We demonstrate the effectiveness of our approach on diverse tasks including personalized, stylized, and conditional generation. In all evaluated scenarios, our method seamlessly integrates the spatial prior of the customized T2I model with a motion prior supplied by the T2V model.
</details>

#### 4. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer
Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Xiaohan Zhang, Xiaotao Gu, Guanyu Feng, Da Yin, Wenyi Hong, Weihan Wang, Yean Cheng, Yuxuan Zhang, Ting Liu, Bin Xu, Yuxiao Dong, Jie Tang

(Zhipu AI, Tsinghua University)
<details span>
<summary><b>Abstract</b></summary>
We introduce CogVideoX, a large-scale diffusion transformer model designed for generating videos based on text prompts. To efficently model video data, we propose to levearge a 3D Variational Autoencoder (VAE) to compresses videos along both spatial and temporal dimensions. To improve the text-video alignment,we propose an expert transformer with the expert adaptive LayerNorm to facilitate the deep fusion between the two modalities. By employing a progressive training technique, CogVideoX is adept at producing coherent, long-duration videos characterized by significant motion. In addition, we develop an effectively text-video data processing pipeline that includes various data preprocessing strategies and a video captioning method. It significantly helps enhance the performance of CogVideoX,
improving both generation quality and semantic alignment. Results show that CogVideoX demonstrates state-of-the-art performance across both multiple machine metrics and human evaluations.
</details>

#### 5. CustomCrafter: Customized Video Generation with Preserving Motion and Concept Composition Abilities
Tao Wu, Yong Zhang, Xintao Wang, Xianpan Zhou, Guangcong Zheng, Zhongang Qi, Ying Shan, Xi Li

(Zhejiang University, Tencent AI Lab, ARC Lab Tencent PCG)
<details span>
<summary><b>Abstract</b></summary>
Customized video generation aims to generate high-quality videos guided by text prompts and subject's reference images. However, since it is only trained on static images, the fine-tuning process of subject learning disrupts abilities of video diffusion models (VDMs) to combine concepts and generate motions. To restore these abilities, some methods use additional video similar to the prompt to fine-tune or guide the model. This requires frequent changes of guiding videos and even re-tuning of the model when generating different motions, which is very inconvenient for users. In this paper, we propose CustomCrafter, a novel framework that preserves the model's motion generation and conceptual combination abilities without additional video and fine-tuning to recovery. For preserving conceptual combination ability, we design a plug-and-play module to update few parameters in VDMs, enhancing the model's ability to capture the appearance details and the ability of concept combinations for new subjects. For motion generation, we observed that VDMs tend to restore the motion of video in the early stage of denoising, while focusing on the recovery of subject details in the later stage. Therefore, we propose Dynamic Weighted Video Sampling Strategy. Using the pluggability of our subject learning modules, we reduce the impact of this module on motion generation in the early stage of denoising, preserving the ability to generate motion of VDMs. In the later stage of denoising, we restore this module to repair the appearance details of the specified subject, thereby ensuring the fidelity of the subject's appearance. Experimental results show that our method has a significant improvement compared to previous methods.
</details>

#### 6. Tora: Trajectory-oriented Diffusion Transformer for Video Generation
Zhenghao Zhang, Junchao Liao, Menghao Li, Zuozhuo Dai, Bingxue Qiu, Siyu Zhu, Long Qin, Weizhi Wang

(Alibaba Group, Fudan University)
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in Diffusion Transformer (DiT) have demonstrated remarkable proficiency in producing high-quality video content. Nonetheless, the potential of transformer-based diffusion models for effectively generating videos with controllable motion remains an area of limited exploration. This paper introduces Tora, the first trajectory-oriented DiT framework that concurrently integrates textual, visual, and trajectory conditions, thereby enabling scalable video generation with effective motion guidance. Specifically, Tora consists of a Trajectory Extractor(TE), a Spatial-Temporal DiT, and a Motion-guidance Fuser(MGF). The TE encodes arbitrary trajectories into hierarchical spacetime motion patches with a 3D video compression network. The MGF integrates the motion patches into the DiT blocks to generate consistent videos that accurately follow designated trajectories. Our design aligns seamlessly with DiT's scalability, allowing precise control of video content's dynamics with diverse durations, aspect ratios, and resolutions. Extensive experiments demonstrate Tora's excellence in achieving high motion fidelity, while also meticulously simulating the intricate movement of the physical world.
</details>

#### 7. BroadWay: Boost Your Text-to-Video Generation Model in a Training-free Way
Jiazi Bu, Pengyang Ling, Pan Zhang, Tong Wu, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Dahua Lin, Jiaqi Wang

(Shanghai Jiao Tong University, University of Science and Technology of China, The Chinese University of Hong Kong, Shanghai Artificial Intelligence Laboratory)
<details span>
<summary><b>Abstract</b></summary>
The text-to-video (T2V) generation models, offering convenient visual creation, have recently garnered increasing attention. Despite their substantial potential, the generated videos may present artifacts, including structural implausibility, temporal inconsistency, and a lack of motion, often resulting in near-static video. In this work, we have identified a correlation between the disparity of temporal attention maps across different blocks and the occurrence of temporal inconsistencies. Additionally, we have observed that the energy contained within the temporal attention maps is directly related to the magnitude of motion amplitude in the generated videos. Based on these observations, we present BroadWay, a training-free method to improve the quality of text-to-video generation without introducing additional parameters, augmenting memory or sampling time. Specifically, BroadWay is composed of two principal components: 1) Temporal Self-Guidance improves the structural plausibility and temporal consistency of generated videos by reducing the disparity between the temporal attention maps across various decoder blocks. 2) Fourier-based Motion Enhancement enhances the magnitude and richness of motion by amplifying the energy of the map. Extensive experiments demonstrate that BroadWay significantly improves the quality of text-to-video generation with negligible additional cost.
</details>

#### 8. Pyramidal Flow Matching for Efficient Video Generative Modeling
Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, Zhouchen Lin

(Peking University, Kuaishou Technology, Beijing University of Posts and Telecommunications)
<details span>
<summary><b>Abstract</b></summary>
Video generation requires modeling a vast spatiotemporal space, which demands significant computational resources and data usage. To reduce the complexity, the prevailing approaches employ a cascaded architecture to avoid direct training with full resolution. Despite reducing computational demands, the separate optimization of each sub-stage hinders knowledge sharing and sacrifices flexibility. This work introduces a unified pyramidal flow matching algorithm. It reinterprets the original denoising trajectory as a series of pyramid stages, where only the final stage operates at the full resolution, thereby enabling more efficient video generative modeling. Through our sophisticated design, the flows of different pyramid stages can be interlinked to maintain continuity. Moreover, we craft autoregressive video generation with a temporal pyramid to compress the full-resolution history. The entire framework can be optimized in an end-to-end manner and with a single unified Diffusion Transformer (DiT). Extensive experiments demonstrate that our method supports generating high-quality 5-second (up to 10-second) videos at 768p resolution and 24 FPS within 20.7k A100 GPU training hours. 
</details>

#### 9. GameGen-X: Interactive Open-world Game Video Generation
Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, Hao Chen

(Hong Kong Univerity of Science and Technology, Univerity of Science and Technology of China, The Chinese Univerity of Hong Kong)
<details span>
<summary><b>Abstract</b></summary>
We introduce GameGen-X, the first diffusion transformer model specifically designed for both generating and interactively controlling open-world game videos. This model facilitates high-quality, open-domain generation by simulating an extensive array of game engine features, such as innovative characters, dynamic environments, complex actions, and diverse events. Additionally, it provides interactive controllability, predicting and altering future content based on the current clip, thus allowing for gameplay simulation. To realize this vision, we first collected and built an Open-World Video Game Dataset from scratch. It is the first and largest dataset for open-world game video generation and control, which comprises over a million diverse gameplay video clips sampling from over 150 games with informative captions from GPT-4o. GameGen-X undergoes a two-stage training process, consisting of foundation model pre-training and instruction tuning. Firstly, the model was pre-trained via text-to-video generation and video continuation, endowing it with the capability for long-sequence, high-quality open-domain game video generation. Further, to achieve interactive controllability, we designed InstructNet to incorporate game-related multi-modal control signal experts. This allows the model to adjust latent representations based on user inputs, unifying character interaction and scene content control for the first time in video generation. During instruction tuning, only the InstructNet is updated while the pre-trained foundation model is frozen, enabling the integration of interactive controllability without loss of diversity and quality of generated video content.
</details>

#### 10. Motion Control for Enhanced Complex Action Video Generation
Qiang Zhou, Shaofeng Zhang, Nianzu Yang, Ye Qian, Hao Li

(INF Tech., Shanghai Jiao Tong University, Fudan University)
<details span>
<summary><b>Abstract</b></summary>
Existing text-to-video (T2V) models often struggle with generating videos with sufficiently pronounced or complex actions. A key limitation lies in the text prompt's inability to precisely convey intricate motion details. To address this, we propose a novel framework, MVideo, designed to produce long-duration videos with precise, fluid actions. MVideo overcomes the limitations of text prompts by incorporating mask sequences as an additional motion condition input, providing a clearer, more accurate representation of intended actions. Leveraging foundational vision models such as GroundingDINO and SAM2, MVideo automatically generates mask sequences, enhancing both efficiency and robustness. Our results demonstrate that, after training, MVideo effectively aligns text prompts with motion conditions to produce videos that simultaneously meet both criteria. This dual control mechanism allows for more dynamic video generation by enabling alterations to either the text prompt or motion condition independently, or both in tandem. Furthermore, MVideo supports motion condition editing and composition, facilitating the generation of videos with more complex actions. MVideo thus advances T2V motion generation, setting a strong benchmark for improved action depiction in current video diffusion models.
</details>

#### 11. AnimateAnything: Consistent and Controllable Animation for video generation
Guojun Lei, Chi Wang, Hong Li, Rong Zhang, Yikai Wang, Weiwei Xu

(State Key Lab of CAD&CG Zhejiang University, Tsinghua University, Beihang University, Zhejiang Gongshang University, ShengShu)
<details span>
<summary><b>Abstract</b></summary>
We present a unified controllable video generation approach AnimateAnything that facilitates precise and consistent video manipulation across various conditions, including camera trajectories, text prompts, and user motion annotations. Specifically, we carefully design a multi-scale control feature fusion network to construct a common motion representation for different conditions. It explicitly converts all control information into frame-by-frame optical flows. Then we incorporate the optical flows as motion priors to guide final video generation. In addition, to reduce the flickering issues caused by large-scale motion, we propose a frequency-based stabilization module. It can enhance temporal coherence by ensuring the video's frequency domain consistency. Experiments demonstrate that our method outperforms the state-of-the-art approaches. 
</details>

#### 12. FlipSketch: Flipping Static Drawings to Text-Guided Sketch Animations (Text-to-Video Finetuning)
Hmrishav Bandyopadhyay, Yi-Zhe Song

(SketchX CVSSP University of Surrey, United Kingdom)
<details span>
<summary><b>Abstract</b></summary>
Sketch animations offer a powerful medium for visual storytelling, from simple flip-book doodles to professional studio productions. While traditional animation requires teams of skilled artists to draw key frames and in-between frames, existing automation attempts still demand significant artistic effort through precise motion paths or keyframe specification. We present FlipSketch, a system that brings back the magic of flip-book animation -- just draw your idea and describe how you want it to move! Our approach harnesses motion priors from text-to-video diffusion models, adapting them to generate sketch animations through three key innovations: (i) fine-tuning for sketch-style frame generation, (ii) a reference frame mechanism that preserves visual integrity of input sketch through noise refinement, and (iii) a dual-attention composition that enables fluid motion without losing visual consistency. Unlike constrained vector animations, our raster frames support dynamic sketch transformations, capturing the expressive freedom of traditional animation. The result is an intuitive system that makes sketch animation as simple as doodling and describing, while maintaining the artistic essence of hand-drawn animation.
</details>

#### 13. DreamRunner: Fine-Grained Storytelling Video Generation with Retrieval-Augmented Motion Adaptation
Zun Wang, Jialu Li, Han Lin, Jaehong Yoon, Mohit Bansal

(University of North Carolina, Chapel Hill)
<details span>
<summary><b>Abstract</b></summary>
Storytelling video generation (SVG) has recently emerged as a task to create long, multi-motion, multi-scene videos that consistently represent the story described in the input text script. SVG holds great potential for diverse content creation in media and entertainment; however, it also presents significant challenges: (1) objects must exhibit a range of fine-grained, complex motions, (2) multiple objects need to appear consistently across scenes, and (3) subjects may require multiple motions with seamless transitions within a single scene. To address these challenges, we propose DreamRunner, a novel story-to-video generation method: First, we structure the input script using a large language model (LLM) to facilitate both coarse-grained scene planning as well as fine-grained object-level layout and motion planning. Next, DreamRunner presents retrieval-augmented test-time adaptation to capture target motion priors for objects in each scene, supporting diverse motion customization based on retrieved videos, thus facilitating the generation of new videos with complex, scripted motions. Lastly, we propose a novel spatial-temporal region-based 3D attention and prior injection module SR3AI for fine-grained object-motion binding and frame-by-frame semantic control. We compare DreamRunner with various SVG baselines, demonstrating state-of-the-art performance in character consistency, text alignment, and smooth transitions. Additionally, DreamRunner exhibits strong fine-grained condition-following ability in compositional text-to-video generation, significantly outperforming baselines on T2V-ComBench. Finally, we validate DreamRunner's robust ability to generate multi-object interactions with qualitative examples.
</details>

#### 14. Motion Prompting: Controlling Video Generation with Motion Trajectories
Daniel Geng, Charles Herrmann, Junhwa Hur, Forrester Cole, Serena Zhang, Tobias Pfaff, Tatiana Lopez-Guevara, Carl Doersch, Yusuf Aytar, Michael Rubinstein, Chen Sun, Oliver Wang, Andrew Owens, Deqing Sun

(Google DeepMind, University of Michigan, Brown University)
<details span>
<summary><b>Abstract</b></summary>
Motion control is crucial for generating expressive and compelling video content; however, most existing video generation models rely mainly on text prompts for control, which struggle to capture the nuances of dynamic actions and temporal compositions. To this end, we train a video generation model conditioned on spatio-temporally sparse or dense motion trajectories. In contrast to prior motion conditioning work, this flexible representation can encode any number of trajectories, object-specific or global scene motion, and temporally sparse motion; due to its flexibility we refer to this conditioning as motion prompts. While users may directly specify sparse trajectories, we also show how to translate high-level user requests into detailed, semi-dense motion prompts, a process we term motion prompt expansion. We demonstrate the versatility of our approach through various applications, including camera and object motion control, "interacting" with an image, motion transfer, and image editing. Our results showcase emergent behaviors, such as realistic physics, suggesting the potential of motion prompts for probing video models and interacting with future generative world models. Finally, we evaluate quantitatively, conduct a human study, and demonstrate strong performance. 
</details>

---

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2024 | **StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text**  | 21 Mar 2024 |          [Link](https://arxiv.org/abs/2403.14773)          | [Link](https://github.com/Picsart-AI-Research/StreamingT2V) | [Link](https://streamingt2v.github.io/)  |
| 2024 | **Text-Animator: Controllable Visual Text Video Generation**  | 25 Jun 2024 |          [Link](https://arxiv.org/abs/2406.17777)          | [Link](https://github.com/laulampaul/text-animator) | [Link](https://laulampaul.github.io/text-animator.html)  |
| 2024 | **Still-Moving: Customized Video Generation without Customized Video Data**  | 11 Jul 2024  | [Link](https://arxiv.org/abs/2407.08674) |          --         | [Link](https://still-moving.github.io/) |
| 2024 | **CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer**  | 12 Aug 2024  | [Link](https://arxiv.org/abs/2408.06072) |          [Link](https://github.com/THUDM/CogVideo)          | [Hugging Face](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox) |
| 2024 | **CustomCrafter: Customized Video Generation with Preserving Motion and Concept Composition Abilities**  | 23 Aug 2024  | [Link](https://arxiv.org/abs/2408.13239) |          [Link](https://github.com/WuTao-CS/CustomCrafter)        | [Link](https://customcrafter.github.io/) |
| 2024 | **Tora: Trajectory-oriented Diffusion Transformer for Video Generation**  | 27 Aug 2024  | [Link](https://arxiv.org/abs/2407.21705)  |      [Link](https://github.com/alibaba/Tora)      | [Link](https://ali-videoai.github.io/tora_video/) |
| 2024 | **BroadWay: Boost Your Text-to-Video Generation Model in a Training-free Way**  | 8 Oct 2024  | [Link](https://arxiv.org/abs/2410.06241)  |      --      | -- |
| 2024 | **Pyramidal Flow Matching for Efficient Video Generative Modeling**  | 8 Oct 2024  | [Link](https://arxiv.org/abs/2410.05954)  |      [Link](https://github.com/jy0205/Pyramid-Flow)      | [Link](https://pyramid-flow.github.io/) |
| 2024 | **GameGen-X: Interactive Open-world Game Video Generation**  |  1 Nov 2024  | [Link](https://arxiv.org/abs/2411.00769)  |          [Link](https://github.com/GameGen-X/GameGen-X)        | [Link](https://gamegen-x.github.io/) |
| 2024 | **MVideo: Motion Control for Enhanced Complex Action Video Generation**  |  13 Nov 2024  | [Link](https://arxiv.org/abs/2411.08328)  |     --      | [Link](https://mvideo-v1.github.io/) |
| 2024 | **AnimateAnything: Consistent and Controllable Animation for video generation**  |  16 Nov 2024  | [Link](https://arxiv.org/abs/2411.10836)  |     [Link](https://github.com/yu-shaonian/AnimateAnything)      | [Link](https://yu-shaonian.github.io/Animate_Anything/) |
| 2024 | **FlipSketch: Flipping Static Drawings to Text-Guided Sketch Animations**  |  16 Nov 2024  | [Link](https://arxiv.org/abs/2411.10818)  |    --      | [Link](https://github.com/hmrishavbandy/FlipSketch) |
| 2024 | **DreamRunner: Fine-Grained Storytelling Video Generation with Retrieval-Augmented Motion Adaptation**  |  25 Nov 2024  | [Link](https://arxiv.org/abs/2411.16657)  |    [Link](https://github.com/wz0919/DreamRunner)      | [Link](https://dreamrunner-story2video.github.io/) |
| 2024 | **Motion Prompting: Controlling Video Generation with Motion Trajectories**  |  3 Dec 2024  | [Link](https://arxiv.org/abs/2412.02700)  |    --     | [Link](https://motion-prompting.github.io/) |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@article{henschel2024streamingt2v,
  title={StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text},
  author={Henschel, Roberto and Khachatryan, Levon and Hayrapetyan, Daniil and Poghosyan, Hayk and Tadevosyan, Vahram and Wang, Zhangyang and Navasardyan, Shant and Shi, Humphrey},
  journal={arXiv preprint arXiv:2403.14773},
  year={2024}
}

@misc{liu2024textanimatorcontrollablevisualtext,
      title={Text-Animator: Controllable Visual Text Video Generation}, 
      author={Lin Liu and Quande Liu and Shengju Qian and Yuan Zhou and Wengang Zhou and Houqiang Li and Lingxi Xie and Qi Tian},
      year={2024},
      eprint={2406.17777},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.17777}, 
}

@misc{chefer2024stillmovingcustomizedvideogeneration,
      title={Still-Moving: Customized Video Generation without Customized Video Data}, 
      author={Hila Chefer and Shiran Zada and Roni Paiss and Ariel Ephrat and Omer Tov and Michael Rubinstein and Lior Wolf and Tali Dekel and Tomer Michaeli and Inbar Mosseri},
      year={2024},
      eprint={2407.08674},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.08674}, 
}

@misc{yang2024cogvideoxtexttovideodiffusionmodels,
      title={CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer}, 
      author={Zhuoyi Yang and Jiayan Teng and Wendi Zheng and Ming Ding and Shiyu Huang and Jiazheng Xu and Yuanming Yang and Wenyi Hong and Xiaohan Zhang and Guanyu Feng and Da Yin and Xiaotao Gu and Yuxuan Zhang and Weihan Wang and Yean Cheng and Ting Liu and Bin Xu and Yuxiao Dong and Jie Tang},
      year={2024},
      eprint={2408.06072},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.06072}, 
}

@misc{wu2024customcraftercustomizedvideogeneration,
      title={CustomCrafter: Customized Video Generation with Preserving Motion and Concept Composition Abilities}, 
      author={Tao Wu and Yong Zhang and Xintao Wang and Xianpan Zhou and Guangcong Zheng and Zhongang Qi and Ying Shan and Xi Li},
      year={2024},
      eprint={2408.13239},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.13239}, 
}

@misc{tora,
        title={Tora: Trajectory-oriented Diffusion Transformer for Video Generation}, 
        author={Zhenghao Zhang and Junchao Liao and Menghao Li and Long Qin and Weizhi Wang},   
        year={2024},
        eprint={2407.21705},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2407.21705}, 
}

@misc{bu2024broadwayboosttexttovideogeneration,
      title={BroadWay: Boost Your Text-to-Video Generation Model in a Training-free Way}, 
      author={Jiazi Bu and Pengyang Ling and Pan Zhang and Tong Wu and Xiaoyi Dong and Yuhang Zang and Yuhang Cao and Dahua Lin and Jiaqi Wang},
      year={2024},
      eprint={2410.06241},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.06241}, 
}

@article{jin2024pyramidal,
  title={Pyramidal Flow Matching for Efficient Video Generative Modeling},
  author={Jin, Yang and Sun, Zhicheng and Li, Ningyuan and Xu, Kun and Xu, Kun and Jiang, Hao and Zhuang, Nan and Huang, Quzhe and Song, Yang and Mu, Yadong and Lin, Zhouchen},
  jounal={arXiv preprint arXiv:2410.05954},
  year={2024}
}

@misc{che2024gamegenxinteractiveopenworldgame,
      title={GameGen-X: Interactive Open-world Game Video Generation}, 
      author={Haoxuan Che and Xuanhua He and Quande Liu and Cheng Jin and Hao Chen},
      year={2024},
      eprint={2411.00769},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.00769}, 
}

@misc{zhou2024motioncontrolenhancedcomplex,
      title={Motion Control for Enhanced Complex Action Video Generation}, 
      author={Qiang Zhou and Shaofeng Zhang and Nianzu Yang and Ye Qian and Hao Li},
      year={2024},
      eprint={2411.08328},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.08328}, 
}

@misc{lei2024animateanythingconsistentcontrollableanimation,
      title={AnimateAnything: Consistent and Controllable Animation for Video Generation}, 
      author={Guojun Lei and Chi Wang and Hong Li and Rong Zhang and Yikai Wang and Weiwei Xu},
      year={2024},
      eprint={2411.10836},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.10836}, 
}

@misc{bandyopadhyay2024flipsketch,
  title={FlipSketch: Flipping assets Drawings to Text-Guided Sketch Animations}, 
  author={Hmrishav Bandyopadhyay and Yi-Zhe Song},
  year={2024},
  eprint={2411.10818},
  archivePrefix={arXiv},
  primaryClass={cs.GR},
  url={https://arxiv.org/abs/2411.10818}, 
}

@article{zun2024dreamrunner,
    author = {Zun Wang and Jialu Li and Han Lin and Jaehong Yoon and Mohit Bansal},
    title  = {DreamRunner: Fine-Grained Storytelling Video Generation with Retrieval-Augmented Motion Adaptation},
    journal   = {arxiv},
    year      = {2024},
    url       = {https://arxiv.org/abs/2411.16657}
}

@article{geng2024motionprompting,
  author    = {Geng, Daniel and Herrmann, Charles and Hur, Junhwa and Cole, Forrester and Zhang, Serena and Pfaff, Tobias and Lopez-Guevara, Tatiana and Doersch, Carl and Aytar, Yusuf and Rubinstein, Michael and Sun, Chen and Wang, Oliver and Owens, Andrew and Sun, Deqing},
  title     = {Motion Prompting: Controlling Video Generation with Motion Trajectories},
  journal   = {arXiv preprint arXiv:2412.02700},
  year      = {2024},
}
```
</details>

### Other Additional Info

- OSS video generation models: [Mochi 1](https://github.com/genmoai/models) preview is an open state-of-the-art video generation model with high-fidelity motion and strong prompt adherence.
- Survey: The Dawn of Video Generation: Preliminary Explorations with SORA-like Models, [arXiv](https://arxiv.org/abs/2410.05227), [Project Page](https://ailab-cvc.github.io/VideoGen-Eval/), [GitHub Repo](https://github.com/AILab-CVC/VideoGen-Eval)

#### 📚 Dataset Works

#### 1. VidGen-1M: A Large-Scale Dataset for Text-to-video Generation
Zhiyu Tan, Xiaomeng Yang, Luozheng Qin, Hao Li

(Fudan University, ShangHai Academy of AI for Science)
<details span>
<summary><b>Abstract</b></summary>
The quality of video-text pairs fundamentally determines the upper bound of text-to-video models. Currently, the datasets used for training these models suffer from significant shortcomings, including low temporal consistency, poor-quality captions, substandard video quality, and imbalanced data distribution. The prevailing video curation process, which depends on image models for tagging and manual rule-based curation, leads to a high computational load and leaves behind unclean data. As a result, there is a lack of appropriate training datasets for text-to-video models. To address this problem, we present VidGen-1M, a superior training dataset for text-to-video models. Produced through a coarse-to-fine curation strategy, this dataset guarantees high-quality videos and detailed captions with excellent temporal consistency. When used to train the video generation model, this dataset has led to experimental results that surpass those obtained with other models.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2024 | **VidGen-1M: A Large-Scale Dataset for Text-to-video Generation**  | 5 Aug 2024  |          [Link](https://arxiv.org/abs/2408.02629)          | [Link](https://github.com/SAIS-FUXI/VidGen) | [Link](https://sais-fuxi.github.io/projects/vidgen-1m/)  |

<details close>
<summary>References</summary>

```
%axiv papers

@article{tan2024vidgen,
  title={VidGen-1M: A Large-Scale Dataset for Text-to-video Generation},
  author={Tan, Zhiyu and Yang, Xiaomeng, and Qin, Luozheng and Li Hao},
  booktitle={arXiv preprint arxiv:2408.02629},
  year={2024}
}


```
</details>

--------------

## Related Resources

### Text to 'other tasks'
(Here other tasks refer to *CAD*, *Model* and *Music* etc.)

#### Text to CAD
+ 2024 | CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM | arXiv 7 Nov 2024 | [Paper](https://arxiv.org/abs/2411.04954)  | [Code](https://github.com/CAD-MLLM/CAD-MLLM) | [Project Page](https://cad-mllm.github.io/) 
+ 2024 | Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts | NeurIPS 2024 Spotlight | [Paper](https://arxiv.org/abs/2409.17106)  | [Project Page](https://sadilkhan.github.io/text2cad-project/)

#### Text to Music
+ 2024 | FLUX that Plays Music | arXiv 1 Sep 2024 | [Paper](https://arxiv.org/abs/2409.00587) | [Code](https://github.com/feizc/FluxMusic) | [Hugging Face](https://huggingface.co/feizhengcong/FluxMusic)
</details>

#### Text to Model
+ 2024 | Text-to-Model: Text-Conditioned Neural Network Diffusion for Train-Once-for-All Personalization | arXiv 23 May 2024 | [Paper](https://arxiv.org/abs/2405.14132)



### Survey and Awesome Repos 
<details close>
<summary>🔥 Topic 1: 3D Gaussian Splatting</summary>
 
#### Survey
- [Gaussian Splatting: 3D Reconstruction and Novel View Synthesis, a Review](https://arxiv.org/abs/2405.03417), ArXiv Mon, 6 May 2024
- [Recent Advances in 3D Gaussian Splatting](https://arxiv.org/abs/2403.11134), ArXiv Sun, 17 Mar 2024
- [3D Gaussian as a New Vision Era: A Survey](https://arxiv.org/abs/2402.07181), ArXiv Sun, 11 Feb 2024
- [A Survey on 3D Gaussian Splatting](https://arxiv.org/pdf/2401.03890.pdf), ArXiv 2024
  
#### Awesome Repos
- Resource1: [Awesome 3D Gaussian Splatting Resources](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)
- Resource2: [3D Gaussian Splatting Papers](https://github.com/Awesome3DGS/3D-Gaussian-Splatting-Papers)
- Resource3: [3DGS and Beyond Docs](https://github.com/yangjiheng/3DGS_and_Beyond_Docs)

</details>

<details close>
<summary>🔥 Topic 2: AIGC 3D </summary>
 
#### Survey
- [Advances in 3D Generation: A Survey](https://arxiv.org/abs/2401.17807), ArXiv 2024
- [A Comprehensive Survey on 3D Content Generation](https://arxiv.org/abs/2402.01166), ArXiv 2024
- [A Survey On Text-to-3D Contents Generation In The Wild](https://arxiv.org/pdf/2405.09431), ArXiv 2024

#### Awesome Repos
- Resource1: [Awesome 3D AIGC 1](https://github.com/mdyao/Awesome-3D-AIGC) and [Awesome 3D AIGC 2](https://github.com/hitcslj/Awesome-AIGC-3D)
- Resource2: [Awesome Text 2 3D](https://github.com/StellarCheng/Awesome-Text-to-3D)

#### Benchmars
- text-to-3d generation: [GPT-4V(ision) is a Human-Aligned Evaluator for Text-to-3D Generation](https://arxiv.org/abs/2401.04092), Wu et al., arXiv 2024 | [Code](https://github.com/3DTopia/GPTEval3D)
</details>

<details close>
<summary>🔥 Topic 3: LLM 3D </summary>
 
#### Awesome Repos
- Resource1: [Awesome LLM 3D](https://github.com/ActiveVisionLab/Awesome-LLM-3D)


#### 3D Human
- Survey: [PROGRESS AND PROSPECTS IN 3D GENERATIVE AI: A TECHNICAL OVERVIEW INCLUDING 3D HUMAN](https://arxiv.org/pdf/2401.02620.pdf), ArXiv 2024
- Survey: [A Survey on 3D Human Avatar Modeling -- From Reconstruction to Generation](https://arxiv.org/abs/2406.04253), ArXiv 6 June 2024
- Resource1: [Awesome Digital Human](https://github.com/weihaox/awesome-digital-human)
- Resource2: [Awesome-Avatars](https://github.com/pansanity666/Awesome-Avatars)

</details>

<details close>
<summary>🔥 Topic 4: AIGC 4D </summary>
	
#### Awesome Repos
- Resource1: [Awesome 4D Generation](https://github.com/cwchenwang/awesome-4d-generation)

</details>

<details close>
<summary>Dynamic Gaussian Splatting</summary>
<details close>
<summary>Neural Deformable 3D Gaussians</summary>
 
(CVPR 2024) Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction [Paper](https://arxiv.org/abs/2309.13101) [Code](https://github.com/ingra14m/Deformable-3D-Gaussians) [Page](https://ingra14m.github.io/Deformable-Gaussians/)
 
(CVPR 2024) 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering [Paper](https://arxiv.org/abs/2310.08528) [Code](https://github.com/hustvl/4DGaussians) [Page](https://guanjunwu.github.io/4dgs/index.html)

(CVPR 2024) SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes [Paper](https://arxiv.org/abs/2312.14937) [Code](https://github.com/yihua7/SC-GS) [Page](https://yihua7.github.io/SC-GS-web/)

(CVPR 2024, Highlight) 3DGStream: On-the-Fly Training of 3D Gaussians for Efficient Streaming of Photo-Realistic Free-Viewpoint Videos [Paper](https://arxiv.org/abs/2403.01444) [Code](https://github.com/SJoJoK/3DGStream) [Page](https://sjojok.github.io/3dgstream/)

</details>

<details close>
<summary>4D Gaussians</summary>

(ArXiv 2024.02.07) 4D Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes [Paper](https://arxiv.org/abs/2402.03307)
 
(ICLR 2024) Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting [Paper](https://arxiv.org/abs/2310.10642) [Code](https://github.com/fudan-zvg/4d-gaussian-splatting) [Page](https://fudan-zvg.github.io/4d-gaussian-splatting/)

</details>

<details close>
<summary>Dynamic 3D Gaussians</summary>

(CVPR 2024) Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle [Paper](https://arxiv.org/abs/2312.03431) [Page](https://nju-3dv.github.io/projects/Gaussian-Flow/)
 
(3DV 2024) Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis [Paper](https://arxiv.org/abs/2308.09713) [Code](https://github.com/JonathonLuiten/Dynamic3DGaussians) [Page](https://dynamic3dgaussians.github.io/)

</details>

</details>

--------------

## License 
Awesome Text2X Resources is released under the [MIT license](./LICENSE).
