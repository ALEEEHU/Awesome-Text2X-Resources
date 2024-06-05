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
* `2024.04.05` adjusted the layout and added accepted lists and ArXiv lists to each section.


<div><div align="center">
	<img width="500" height="350" src="media/logo.svg" alt="Awesome"></div>

## Table of contents

- [Text to 4D](#text-to-4d)
  * [Accepted Papers](#-4d-accepted-papers)
  * [ArXiv Papers](#-4d-arxiv-papers)
- [Text to Scene](#text-to-scene)
  * [Accepted Papers](#-scene-accepted-papers)
  * [ArXiv Papers](#-scene-arxiv-papers)
- [Text to 3D Human](#text-to-3d-human)
  * [Paper lists](#text-to-3d-human-paper-lists)
  * [Additional Info](#additional-info)
- [Text to Human Motion](#text-to-human-motion)
  * [Paper lists](#text-to-human-motion-paper-lists)
  * [Datasets](#datasets)
- [Related Resources](#related-resources)
  * [Dynamic Gaussian Splatting](#dynamic-gaussian-splatting)
  * [Survey and Awesome Repos](#survey-and-awesome-repos)

## Update Logs
<details span>
<summary><b>Update Logs:</b></summary>
<br>

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

| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **Text-To-4D Dynamic Scene Generation**  | ICML 2023 |          [Link](https://arxiv.org/abs/2301.11280)          | -  | [Link](https://make-a-video3d.github.io/)  |
| 2023 | **Consistent4D: Consistent 360° Dynamic Object Generation from Monocular Video**  | ICLR 2024 |          [Link](https://arxiv.org/abs/2311.02848)          | [link](https://github.com/yanqinJiang/Consistent4D)  | [Link](https://consistent4d.github.io/)  |
| 2023 | **4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2311.17984)          | [link](https://github.com/sherwinbahmani/4dfy)  | [Link](https://sherwinbahmani.github.io/4dfy/)  |
| 2023 | **Dream-in-4D: A Unified Approach for Text- and Image-guided 4D Scene Generation**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2311.16854)          | [Link](https://github.com/NVlabs/dream-in-4d)  | [Link](https://research.nvidia.com/labs/nxp/dream-in-4d/)  |
| 2023 | **Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2312.13763)          | - | [Link](https://research.nvidia.com/labs/toronto-ai/AlignYourGaussians/)  |
| 2023 | **Control4D: Efficient 4D Portrait Editing with Text**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2305.20082)          | -  | [Link](https://control4darxiv.github.io./)  |


<details close>
<summary>Accepted Papers References</summary>

```
%accepted papers

@article{singer2023text4d,
  author = {Singer, Uriel and Sheynin, Shelly and Polyak, Adam and Ashual, Oron and
           Makarov, Iurii and Kokkinos, Filippos and Goyal, Naman and Vedaldi, Andrea and
           Parikh, Devi and Johnson, Justin and Taigman, Yaniv},
  title = {Text-To-4D Dynamic Scene Generation},
  journal = {arXiv:2301.11280},
  year = {2023},
}

@inproceedings{
jiang2024consistentd,
title={Consistent4D: Consistent 360{\textdegree} Dynamic Object Generation from Monocular Video},
author={Yanqin Jiang and Li Zhang and Jin Gao and Weiming Hu and Yao Yao},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=sPUrdFGepF}
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

#### 6. TC4D: Trajectory-Conditioned Text-to-4D Generation
Sherwin Bahmani, Xian Liu, Yifan Wang, Ivan Skorokhodov, Victor Rong, Ziwei Liu, Xihui Liu, Jeong Joon Park, Sergey Tulyakov, Gordon Wetzstein, Andrea Tagliasacchi, David B. Lindell

(University of Toronto, Vector Institute, Snap Inc., CUHK, Stanford University, NTU, HKU, University of Michigan, SFU, Google DeepMind)
<details span>
<summary><b>Abstract</b></summary>
Recent techniques for text-to-4D generation synthesize dynamic 3D scenes using supervision from pre-trained text-to-video models. However, existing representations for motion, such as deformation models or time-dependent neural representations, are limited in the amount of motion they can generate-they cannot synthesize motion extending far beyond the bounding box used for volume rendering. The lack of a more flexible motion model contributes to the gap in realism between 4D generation methods and recent, near-photorealistic video generation models. Here, we propose TC4D: trajectory-conditioned text-to-4D generation, which factors motion into global and local components. We represent the global motion of a scene's bounding box using rigid transformation along a trajectory parameterized by a spline. We learn local deformations that conform to the global trajectory using supervision from a text-to-video model. Our approach enables the synthesis of scenes animated along arbitrary trajectories, compositional scene generation, and significant improvements to the realism and amount of generated motion, which we evaluate qualitatively and through a user study. 
</details>

#### 7. Comp4D: LLM-Guided Compositional 4D Scene Generation
Dejia Xu, Hanwen Liang, Neel P. Bhatt, Hezhen Hu, Hanxue Liang, Konstantinos N. Plataniotis, Zhangyang Wang

(University of Texas at Austin, University of Toronto, University of Cambridge)
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in diffusion models for 2D and 3D content creation have sparked a surge of interest in generating 4D content. However, the scarcity of 3D scene datasets constrains current methodologies to primarily object-centric generation. To overcome this limitation, we present Comp4D, a novel framework for Compositional 4D Generation. Unlike conventional methods that generate a singular 4D representation of the entire scene, Comp4D innovatively constructs each 4D object within the scene separately. Utilizing Large Language Models (LLMs), the framework begins by decomposing an input text prompt into distinct entities and maps out their trajectories. It then constructs the compositional 4D scene by accurately positioning these objects along their designated paths. To refine the scene, our method employs a compositional score distillation technique guided by the pre-defined trajectories, utilizing pre-trained diffusion models across text-to-image, text-to-video, and text-to-3D domains. Extensive experiments demonstrate our outstanding 4D content creation capability compared to prior arts, showcasing superior visual quality, motion fidelity, and enhanced object interactions.
</details>

#### 8. STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians
Yifei Zeng, Yanqin Jiang, Siyu Zhu, Yuanxun Lu, Youtian Lin, Hao Zhu, Weiming Hu, Xun Cao, Yao Yao

(Nanjing University, CASIA, Fudan University)
<details span>
<summary><b>Abstract</b></summary>
Recent progress in pre-trained diffusion models and 3D generation have spurred interest in 4D content creation. However, achieving high-fidelity 4D generation with spatial-temporal consistency remains a challenge. In this work, we propose STAG4D, a novel framework that combines pre-trained diffusion models with dynamic 3D Gaussian splatting for high-fidelity 4D generation. Drawing inspiration from 3D generation techniques, we utilize a multi-view diffusion model to initialize multi-view images anchoring on the input video frames, where the video can be either real-world captured or generated by a video diffusion model. To ensure the temporal consistency of the multi-view sequence initialization, we introduce a simple yet effective fusion strategy to leverage the first frame as a temporal anchor in the self-attention computation. With the almost consistent multi-view sequences, we then apply the score distillation sampling to optimize the 4D Gaussian point cloud. The 4D Gaussian spatting is specially crafted for the generation task, where an adaptive densification strategy is proposed to mitigate the unstable Gaussian gradient for robust optimization. Notably, the proposed pipeline does not require any pre-training or fine-tuning of diffusion networks, offering a more accessible and practical solution for the 4D generation task. Extensive experiments demonstrate that our method outperforms prior 4D generation works in rendering quality, spatial-temporal consistency, and generation robustness, setting a new state-of-the-art for 4D generation from diverse inputs, including text, image, and video.
</details>

#### 9. SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer
Zijie Wu, Chaohui Yu, Yanqin Jiang, Chenjie Cao, Fan Wang, Xiang Bai

(Huazhong University of Science and Technology, DAMO Academy Alibaba Group)
<details span>
<summary><b>Abstract</b></summary>
Recent advances in 2D/3D generative models enable the generation of dynamic 3D objects from a single-view video. Existing approaches utilize score distillation sampling to form the dynamic scene as dynamic NeRF or dense 3D Gaussians. However, these methods struggle to strike a balance among reference view alignment, spatio-temporal consistency, and motion fidelity under single-view conditions due to the implicit nature of NeRF or the intricate dense Gaussian motion prediction. To address these issues, this paper proposes an efficient, sparse-controlled video-to-4D framework named SC4D, that decouples motion and appearance to achieve superior video-to-4D generation. Moreover, we introduce Adaptive Gaussian (AG) initialization and Gaussian Alignment (GA) loss to mitigate shape degeneration issue, ensuring the fidelity of the learned motion and shape. Comprehensive experimental results demonstrate that our method surpasses existing methods in both quality and efficiency. In addition, facilitated by the disentangled modeling of motion and appearance of SC4D, we devise a novel application that seamlessly transfers the learned motion onto a diverse array of 4D entities according to textual descriptions.
</details>

#### 10. DreamScene4D: Dynamic Multi-Object Scene Generation from Monocular Videos
Wen-Hsuan Chu*, Lei Ke*, Katerina Fragkiadaki (Carnegie Mellon University)
<details span>
<summary><b>Abstract</b></summary>
Existing VLMs can track in-the-wild 2D video objects while current generative models provide powerful visual priors for synthesizing novel views for the highly under-constrained 2D-to-3D object lifting. Building upon this exciting progress, we present DreamScene4D, the first approach that can generate three-dimensional dynamic scenes of multiple objects from monocular in-the-wild videos with large object motion across occlusions and novel viewpoints. Our key insight is to design a "decompose-then-recompose" scheme to factorize both the whole video scene and each object's 3D motion. We first decompose the video scene by using open-vocabulary mask trackers and an adapted image diffusion model to segment, track, and amodally complete the objects and background in the video. Each object track is mapped to a set of 3D Gaussians that deform and move in space and time. We also factorize the observed motion into multiple components to handle fast motion. The camera motion can be inferred by re-rendering the background to match the video frames. For the object motion, we first model the object-centric deformation of the objects by leveraging rendering losses and multi-view generative priors in an object-centric frame, then optimize object-centric to world-frame transformations by comparing the rendered outputs against the perceived pixel and optical flow. Finally, we recompose the background and objects and optimize for relative object scales using monocular depth prediction guidance. We show extensive results on the challenging DAVIS, Kubric, and self-captured videos, detail some limitations, and provide future directions. Besides 4D scene generation, our results show that DreamScene4D enables accurate 2D point motion tracking by projecting the inferred 3D trajectories to 2D, while never explicitly trained to do so.
</details>

#### 11. Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models
Hanwen Liang, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N. Plataniotis, Yao Zhao, Yunchao Wei

(University of Toronto, Beijing Jiaotong University, University of Texas at Austin, University of Cambridge)
<details span>
<summary><b>Abstract</b></summary>
The availability of large-scale multimodal datasets and advancements in diffusion models have significantly accelerated progress in 4D content generation. Most prior approaches rely on multiple image or video diffusion models, utilizing score distillation sampling for optimization or generating pseudo novel views for direct supervision. However, these methods are hindered by slow optimization speeds and multi-view inconsistency issues. Spatial and temporal consistency in 4D geometry has been extensively explored respectively in 3D-aware diffusion models and traditional monocular video diffusion models. Building on this foundation, we propose a strategy to migrate the temporal consistency in video diffusion models to the spatial-temporal consistency required for 4D generation. Specifically, we present a novel framework, Diffusion4D, for efficient and scalable 4D content generation. Leveraging a meticulously curated dynamic 3D dataset, we develop a 4D-aware video diffusion model capable of synthesizing orbital views of dynamic 3D assets. To control the dynamic strength of these assets, we introduce a 3D-to-4D motion magnitude metric as guidance. Additionally, we propose a novel motion magnitude reconstruction loss and 3D-aware classifier-free guidance to refine the learning and generation of motion dynamics. After obtaining orbital views of the 4D asset, we perform explicit 4D construction with Gaussian splatting in a coarse-to-fine manner. The synthesized multi-view consistent 4D image set enables us to swiftly generate high-fidelity and diverse 4D assets within just several minutes. Extensive experiments demonstrate that our method surpasses prior state-of-the-art techniques in terms of generation efficiency and 4D geometry consistency across various prompt modalities.
</details>

#### 12. Vidu4D: Single Generated Video to High-Fidelity 4D Reconstruction with Dynamic Gaussian Surfels
Yikai Wang, Xinzhou Wang, Zilong Chen, Zhengyi Wang, Fuchun Sun, Jun Zhu

(Tsinghua University, ShengShu, Tongji University)
<details span>
<summary><b>Abstract</b></summary>
Video generative models are receiving particular attention given their ability to generate realistic and imaginative frames. Besides, these models are also observed to exhibit strong 3D consistency, significantly enhancing their potential to act as world simulators. In this work, we present Vidu4D, a novel reconstruction model that excels in accurately reconstructing 4D (i.e., sequential 3D) representations from single generated videos, addressing challenges associated with non-rigidity and frame distortion. This capability is pivotal for creating high-fidelity virtual contents that maintain both spatial and temporal coherence. At the core of Vidu4D is our proposed Dynamic Gaussian Surfels (DGS) technique. DGS optimizes time-varying warping functions to transform Gaussian surfels (surface elements) from a static state to a dynamically warped state. This transformation enables a precise depiction of motion and deformation over time. To preserve the structural integrity of surface-aligned Gaussian surfels, we design the warped-state geometric regularization based on continuous warping fields for estimating normals. Additionally, we learn refinements on rotation and scaling parameters of Gaussian surfels, which greatly alleviates texture flickering during the warping process and enhances the capture of fine-grained appearance details. Vidu4D also contains a novel initialization state that provides a proper start for the warping fields in DGS. Equipping Vidu4D with an existing video generative model, the overall framework demonstrates high-fidelity text-to-4D generation in both appearance and geometry.
</details>

#### 13. EG4D: Explicit Generation of 4D Object without Score Distillation
Qi Sun, Zhiyang Guo, Ziyu Wan, Jing Nathan Yan, Shengming Yin, Wengang Zhou, Jing Liao, Houqiang Li

(USTC, City University of Hong Kong, Cornell University)
<details span>
<summary><b>Abstract</b></summary>
In recent years, the increasing demand for dynamic 3D assets in design and gaming applications has given rise to powerful generative pipelines capable of synthesizing high-quality 4D objects. Previous methods generally rely on score distillation sampling (SDS) algorithm to infer the unseen views and motion of 4D objects, thus leading to unsatisfactory results with defects like over-saturation and Janus problem. Therefore, inspired by recent progress of video diffusion models, we propose to optimize a 4D representation by explicitly generating multi-view videos from one input image. However, it is far from trivial to handle practical challenges faced by such a pipeline, including dramatic temporal inconsistency, inter-frame geometry and texture diversity, and semantic defects brought by video generation results. To address these issues, we propose DG4D, a novel multi-stage framework that generates high-quality and consistent 4D assets without score distillation. Specifically, collaborative techniques and solutions are developed, including an attention injection strategy to synthesize temporal-consistent multi-view videos, a robust and efficient dynamic reconstruction method based on Gaussian Splatting, and a refinement stage with diffusion prior for semantic restoration. The qualitative results and user preference study demonstrate that our framework outperforms the baselines in generation quality by a considerable margin.
</details>

#### 14. PLA4D: Pixel-Level Alignments for Text-to-4D Gaussian Splatting
Qiaowei Miao, Yawei Luo, Yi Yang (Zhejiang University)
<details span>
<summary><b>Abstract</b></summary>
As text-conditioned diffusion models (DMs) achieve breakthroughs in image, video, and 3D generation, the research community's focus has shifted to the more challenging task of text-to-4D synthesis, which introduces a temporal dimension to generate dynamic 3D objects. In this context, we identify Score Distillation Sampling (SDS), a widely used technique for text-to-3D synthesis, as a significant hindrance to text-to-4D performance due to its Janus-faced and texture-unrealistic problems coupled with high computational costs. In this paper, we propose Pixel-Level Alignments for Text-to-4D Gaussian Splatting (PLA4D), a novel method that utilizes text-to-video frames as explicit pixel alignment targets to generate static 3D objects and inject motion into them. Specifically, we introduce Focal Alignment to calibrate camera poses for rendering and GS-Mesh Contrastive Learning to distill geometry priors from rendered image contrasts at the pixel level. Additionally, we develop Motion Alignment using a deformation network to drive changes in Gaussians and implement Reference Refinement for smooth 4D object surfaces. These techniques enable 4D Gaussian Splatting to align geometry, texture, and motion with generated videos at the pixel level. Compared to previous methods, PLA4D produces synthesized outputs with better texture details in less time and effectively mitigates the Janus-faced problem. PLA4D is fully implemented using open-source models, offering an accessible, user-friendly, and promising direction for 4D digital content creation.
</details>

---

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **Animate124: Animating One Image to 4D Dynamic Scene**  | 24 Nov 2023 |          [Link](https://arxiv.org/abs/2311.14603)          | [link](https://github.com/HeliosZhao/Animate124)  | [Link](https://animate124.github.io/)  |
| 2023 | **4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency**  | 28 Dec 2023 |          [Link](https://arxiv.org/abs/2312.17225)          | [link](https://github.com/VITA-Group/4DGen)  | [Link](https://vita-group.github.io/4DGen/)  |
| 2023 | **DreamGaussian4D:Generative 4D Gaussian Splatting**  | 28 Dec 2023 |          [Link](https://arxiv.org/abs/2312.17142)          | [link](https://github.com/jiawei-ren/dreamgaussian4d)  | [Link](https://jiawei-ren.github.io/projects/dreamgaussian4d/)  |
| 2024 | **Fast Dynamic 3D Object Generation from a Single-view Video**  | 16 Jan 2024 |          [Link](https://arxiv.org/abs/2401.08742)          | [link](https://github.com/fudan-zvg/Efficient4D)  | [Link](https://fudan-zvg.github.io/Efficient4D/)  |
| 2024 | **GaussianFlow: Splatting Gaussian Dynamics for 4D Content Creation**  | 19 Mar 2024 |         [Link](https://arxiv.org/abs/2403.12365)         | [link](https://github.com/Zerg-Overmind/GaussianFlow)   | [Link](https://zerg-overmind.github.io/GaussianFlow.github.io/) |
| 2024 | **TC4D: Trajectory-Conditioned Text-to-4D Generation**  | 26 Mar 2024 |         [Link](https://arxiv.org/abs/2403.17920)         | [link](https://github.com/sherwinbahmani/tc4d)   | [Link](https://sherwinbahmani.github.io/tc4d/) |
| 2024 | **Comp4D: LLM-Guided Compositional 4D Scene Generation**  |  25 Mar 2024 |          [Link](https://arxiv.org/abs/2403.16993)          | [Link](https://github.com/VITA-Group/Comp4D)  |[Link](https://vita-group.github.io/Comp4D/#) |
| 2024 | **STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians**  | 22 Mar 2024 |          [Link](https://arxiv.org/abs/2403.14939)          | [Link](https://github.com/zeng-yifei/STAG4D)  |[Link](https://nju-3dv.github.io/projects/STAG4D/) |
| 2024 | **SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer**  | 4 Apr 2024  |          [Link](https://arxiv.org/abs/2404.03736)          | [Link](https://github.com/JarrentWu1031/SC4D)  |[Link](https://sc4d.github.io/) |
| 2024 | **DreamScene4D: Dynamic Multi-Object Scene Generation from Monocular Videos**  | 3 May 2024 |          [Link](https://arxiv.org/abs/2405.02280)          | [Link](https://github.com/dreamscene4d/dreamscene4d)  |[Link](https://dreamscene4d.github.io/) |
| 2024 | **Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models**  | 26 May 2024 |          [Link](https://arxiv.org/abs/2405.16645)          | [Link](https://github.com/VITA-Group/Diffusion4D)  |[Link](https://vita-group.github.io/Diffusion4D/) |
| 2024 | **Vidu4D: Single Generated Video to High-Fidelity 4D Reconstruction with Dynamic Gaussian Surfels**  | 27 May 2024 |       [link](https://arxiv.org/abs/2405.16822)        | [link](https://github.com/yikaiw/vidu4d)  | [Link](https://vidu4d-dgs.github.io/)  |
| 2024 | **EG4D: Explicit Generation of 4D Object without Score Distillation**  | 28 May 2024 |          [Link](https://arxiv.org/abs/2405.18132)          | [Link](https://github.com/jasongzy/EG4D)  | -- |
| 2024 | **PLA4D: Pixel-Level Alignments for Text-to-4D Gaussian Splatting**  | 4 Jun 2024 |          [Link](https://arxiv.org/abs/2405.19957)          | --  | [Link](https://github.com/MiaoQiaowei/PLA4D.github.io) |

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

@article{bah2024tc4d,
  author = {Bahmani, Sherwin and Liu, Xian and Yifan, Wang and Skorokhodov, Ivan and Rong, Victor and Liu, Ziwei and Liu, Xihui and Park, Jeong Joon and Tulyakov, Sergey and Wetzstein, Gordon and Tagliasacchi, Andrea and Lindell, David B.},
  title = {TC4D: Trajectory-Conditioned Text-to-4D Generation},
  journal = {arXiv},
  year = {2024},
}

@misc{xu2024comp4d,
      title={Comp4D: LLM-Guided Compositional 4D Scene Generation}, 
      author={Dejia Xu and Hanwen Liang and Neel P. Bhatt and Hezhen Hu and Hanxue Liang and Konstantinos N. Plataniotis and Zhangyang Wang},
      year={2024},
      eprint={2403.16993},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{zeng2024stag4d,
    title={STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians}, 
    author={Yifei Zeng and Yanqin Jiang and Siyu Zhu and Yuanxun Lu and Youtian Lin and Hao Zhu and Weiming Hu and Xun Cao and Yao Yao},
    year={2024}
}

@article{wu2024sc4d,
    author = {Wu, Zijie and Yu, Chaohui and Jiang, Yanqin and Cao, Chenjie and Wang Fan and Bai, Xiang.},
    title  = {SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer},
    journal = {arxiv:2404.03736},
    year   = {2024}
}

@article{dreamscene4d,
  title={DreamScene4D: Dynamic Multi-Object Scene Generation from Monocular Videos},
  author={Chu, Wen-Hsuan and Ke, Lei and Fragkiadaki, Katerina},
  journal={arxiv:2405.02280},
  year={2024}
}

@misc{liang2024diffusion4d,
      title={Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models}, 
      author={Hanwen Liang and Yuyang Yin and Dejia Xu and Hanxue Liang and Zhangyang Wang and Konstantinos N. Plataniotis and Yao Zhao and Yunchao Wei},
      year={2024},
      eprint={2405.16645},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{wang2024vidu4d,
  title={Vidu4D: Single Generated Video to High-Fidelity 4D Reconstruction with Dynamic Gaussian Surfels},
  author={Yikai Wang and Xinzhou Wang and Zilong Chen and Zhengyi Wang and Fuchun Sun and Jun Zhu},
  journal={arXiv preprint arXiv},
  year={2024}
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
```
</details>

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

#### 5. ReplaceAnything3D:Text-Guided 3D Scene Editing with Compositional Neural Radiance Fields
Edward Bartrum, Thu Nguyen-Phuoc, Chris Xie, Zhengqin Li, Numair Khan, Armen Avetisyan, Douglas Lanman, Lei Xiao

(University College London, Alan Turing Institute, Reality Labs Research Meta)
<details span>
<summary><b>Abstract</b></summary>
We introduce ReplaceAnything3D model (RAM3D), a novel text-guided 3D scene editing method that enables the replacement of specific objects within a scene. Given multi-view images of a scene, a text prompt describing the object to replace, and a text prompt describing the new object, our Erase-and-Replace approach can effectively swap objects in the scene with newly generated content while maintaining 3D consistency across multiple viewpoints. We demonstrate the versatility of ReplaceAnything3D by applying it to various realistic 3D scenes, showcasing results of modified foreground objects that are well-integrated with the rest of the scene without affecting its overall integrity.
</details>

#### 6. 3D-SceneDreamer: Text-Driven 3D-Consistent Scene Generation
Frank Zhang, Yibo Zhang, Quan Zheng, Rui Ma, Wei Hua, Hujun Bao, Weiwei Xu, Changqing Zou

(Zhejiang University, Jilin University, Zhejiang Lab, Institute of Software Chinese Academy of Sciences)
<details span>
<summary><b>Abstract</b></summary>
Text-driven 3D scene generation techniques have made rapid progress in recent years. Their success is mainly attributed to using existing generative models to iteratively perform image warping and inpainting to generate 3D scenes. However, these methods heavily rely on the outputs of existing models, leading to error accumulation in geometry and appearance that prevent the models from being used in various scenarios (e.g., outdoor and unreal scenarios). To address this limitation, we generatively refine the newly generated local views by querying and aggregating global 3D information, and then progressively generate the 3D scene. Specifically, we employ a tri-plane features-based NeRF as a unified representation of the 3D scene to constrain global 3D consistency, and propose a generative refinement network to synthesize new contents with higher quality by exploiting the natural image prior from 2D diffusion model as well as the global 3D information of the current scene. Our extensive experiments demonstrate that, in comparison to previous methods, our approach supports wide variety of scene generation and arbitrary camera trajectories with improved visual quality and 3D consistency.
</details>

#### 7. DreamScene: 3D Gaussian-based Text-to-3D Scene Generation via Formation Pattern Sampling
Haoran Li, Haolin Shi, Wenli Zhang, Wenjun Wu, Yong Liao, Lin Wang, Lik-hang Lee, Pengyuan Zhou

(University of Science and Technology of China, HKUST, The Hong Kong Polytechnic University)
<details span>
<summary><b>Abstract</b></summary>
Text-to-3D scene generation holds immense potential for the gaming, film, and architecture sectors. Despite significant progress, existing methods struggle with maintaining high quality, consistency, and editing flexibility. In this paper, we propose DreamScene, a 3D Gaussian-based novel text-to-3D scene generation framework, to tackle the aforementioned three challenges mainly via two strategies. First, DreamScene employs Formation Pattern Sampling (FPS), a multi-timestep sampling strategy guided by the formation patterns of 3D objects, to form fast, semantically rich, and high-quality representations. FPS uses 3D Gaussian filtering for optimization stability, and leverages reconstruction techniques to generate plausible textures. Second, DreamScene employs a progressive three-stage camera sampling strategy, specifically designed for both indoor and outdoor settings, to effectively ensure object-environment integration and scene-wide 3D consistency. Last, DreamScene enhances scene editing flexibility by integrating objects and environments, enabling targeted adjustments. Extensive experiments validate DreamScene's superiority over current state-of-the-art techniques, heralding its wide-ranging potential for diverse applications.
</details>

#### 8. RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion
Jaidev Shriram, Alex Trevithick, Lingjie Liu, Ravi Ramamoorthi (University of California San Diego, University of Pennsylvania)
<details span>
<summary><b>Abstract</b></summary>
We introduce RealmDreamer, a technique for generation of general forward-facing 3D scenes from text descriptions. Our technique optimizes a 3D Gaussian Splatting representation to match complex text prompts. We initialize these splats by utilizing the state-of-the-art text-to-image generators, lifting their samples into 3D, and computing the occlusion volume. We then optimize this representation across multiple views as a 3D inpainting task with image-conditional diffusion models. To learn correct geometric structure, we incorporate a depth diffusion model by conditioning on the samples from the inpainting model, giving rich geometric structure. Finally, we finetune the model using sharpened samples from image generators. Notably, our technique does not require video or multi-view data and can synthesize a variety of high-quality 3D scenes in different styles, consisting of multiple objects. Its generality additionally allows 3D synthesis from a single image.
</details>

#### 9. DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting
Shijie Zhou, Zhiwen Fan, Dejia Xu, Haoran Chang, Pradyumna Chari, Tejas Bharadwaj, Suya You, Zhangyang Wang, Achuta Kadambi

(University of California Los Angeles, University of Texas at Austin, DEVCOM ARL)
<details span>
<summary><b>Abstract</b></summary>
The increasing demand for virtual reality applications has highlighted the significance of crafting immersive 3D assets. We present a text-to-3D 360∘ scene generation pipeline that facilitates the creation of comprehensive 360∘ scenes for in-the-wild environments in a matter of minutes. Our approach utilizes the generative power of a 2D diffusion model and prompt self-refinement to create a high-quality and globally coherent panoramic image. This image acts as a preliminary "flat" (2D) scene representation. Subsequently, it is lifted into 3D Gaussians, employing splatting techniques to enable real-time exploration. To produce consistent 3D geometry, our pipeline constructs a spatially coherent structure by aligning the 2D monocular depth into a globally optimized point cloud. This point cloud serves as the initial state for the centroids of 3D Gaussians. In order to address invisible issues inherent in single-view inputs, we impose semantic and geometric constraints on both synthesized and input camera views as regularizations. These guide the optimization of Gaussians, aiding in the reconstruction of unseen regions. In summary, our method offers a globally consistent 3D scene within a 360∘ perspective, providing an enhanced immersive experience over existing techniques. 
</details>

---

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints**  | 5 Oct 2023 |          [Link](https://arxiv.org/abs/2310.03602)          | [Link](https://github.com/fangchuan/Ctrl-Room)    | [Link](https://fangchuan.github.io/ctrl-room.github.io/)  |
| 2023 | **Text2Immersion: Generative Immersive Scene with 3D Gaussians**  | 14 Dec 2023 |          [Link](https://arxiv.org/abs/2312.09242)          | --   | [Link](https://ken-ouyang.github.io/text2immersion/index.html)  |
| 2023 | **ShowRoom3D: Text to High-Quality 3D Room Generation Using 3D Priors**  | 20 Dec 2023 |          [Link](https://arxiv.org/abs/2312.13324)          | [Link](https://github.com/showlab/ShowRoom3D)  | [Link](https://showroom3d.github.io/)  |
| 2023 | **Detailed Human-Centric Text Description-Driven Large Scene Synthesis**  | 30 Nov 2023 |          [Link](https://arxiv.org/abs/2311.18654)          | --  |-- |
| 2024 | **ReplaceAnything3D:Text-Guided 3D Scene Editing with Compositional Neural Radiance Fields**  | 31 Jan 2024 |          [Link](https://arxiv.org/abs/2401.17895)          | --  |[Link](https://replaceanything3d.github.io/) |
| 2024 | **3D-SceneDreamer: Text-Driven 3D-Consistent Scene Generation**  | 14 Mar 2024 |          [Link](https://arxiv.org/abs/2403.09439)          | --  | -- |
| 2024 | **DreamScene: 3D Gaussian-based Text-to-3D Scene Generation via Formation Pattern Sampling**  | 4 Apr 2024 |          [Link](https://arxiv.org/abs/2404.03575)          | [Link](https://github.com/DreamScene-Project/DreamScene)  |[Link](https://dreamscene-project.github.io/) |
| 2024 | **RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion**  | 10 Apr 2024 |          [Link](https://arxiv.org/abs/2404.07199)          | Coming Soon!  |[Link](https://realmdreamer.github.io/) |
| 2024 | **DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting**  | 10 Apr 2024 |          [Link](https://arxiv.org/abs/2404.06903)          | Coming Soon!  |[Link](https://dreamscene360.github.io/) |


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

@misc{zhang20243dscenedreamer,
      title={3D-SceneDreamer: Text-Driven 3D-Consistent Scene Generation}, 
      author={Frank Zhang and Yibo Zhang and Quan Zheng and Rui Ma and Wei Hua and Hujun Bao and Weiwei Xu and Changqing Zou},
      year={2024},
      eprint={2403.09439},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{li2024dreamscene,
  title={DreamScene: 3D Gaussian-based Text-to-3D Scene Generation via Formation Pattern Sampling},
  author={Li, Haoran and Shi, Haolin and Zhang, Wenli and Wu, Wenjun and Liao, Yong and Lin Wang and Lik-hang Lee and Zhou, Pengyuan},
  journal={arXiv preprint arXiv:2404.03575},
  year={2024}
}

@article{shriram2024realmdreamer,
        title={RealmDreamer: Text-Driven 3D Scene Generation with 
                Inpainting and Depth Diffusion},
        author={Jaidev Shriram and Alex Trevithick and Lingjie Liu and Ravi Ramamoorthi},
        journal={arXiv},
        year={2024}
}

@article{zhou2024dreamscene360,
  author    = {Zhou, Shijie and Fan, Zhiwen and Xu, Dejia and Chang, Haoran and Chari, Pradyumna and Bharadwaj, Tejas You, Suya and Wang, Zhangyang and Kadambi, Achuta},
  title     = {DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting},
  journal   = {arXiv},
  year      = {2024},
}

```
</details>

--------------

## Text to 3D Human

### 1. HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting  
Xian Liu, Xiaohang Zhan, Jiaxiang Tang, Ying Shan, Gang Zeng, Dahua Lin, Xihui Liu, Ziwei Liu (CUHK, Tencent AI Lab, PKU, HKU, NTU)
<details span>
<summary><b>Abstract</b></summary>
Realistic 3D human generation from text prompts is a desirable yet challenging task. Existing methods optimize 3D representations like mesh or neural fields via score distillation sampling (SDS), which suffers from inadequate fine details or excessive training time. In this paper, we propose an efficient yet effective framework, HumanGaussian, that generates high-quality 3D humans with fine-grained geometry and realistic appearance. Our key insight is that 3D Gaussian Splatting is an efficient renderer with periodic Gaussian shrinkage or growing, where such adaptive density control can be naturally guided by intrinsic human structures. Specifically, 1) we first propose a Structure-Aware SDS that simultaneously optimizes human appearance and geometry. The multi-modal score function from both RGB and depth space is leveraged to distill the Gaussian densification and pruning process. 2) Moreover, we devise an Annealed Negative Prompt Guidance by decomposing SDS into a noisier generative score and a cleaner classifier score, which well addresses the over-saturation issue. The floating artifacts are further eliminated based on Gaussian size in a prune-only phase to enhance generation smoothness. Extensive experiments demonstrate the superior efficiency and competitive quality of our framework, rendering vivid 3D humans under diverse scenarios.
</details>

### 2. HumanNorm: Learning Normal Diffusion Model for High-quality and Realistic 3D Human Generation  
Xin Huang, Ruizhi Shao, Qi Zhang, Hongwen Zhang, Ying Feng, Yebin Liu, Qing Wang 

(Northwestern Polytechnical University, Tsinghua University)
<details span>
<summary><b>Abstract</b></summary>
Recent text-to-3D methods employing diffusion models have made significant advancements in 3D human generation. However, these approaches face challenges due to the limitations of text-to-image diffusion models, which lack an understanding of 3D structures. Consequently, these methods struggle to achieve high-quality human generation, resulting in smooth geometry and cartoon-like appearances. In this paper, we propose HumanNorm, a novel approach for high-quality and realistic 3D human generation. The main idea is to enhance the model's 2D perception of 3D geometry by learning a normal-adapted diffusion model and a normal-aligned diffusion model. The normal-adapted diffusion model can generate high-fidelity normal maps corresponding to user prompts with view-dependent and body-aware text. The normal-aligned diffusion model learns to generate color images aligned with the normal maps, thereby transforming physical geometry details into realistic appearance. Leveraging the proposed normal diffusion model, we devise a progressive geometry generation strategy and a multi-step Score Distillation Sampling (SDS) loss to enhance the performance of 3D human generation. Comprehensive experiments substantiate HumanNorm's ability to generate 3D humans with intricate geometry and realistic appearances. HumanNorm outperforms existing text-to-3D methods in both geometry and texture quality.
</details>

### 3. TeCH: Text-guided Reconstruction of Lifelike Clothed Humans  
Yangyi Huang, Hongwei Yi, Yuliang Xiu, Tingting Liao, Jiaxiang Tang, Deng Cai, Justus Thies

(Mohamed bin Zayed University of Artificial Intelligence, Max Planck Institute for Intelligent Systems, Peking University, State Key Lab of CAD & CG Zhejiang University)
<details span>
<summary><b>Abstract</b></summary>
Despite recent research advancements in reconstructing clothed humans from a single image, accurately restoring the "unseen regions" with high-level details remains an unsolved challenge that lacks attention. Existing methods often generate overly smooth back-side surfaces with a blurry texture. But how to effectively capture all visual attributes of an individual from a single image, which are sufficient to reconstruct unseen areas (e.g., the back view)? Motivated by the power of foundation models, TeCH reconstructs the 3D human by leveraging 1) descriptive text prompts (e.g., garments, colors, hairstyles) which are automatically generated via a garment parsing model and Visual Question Answering (VQA), 2) a personalized fine-tuned Text-to-Image diffusion model (T2I) which learns the "indescribable" appearance. To represent high-resolution 3D clothed humans at an affordable cost, we propose a hybrid 3D representation based on DMTet, which consists of an explicit body shape grid and an implicit distance field. Guided by the descriptive prompts + personalized T2I diffusion model, the geometry and texture of the 3D humans are optimized through multi-view Score Distillation Sampling (SDS) and reconstruction losses based on the original observation. TeCH produces high-fidelity 3D clothed humans with consistent & delicate texture, and detailed full-body geometry. Quantitative and qualitative experiments demonstrate that TeCH outperforms the state-of-the-art methods in terms of reconstruction accuracy and rendering quality. 
</details>

### 4. TADA! Text to Animatable Digital Avatars  
Tingting Liao, Hongwei Yi, Yuliang Xiu, Jiaxaing Tang, Yangyi Huang, Justus Thies, Michael J. Black 

(Mohamed bin Zayed University of Artificial Intelligence, Max Planck Institute for Intelligent Systems, Peking University, State Key Lab of CAD & CG Zhejiang University)
<details span>
<summary><b>Abstract</b></summary>
We introduce TADA, a simple-yet-effective approach that takes textual descriptions and produces expressive 3D avatars with high-quality geometry and lifelike textures, that can be animated and rendered with traditional graphics pipelines. Existing text-based character generation methods are limited in terms of geometry and texture quality, and cannot be realistically animated due to inconsistent alignment between the geometry and the texture, particularly in the face region. To overcome these limitations, TADA leverages the synergy of a 2D diffusion model and an animatable parametric body model. Specifically, we derive an optimizable high-resolution body model from SMPL-X with 3D displacements and a texture map, and use hierarchical rendering with score distillation sampling (SDS) to create high-quality, detailed, holistic 3D avatars from text. To ensure alignment between the geometry and texture, we render normals and RGB images of the generated character and exploit their latent embeddings in the SDS training process. We further introduce various expression parameters to deform the generated character during training, ensuring that the semantics of our generated character remain consistent with the original SMPL-X model, resulting in an animatable character. Comprehensive evaluations demonstrate that TADA significantly surpasses existing approaches on both qualitative and quantitative measures. TADA enables creation of large-scale digital character assets that are ready for animation and rendering, while also being easily editable through natural language. The code will be public for research purposes.
</details>

### 5. DreamWaltz: Make a Scene with Complex 3D Animatable Avatars  
Yukun Huang, Jianan Wang, Ailing Zeng, He Cao, Xianbiao Qi, Yukai Shi, Zheng-Jun Zha, Lei Zhang (USTC, IDEA)
<details span>
<summary><b>Abstract</b></summary>
We present DreamWaltz, a novel framework for generating and animating complex 3D avatars given text guidance and parametric human body prior. While recent methods have shown encouraging results for text-to-3D generation of common objects, creating high-quality and animatable 3D avatars remains challenging. To create high-quality 3D avatars, DreamWaltz proposes 3D-consistent occlusion-aware Score Distillation Sampling (SDS) to optimize implicit neural representations with canonical poses. It provides view-aligned supervision via 3D-aware skeleton conditioning which enables complex avatar generation without artifacts and multiple faces. For animation, our method learns an animatable 3D avatar representation from abundant image priors of diffusion model conditioned on various poses, which could animate complex non-rigged avatars given arbitrary poses without retraining. Extensive evaluations demonstrate that DreamWaltz is an effective and robust approach for creating 3D avatars that can take on complex shapes and appearances as well as novel poses for animation. The proposed framework further enables the creation of complex scenes with diverse compositions, including avatar-avatar, avatar-object and avatar-scene interactions.
</details>

### 6. DreamHuman: Animatable 3D Avatars from Text  
Nikos Kolotouros, Thiemo Alldieck, Andrei Zanfir, Eduard Gabriel Bazavan, Mihai Fieraru, Cristian Sminchisescu (Google Research)
<details span>
<summary><b>Abstract</b></summary>
We present DreamHuman, a method to generate realistic animatable 3D human avatar models solely from textual descriptions. Recent text-to-3D methods have made considerable strides in generation, but are still lacking in important aspects. Control and often spatial resolution remain limited, existing methods produce fixed rather than animated 3D human models, and anthropometric consistency for complex structures like people remains a challenge. DreamHuman connects large text-to-image synthesis models, neural radiance fields, and statistical human body models in a novel modeling and optimization framework. This makes it possible to generate dynamic 3D human avatars with high-quality textures and learned, instance-specific, surface deformations. We demonstrate that our method is capable to generate a wide variety of animatable, realistic 3D human models from text. Our 3D models have diverse appearance, clothing, skin tones and body shapes, and significantly outperform both generic text-to-3D approaches and previous text-based 3D avatar generators in visual fidelity.
</details>

### 7. AvatarCraft: Transforming Text into Neural Human Avatars with Parameterized Shape and Pose Control  
Ruixiang Jiang, Can Wang, Jingbo Zhang, Menglei Chai, Mingming He, Dongdong Chen, Jing Liao

(The Hong Kong Polytechnic University, City University of Hong Kong, Google, Netflix, Microsoft Cloud AI)
<details span>
<summary><b>Abstract</b></summary>
Neural implicit fields are powerful for representing 3D scenes and generating high-quality novel views, but it remains challenging to use such implicit representations for creating a 3D human avatar with a specific identity and artistic style that can be easily animated. Our proposed method, AvatarCraft, addresses this challenge by using diffusion models to guide the learning of geometry and texture for a neural avatar based on a single text prompt. We carefully design the optimization framework of neural implicit fields, including a coarse-to-fine multi-bounding box training strategy, shape regularization, and diffusion-based constraints, to produce high-quality geometry and texture. Additionally, we make the human avatar animatable by deforming the neural implicit field with an explicit warping field that maps the target human mesh to a template human mesh, both represented using parametric human models. This simplifies animation and reshaping of the generated avatar by controlling pose and shape parameters. Extensive experiments on various text descriptions show that AvatarCraft is effective and robust in creating human avatars and rendering novel views, poses, and shapes. 
</details>

### 8. Guide3D: Create 3D Avatars from Text and Image Guidance  
Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, Kwan-Yee K. Wong (HKU, ARC Lab Tencent PCG)
<details span>
<summary><b>Abstract</b></summary>
Recently, text-to-image generation has exhibited remarkable advancements, with the ability to produce visually impressive results. In contrast, text-to-3D generation has not yet reached a comparable level of quality. Existing methods primarily rely on text-guided score distillation sampling (SDS), and they encounter difficulties in transferring 2D attributes of the generated images to 3D content. In this work, we aim to develop an effective 3D generative model capable of synthesizing high-resolution textured meshes by leveraging both textual and image information. To this end, we introduce Guide3D, a zero-shot text-and-image-guided generative model for 3D avatar generation based on diffusion models. Our model involves (1) generating sparse-view images of a text-consistent character using diffusion models, and (2) jointly optimizing multi-resolution differentiable marching tetrahedral grids with pixel-aligned image features. We further propose a similarity-aware feature fusion strategy for efficiently integrating features from different views. Moreover, we introduce two novel training objectives as an alternative to calculating SDS, significantly enhancing the optimization process. We thoroughly evaluate the performance and components of our framework, which outperforms the current state-of-the-art in producing topologically and structurally correct geometry and high-resolution textures. Guide3D enables the direct transfer of 2D-generated images to the 3D space. Our code will be made publicly available.
</details>

### 9. AvatarVerse: High-quality & Stable 3D Avatar Creation from Text and Pose  
Huichao Zhang, Bowen Chen, Hao Yang, Liao Qu, Xu Wang, Li Chen, Chao Long, Feida Zhu, Kang Du, Min Zheng (ByteDance, CMU)
<details span>
<summary><b>Abstract</b></summary>
Creating expressive, diverse and high-quality 3D avatars from highly customized text descriptions and pose guidance is a challenging task, due to the intricacy of modeling and texturing in 3D that ensure details and various styles (realistic, fictional, etc). We present AvatarVerse, a stable pipeline for generating expressive high-quality 3D avatars from nothing but text descriptions and pose guidance. In specific, we introduce a 2D diffusion model conditioned on DensePose signal to establish 3D pose control of avatars through 2D images, which enhances view consistency from partially observed scenarios. It addresses the infamous Janus Problem and significantly stablizes the generation process. Moreover, we propose a progressive high-resolution 3D synthesis strategy, which obtains substantial improvement over the quality of the created 3D avatars. To this end, the proposed AvatarVerse pipeline achieves zero-shot 3D modeling of 3D avatars that are not only more expressive, but also in higher quality and fidelity than previous works. Rigorous qualitative evaluations and user studies showcase AvatarVerse's superiority in synthesizing high-fidelity 3D avatars, leading to a new standard in high-quality and stable 3D avatar creation.
</details>

### 10. AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars 
Fangzhou Hong, Mingyuan Zhang, Liang Pan, Zhongang Cai, Lei Yang, Ziwei Liu 

(S-Lab NTU, SenseTime Research, Shanghai AI Laboratory)
<details span>
<summary><b>Abstract</b></summary>
3D avatar creation plays a crucial role in the digital age. However, the whole production process is prohibitively time-consuming and labor-intensive. To democratize this technology to a larger audience, we propose AvatarCLIP, a zero-shot text-driven framework for 3D avatar generation and animation. Unlike professional software that requires expert knowledge, AvatarCLIP empowers layman users to customize a 3D avatar with the desired shape and texture, and drive the avatar with the described motions using solely natural languages. Our key insight is to take advantage of the powerful vision-language model CLIP for supervising neural human generation, in terms of 3D geometry, texture and animation. Specifically, driven by natural language descriptions, we initialize 3D human geometry generation with a shape VAE network. Based on the generated 3D human shapes, a volume rendering model is utilized to further facilitate geometry sculpting and texture generation. Moreover, by leveraging the priors learned in the motion VAE, a CLIP-guided reference-based motion synthesis method is proposed for the animation of the generated 3D avatar. Extensive qualitative and quantitative experiments validate the effectiveness and generalizability of AvatarCLIP on a wide range of avatars. Remarkably, AvatarCLIP can generate unseen 3D avatars with novel animations, achieving superior zero-shot capability.
</details>

### 11. SEEAvatar: Photorealistic Text-to-3D Avatar Generation with Constrained Geometry and Appearance
Yuanyou Xu, Zongxin Yang, Yi Yang

(ReLER, CCAI, Zhejiang University)
<details span>
<summary><b>Abstract</b></summary>
Powered by large-scale text-to-image generation models, text-to-3D avatar generation has made promising progress. However, most methods fail to produce photorealistic results, limited by imprecise geometry and low-quality appearance. Towards more practical avatar generation, we present SEEAvatar, a method for generating photorealistic 3D avatars from text with SElf-Evolving constraints for decoupled geometry and appearance. For geometry, we propose to constrain the optimized avatar in a decent global shape with a template avatar. The template avatar is initialized with human prior and can be updated by the optimized avatar periodically as an evolving template, which enables more flexible shape generation. Besides, the geometry is also constrained by the static human prior in local parts like face and hands to maintain the delicate structures. For appearance generation, we use diffusion model enhanced by prompt engineering to guide a physically based rendering pipeline to generate realistic textures. The lightness constraint is applied on the albedo texture to suppress incorrect lighting effect. Experiments show that our method outperforms previous methods on both global and local geometry and appearance quality by a large margin. Since our method can produce high-quality meshes and textures, such assets can be directly applied in classic graphics pipeline for realistic rendering under any lighting condition.
</details>

### 12. Make-A-Character: High Quality Text-to-3D Character Generation within Minutes
Jianqiang Ren, Chao He, Lin Liu, Jiahao Chen, Yutong Wang, Yafei Song, Jianfang Li, Tangli Xue, Siqi Hu, Tao Chen, Kunkun Zheng, Jianjing Xiang, Liefeng Bo

(Institute for Intelligent Computing, Alibaba Group)
<details span>
<summary><b>Abstract</b></summary>
There is a growing demand for customized and expressive 3D characters with the emergence of AI agents and Metaverse, but creating 3D characters using traditional computer graphics tools is a complex and time-consuming task. To address these challenges, we propose a user-friendly framework named Make-A-Character (Mach) to create lifelike 3D avatars from text descriptions. The framework leverages the power of large language and vision models for textual intention understanding and intermediate image generation, followed by a series of human-oriented visual perception and 3D generation modules. Our system offers an intuitive approach for users to craft controllable, realistic, fully-realized 3D characters that meet their expectations within 2 minutes, while also enabling easy integration with existing CG pipeline for dynamic expressiveness. 
</details>

### 13. HeadArtist: Text-conditioned 3D Head Generation with Self Score Distillation
Hongyu Liu, Xuan Wang, Ziyu Wan, Yujun Shen, Yibing Song, Jing Liao, Qifeng Chen

(HKUST, Ant Group, City University of HongKong AI Institute, Fudan University)
<details span>
<summary><b>Abstract</b></summary>
This work presents HeadArtist for 3D head generation from text descriptions. With a landmark-guided ControlNet serving as the generative prior, we come up with an efficient pipeline that optimizes a parameterized 3D head model under the supervision of the prior distillation itself. We call such a process self score distillation (SSD). In detail, given a sampled camera pose, we first render an image and its corresponding landmarks from the head model, and add some particular level of noise onto the image. The noisy image, landmarks, and text condition are then fed into the frozen ControlNet twice for noise prediction. Two different classifier-free guidance (CFG) weights are applied during these two predictions, and the prediction difference offers a direction on how the rendered image can better match the text of interest. Experimental results suggest that our approach delivers high-quality 3D head sculptures with adequate geometry and photorealistic appearance, significantly outperforming state-ofthe-art methods. We also show that the same pipeline well supports editing the generated heads, including both geometry deformation and appearance change.
</details>

### 14. HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting
Zhenglin Zhou, Fan Ma, Hehe Fan, Yi Yang (ReLER, CCAI, Zhejiang University)
<details span>
<summary><b>Abstract</b></summary>
Creating digital avatars from textual prompts has long been a desirable yet challenging task. Despite the promising outcomes obtained through 2D diffusion priors in recent works, current methods face challenges in achieving high-quality and animated avatars effectively. In this paper, we present HeadStudio, a novel framework that utilizes 3D Gaussian splatting to generate realistic and animated avatars from text prompts. Our method drives 3D Gaussians semantically to create a flexible and achievable appearance through the intermediate FLAME representation. Specifically, we incorporate the FLAME into both 3D representation and score distillation: 1) FLAME-based 3D Gaussian splatting, driving 3D Gaussian points by rigging each point to a FLAME mesh. 2) FLAME-based score distillation sampling, utilizing FLAME-based fine-grained control signal to guide score distillation from the text prompt. Extensive experiments demonstrate the efficacy of HeadStudio in generating animatable avatars from textual prompts, exhibiting visually appealing appearances. The avatars are capable of rendering high-quality real-time (≥40 fps) novel views at a resolution of 1024. They can be smoothly controlled by real-world speech and video. We hope that HeadStudio can advance digital avatar creation and that the present method can widely be applied across various domains.
</details>

### 15. En3D: An Enhanced Generative Model for Sculpting 3D Humans from 2D Synthetic Data
Yifang Men, Biwen Lei, Yuan Yao, Miaomiao Cui, Zhouhui Lian, Xuansong Xie

(Institute for Intelligent Computing Alibaba Group, Peking University)
<details span>
<summary><b>Abstract</b></summary>
We present En3D, an enhanced generative scheme for sculpting high-quality 3D human avatars. Unlike previous works that rely on scarce 3D datasets or limited 2D collections with imbalanced viewing angles and imprecise pose priors, our approach aims to develop a zero-shot 3D generative scheme capable of producing visually realistic, geometrically accurate and content-wise diverse 3D humans without relying on pre-existing 3D or 2D assets. To address this challenge, we introduce a meticulously crafted workflow that implements accurate physical modeling to learn the enhanced 3D generative model from synthetic 2D data. During inference, we integrate optimization modules to bridge the gap between realistic appearances and coarse 3D shapes. Specifically, En3D comprises three modules: a 3D generator that accurately models generalizable 3D humans with realistic appearance from synthesized balanced, diverse, and structured human images; a geometry sculptor that enhances shape quality using multi-view normal constraints for intricate human anatomy; and a texturing module that disentangles explicit texture maps with fidelity and editability, leveraging semantical UV partitioning and a differentiable rasterizer. Experimental results show that our approach significantly outperforms prior works in terms of image quality, geometry accuracy and content diversity. We also showcase the applicability of our generated avatars for animation and editing, as well as the scalability of our approach for content-style free adaptation.
</details>

### 16. DivAvatar: Diverse 3D Avatar Generation with a Single Prompt
Weijing Tao, Biwen Lei, Kunhao Liu, Shijian Lu, Miaomiao Cui, Xuansong Xie, Chunyan Miao

(Nanyang Technological University, Alibaba Group)
<details span>
<summary><b>Abstract</b></summary>
Text-to-Avatar generation has recently made significant strides due to advancements in diffusion models. However, most existing work remains constrained by limited diversity, producing avatars with subtle differences in appearance for a given text prompt. We design DivAvatar, a novel framework that generates diverse avatars, empowering 3D creatives with a multitude of distinct and richly varied 3D avatars from a single text prompt. Different from most existing work that exploits scene-specific 3D representations such as NeRF, DivAvatar finetunes a 3D generative model (i.e., EVA3D), allowing diverse avatar generation from simply noise sampling in inference time. DivAvatar has two key designs that help achieve generation diversity and visual quality. The first is a noise sampling technique during training phase which is critical in generating diverse appearances. The second is a semantic-aware zoom mechanism and a novel depth loss, the former producing appearances of high textual fidelity by separate fine-tuning of specific body parts and the latter improving geometry quality greatly by smoothing the generated mesh in the features space. Extensive experiments show that DivAvatar is highly versatile in generating avatars of diverse appearances.
</details>

### 17. InstructHumans: Editing Animated 3D Human Textures with Instructions (text to 3d human texture editing)
Jiayin Zhu, Linlin Yang, Angela Yao

(National University of Singapore, Communication University of China)
<details span>
<summary><b>Abstract</b></summary>
We present InstructHumans, a novel framework for instruction-driven 3D human texture editing. Existing text-based editing methods use Score Distillation Sampling (SDS) to distill guidance from generative models. This work shows that naively using such scores is harmful to editing as they destroy consistency with the source avatar. Instead, we propose an alternate SDS for Editing (SDS-E) that selectively incorporates subterms of SDS across diffusion timesteps. We further enhance SDS-E with spatial smoothness regularization and gradient-based viewpoint sampling to achieve high-quality edits with sharp and high-fidelity detailing. InstructHumans significantly outperforms existing 3D editing methods, consistent with the initial avatar while faithful to the textual instructions.
</details>

### 18. MagicMirror: Fast and High-Quality Avatar Generation with a Constrained Search Space
Armand Comas-Massagué, Di Qiu, Menglei Chai, Marcel Bühler, Amit Raj, Ruiqi Gao, Qiangeng Xu, Mark Matthews, Paulo Gotardo, Octavia Camps, Sergio Orts-Escolano, Thabo Beeler

(Google, Northeastern Univeristy, ETH Zurich, Google DeepMind)
<details span>
<summary><b>Abstract</b></summary>
We introduce a novel framework for 3D human avatar generation and personalization, leveraging text prompts to enhance user engagement and customization. Central to our approach are key innovations aimed at overcoming the challenges in photo-realistic avatar synthesis. Firstly, we utilize a conditional Neural Radiance Fields (NeRF) model, trained on a large-scale unannotated multi-view dataset, to create a versatile initial solution space that accelerates and diversifies avatar generation. Secondly, we develop a geometric prior, leveraging the capabilities of Text-to-Image Diffusion Models, to ensure superior view invariance and enable direct optimization of avatar geometry. These foundational ideas are complemented by our optimization pipeline built on Variational Score Distillation (VSD), which mitigates texture loss and over-saturation issues. As supported by our extensive experiments, these strategies collectively enable the creation of custom avatars with unparalleled visual quality and better adherence to input text prompts. 
</details>

### Text to 3D Human Paper lists
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting**  | CVPR 2024  |          [Link](https://arxiv.org/abs/2311.17061)          | [Link](https://github.com/alvinliu0/HumanGaussian)  | [Link](https://alvinliu0.github.io/projects/HumanGaussian)  | 
| 2023 | **HumanNorm: Learning Normal Diffusion Model for High-quality and Realistic 3D Human Generation**  | CVPR 2024  |          [Link](https://arxiv.org/abs/2310.01406)          | [Link](https://github.com/xhuangcv/humannorm)  | [Link](https://humannorm.github.io/)  |
| 2023 | **TeCH: Text-guided Reconstruction of Lifelike Clothed Humans**  | 3DV 2024  |          [Link](https://arxiv.org/abs/2308.08545)          | [Link](https://github.com/huangyangyi/TeCH)  | [Link](https://huangyangyi.github.io/TeCH/)  |
| 2023 | **TADA! Text to Animatable Digital Avatars**  | 3DV 2024  |          [Link](https://arxiv.org/abs/2308.10899)          | [Link](https://github.com/TingtingLiao/TADA)  | [Link](https://tada.is.tue.mpg.de/)  |
| 2023 | **DreamWaltz: Make a Scene with Complex 3D Animatable Avatars**  | NeurIPS 2023  |          [Link](https://arxiv.org/abs/2305.12529)          | [Link](https://github.com/IDEA-Research/DreamWaltz)  | [Link](https://idea-research.github.io/DreamWaltz/)  |
| 2023 | **DreamHuman: Animatable 3D Avatars from Text**  | NeurIPS 2023 (Spotlight)  |          [Link](https://arxiv.org/abs/2306.09329)          |  --  | [Link](https://dream-human.github.io/)  |
| 2023 | **AvatarCraft: Transforming Text into Neural Human Avatars with Parameterized Shape and Pose Control**  | ICCV 2023 |          [Link](https://arxiv.org/abs/2303.17606)          |  [Link](https://github.com/songrise/avatarcraft)   | [Link](https://avatar-craft.github.io/)  |
| 2023 | **Guide3D: Create 3D Avatars from Text and Image Guidance**  | arXiv  |          [Link](https://arxiv.org/abs/2308.09705)          |  [Link](https://github.com/yukangcao/Guide3D) | -- |
| 2023 | **AvatarVerse: High-quality & Stable 3D Avatar Creation from Text and Pose**  | AAAI2024  |          [Link](https://arxiv.org/abs/2308.03610)          |  [Link](https://github.com/bytedance/AvatarVerse)  | [Link](https://avatarverse3d.github.io/)  |
| 2022 | **AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars**  | SIGGRAPH 2022 (Journal Track)  |          [Link](https://arxiv.org/abs/2205.08535)          | [Link](https://github.com/hongfz16/AvatarCLIP)  | [Link](https://hongfz16.github.io/projects/AvatarCLIP.html)  |
| 2023 | **SEEAvatar: Photorealistic Text-to-3D Avatar Generation with Constrained Geometry and Appearance**  | arXiv  |          [Link](https://arxiv.org/abs/2312.08889)          | Coming Soon!  | [Link](https://seeavatar3d.github.io/)  |
| 2023 | **Make-A-Character: High Quality Text-to-3D Character Generation within Minutes**  | arXiv  |          [Link](https://arxiv.org/abs/2312.15430)          |  [Link](https://github.com/Human3DAIGC/Make-A-Character)  | [Link](https://human3daigc.github.io/MACH/)  |
| 2023 | **HeadArtist: Text-conditioned 3D Head Generation with Self Score Distillation**  | arXiv  |          [Link](https://arxiv.org/abs/2312.07539)          |  [Link](https://github.com/KumapowerLIU/HeadArtist)  | [Link](https://kumapowerliu.github.io/HeadArtist/)  |
| 2024 | **HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting**  | arXiv 2024  |          [Link](https://arxiv.org/abs/2402.06149)          |  [Link](https://github.com/ZhenglinZhou/HeadStudio/)  | [Link](https://zhenglinzhou.github.io/HeadStudio-ProjectPage/)  |
| 2024 | **En3D: An Enhanced Generative Model for Sculpting 3D Humans from 2D Synthetic Data**  | arXiv 2024  |          [Link](https://arxiv.org/abs/2401.01173)          |  [Link](https://github.com/menyifang/En3D)  | [Link](https://menyifang.github.io/projects/En3D/index.html)  |
| 2024 | **DivAvatar: Diverse 3D Avatar Generation with a Single Prompt**  | arXiv 2024  |          [Link](https://arxiv.org/abs/2402.17292)          | --  | --  |
| 2024 | **InstructHumans: Editing Animated 3D Human Textures with Instructions**  | arXiv 2024  |          [Link](https://arxiv.org/abs/2404.04037)          | [Link](https://github.com/viridityzhu/InstructHumans)  | [Link](https://jyzhu.top/instruct-humans/)  |
| 2024 | **MagicMirror: Fast and High-Quality Avatar Generation with a Constrained Search Space**  | arXiv 2024  |          [Link](https://arxiv.org/abs/2404.01296)      | -- | [Link](https://syntec-research.github.io/MagicMirror/)  |

### Text to 3D Human Reference
<details close>
<summary>Text to 3D Human</summary>

```
% text to 3d human

@article{liu2023humangaussian,
    title={HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting},
    author={Liu, Xian and Zhan, Xiaohang and Tang, Jiaxiang and Shan, Ying and Zeng, Gang and Lin, Dahua and Liu, Xihui and Liu, Ziwei},
    journal={arXiv preprint arXiv:2311.17061},
    year={2023}
}

@article{humannorm2023,
title={HumanNorm: Learning Normal Diffusion Model for High-quality and Realistic 3D Human Generation},
author={Huang, Xin and Shao, Ruizhi and Zhang, Qi and Zhang, Hongwen and Feng, Ying and Liu, Yebin and Wang, Qing},
journal={arXiv},
year={2023}
}

@inproceedings{huang2024tech,
  title={{TeCH: Text-guided Reconstruction of Lifelike Clothed Humans}},
  author={Huang, Yangyi and Yi, Hongwei and Xiu, Yuliang and Liao, Tingting and Tang, Jiaxiang and Cai, Deng and Thies, Justus},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2024}
}

@inproceedings{liao2024tada,
  title={{TADA! Text to Animatable Digital Avatars}},
  author={Liao, Tingting and Yi, Hongwei and Xiu, Yuliang and Tang, Jiaxiang and Huang, Yangyi and Thies, Justus and Black, Michael J.},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2024}
}

@article{huang2023dreamwaltz,
 title={DreamWaltz: Make a Scene with Complex 3D Animatable Avatars},
 author={Yukun Huang and Jianan Wang and Ailing Zeng and He Cao and Xianbiao Qi and Yukai Shi and Zheng-Jun Zha and Lei Zhang},
 year={2023},
 eprint={2305.12529},
 archivePrefix={arXiv},
 primaryClass={cs.CV}
}

@article{kolotouros2023dreamhuman,
  title={DreamHuman: Animatable 3D Avatars from Text},
  author={Kolotouros, Nikos and Alldieck, Thiemo and Zanfir, Andrei and Bazavan, Eduard Gabriel and Fieraru, Mihai and Sminchisescu, Cristian},
  booktitle={arXiv preprint arxiv:2306.09329},
  year={2023}
}

@article{jiang2023avatarcraft,
  title={AvatarCraft: Transforming Text into Neural Human Avatars with Parameterized Shape and Pose Control},
  author={Jiang, Ruixiang and Wang, Can and Zhang, Jingbo and Chai, Menglei and He, Mingming and Chen, Dongdong and Liao, Jing},
  journal={arXiv preprint arXiv:2303.17606},
  year={2023}
}

@article{cao2023guide3d,
  title={Guide3D: Create 3D Avatars from Text and Image Guidance},
  author={Cao, Yukang and Cao, Yan-Pei and Han, Kai and Shan, Ying and Wong, Kwan-Yee K},
  journal={arXiv preprint arXiv:2308.09705},
  year={2023}
}

@misc{zhang2023avatarverse,
  title={AvatarVerse: High-quality & Stable 3D Avatar Creation from Text and Pose},
  author={Huichao Zhang and Bowen Chen and Hao Yang and Liao Qu and Xu Wang and Li Chen and Chao Long and Feida Zhu and Kang Du and Min Zheng},
  year={2023},
  eprint={2308.03610},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

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

@article{xu2023seeavatar,
  title={SEEAvatar: Photorealistic Text-to-3D Avatar Generation with Constrained Geometry and Appearance},
  author={Xu, Yuanyou and Yang, Zongxin and Yang, Yi},
  journal={arXiv preprint arXiv:2312.08889},
  year={2023}
}

@article{ren2023makeacharacter,
      title={Make-A-Character: High Quality Text-to-3D Character Generation within Minutes},
      author={Jianqiang Ren and Chao He and Lin Liu and Jiahao Chen and Yutong Wang and Yafei Song and Jianfang Li and Tangli Xue and Siqi Hu and Tao Chen and Kunkun Zheng and Jianjing Xiang and Liefeng Bo},
      year={2023},
      journal = {arXiv preprint arXiv:2312.15430}
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

@inproceedings{men2024en3d,
  title={En3D: An Enhanced Generative Model for Sculpting 3D Humans from 2D Synthetic Data},
  author={Men, Yifang and Lei, Biwen and Yao, Yuan and Cui, Miaomiao and Lian, Zhouhui and Xie, Xuansong},
  journal={arXiv preprint arXiv:2401.01173},
  website={https://menyifang.github.io/projects/En3D/index.html},
  year={2024}
}

@article{tao2024divavatar,
  title={DivAvatar: Diverse 3D Avatar Generation with a Single Prompt},
  author={Tao, Weijing and Lei, Biwen and Liu, Kunhao and Lu, Shijian and Cui, Miaomiao and Xie, Xuansong and Miao, Chunyan},
  journal={arXiv preprint arXiv:2402.17292},
  year={2024}
}

@article{zhu2024InstructHumans,
         author={Zhu, Jiayin and Yang, Linlin and Yao, Angela},
         title={InstructHumans: Editing Animated 3D Human Textures with Instructions},
         journal={arXiv preprint arXiv:2404.04037},
         year={2024}
}

@article{comas2024magicmirror,
  title={MagicMirror: Fast and High-Quality Avatar Generation with a Constrained Search Space},
  author={Comas-Massagu{\'e}, Armand and Qiu, Di and Chai, Menglei and B{\"u}hler, Marcel and Raj, Amit and Gao, Ruiqi and Xu, Qiangeng and Matthews, Mark and Gotardo, Paulo and Camps, Octavia and others},
  journal={arXiv preprint arXiv:2404.01296},
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

--------------

## Text to Human Motion
### 1. Synthesizing Moving People with 3D Control  
Boyi Li, Jathushan Rajasegaran, Yossi Gandelsman, Alexei A. Efros, Jitendra Malik (UC Berkeley)
<details span>
<summary><b>Abstract</b></summary>
In this paper, we present a diffusion model-based framework for animating people from a single image for a given target 3D motion sequence. Our approach has two core components: a) learning priors about invisible parts of the human body and clothing, and b) rendering novel body poses with proper clothing and texture. For the first part, we learn an in-filling diffusion model to hallucinate unseen parts of a person given a single image. We train this model on texture map space, which makes it more sample-efficient since it is invariant to pose and viewpoint. Second, we develop a diffusion-based rendering pipeline, which is controlled by 3D human poses. This produces realistic renderings of novel poses of the person, including clothing, hair, and plausible in-filling of unseen regions. This disentangled approach allows our method to generate a sequence of images that are faithful to the target motion in the 3D pose and, to the input image in terms of visual similarity. In addition to that, the 3D control allows various synthetic camera trajectories to render a person. Our experiments show that our method is resilient in generating prolonged motions and varied challenging and complex poses compared to prior methods. 
</details>

### 2. Multi-Track Timeline Control for Text-Driven 3D Human Motion Generation  
Mathis Petrovich, Or Litany, Umar Iqbal, Michael J. Black, Gül Varol, Xue Bin Peng, Davis Rempe

(LIGM École des Ponts Univ Gustave Eiffel CNRS, Max Planck Institute for Intelligent Systems, NVIDIA, Technion, Simon Fraser University)
<details span>
<summary><b>Abstract</b></summary>
Recent advances in generative modeling have led to promising progress on synthesizing 3D human motion from text, with methods that can generate character animations from short prompts and specified durations. However, using a single text prompt as input lacks the fine-grained control needed by animators, such as composing multiple actions and defining precise durations for parts of the motion. To address this, we introduce the new problem of timeline control for text-driven motion synthesis, which provides an intuitive, yet fine-grained, input interface for users. Instead of a single prompt, users can specify a multi-track timeline of multiple prompts organized in temporal intervals that may overlap. This enables specifying the exact timings of each action and composing multiple actions in sequence or at overlapping intervals. To generate composite animations from a multi-track timeline, we propose a new test-time denoising method. This method can be integrated with any pre-trained motion diffusion model to synthesize realistic motions that accurately reflect the timeline. At every step of denoising, our method processes each timeline interval (text prompt) individually, subsequently aggregating the predictions with consideration for the specific body parts engaged in each action. Experimental comparisons and ablations validate that our method produces realistic motions that respect the semantics and timing of given text prompts. 
</details>

### 3. FlowMDM: Seamless Human Motion Composition with Blended Positional Encodings
German Barquero, Sergio Escalera, Cristina Palmero
(University of Barcelona and Computer Vision Center, Spain)
<details span>
<summary><b>Abstract</b></summary>
Conditional human motion generation is an important topic with many applications in virtual reality, gaming, and robotics. While prior works have focused on generating motion guided by text, music, or scenes, these typically result in isolated motions confined to short durations. Instead, we address the generation of long, continuous sequences guided by a series of varying textual descriptions. In this context, we introduce FlowMDM, the first diffusion-based model that generates seamless Human Motion Compositions (HMC) without any postprocessing or redundant denoising steps. For this, we introduce the Blended Positional Encodings, a technique that leverages both absolute and relative positional encodings in the denoising chain. More specifically, global motion coherence is recovered at the absolute stage, whereas smooth and realistic transitions are built at the relative stage. As a result, we achieve state-of-the-art results in terms of accuracy, realism, and smoothness on the Babel and HumanML3D datasets. FlowMDM excels when trained with only a single description per motion sequence thanks to its Pose-Centric Cross-ATtention, which makes it robust against varying text descriptions at inference time. Finally, to address the limitations of existing HMC metrics, we propose two new metrics: the Peak Jerk and the Area Under the Jerk, to detect abrupt transitions.
</details>

### 4. EMDM: Efficient Motion Diffusion Model for Fast, High-Quality Human Motion Generation  
Wenyang Zhou, Zhiyang Dou, Zeyu Cao, Zhouyingcheng Liao, Jingbo Wang, Wenjia Wang, Yuan Liu, Taku Komura, Wenping Wang, Lingjie Liu
(University of Cambridge, University of Hong Kong, Shanghai AI Laboratory, Texas A&M University, University of Pennsylvania)
<details span>
<summary><b>Abstract</b></summary>
We introduce Efficient Motion Diffusion Model (EMDM) for fast and high-quality human motion generation. Although previous motion diffusion models have shown impressive results, they struggle to achieve fast generation while maintaining high-quality human motions. Motion latent diffusion has been proposed for efficient motion generation. However, effectively learning a latent space can be non-trivial in such a two-stage manner. Meanwhile, accelerating motion sampling by increasing the step size, e.g., DDIM, typically leads to a decline in motion quality due to the inapproximation of complex data distributions when naively increasing the step size. In this paper, we propose EMDM that allows for much fewer sample steps for fast motion generation by modeling the complex denoising distribution during multiple sampling steps. Specifically, we develop a Conditional Denoising Diffusion GAN to capture multimodal data distributions conditioned on both control signals, i.e., textual description and denoising time step. By modeling the complex data distribution, a larger sampling step size and fewer steps are achieved during motion synthesis, significantly accelerating the generation process. To effectively capture the human dynamics and reduce undesired artifacts, we employ motion geometric loss during network training, which improves the motion quality and training efficiency. As a result, EMDM achieves a remarkable speed-up at the generation stage while maintaining high-quality motion generation in terms of fidelity and diversity.
</details>

### 5. SinMDM: Single Motion Diffusion 
Sigal Raab, Inbal Leibovitch, Guy Tevet, Moab Arar, Amit H. Bermano, Daniel Cohen-Or
(Tel Aviv University, Israel)
<details span>
<summary><b>Abstract</b></summary>
Synthesizing realistic animations of humans, animals, and even imaginary creatures, has long been a goal for artists and computer graphics professionals. Compared to the imaging domain, which is rich with large available datasets, the number of data instances for the motion domain is limited, particularly for the animation of animals and exotic creatures (e.g., dragons), which have unique skeletons and motion patterns. In this work, we present a Single Motion Diffusion Model, dubbed SinMDM, a model designed to learn the internal motifs of a single motion sequence with arbitrary topology and synthesize motions of arbitrary length that are faithful to them. We harness the power of diffusion models and present a denoising network explicitly designed for the task of learning from a single input motion. SinMDM is designed to be a lightweight architecture, which avoids overfitting by using a shallow network with local attention layers that narrow the receptive field and encourage motion diversity. SinMDM can be applied in various contexts, including spatial and temporal in-betweening, motion expansion, style transfer, and crowd animation. Our results show that SinMDM outperforms existing methods both in quality and time-space efficiency. Moreover, while current approaches require additional training for different applications, our work facilitates these applications at inference time.
</details>

### 6. MDM: Human Motion Diffusion Model 
Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, Amit H. Bermano (Tel Aviv University, Israel)
<details span>
<summary><b>Abstract</b></summary>
Natural and expressive human motion generation is the holy grail of computer animation. It is a challenging task, due to the diversity of possible motion, human perceptual sensitivity to it, and the difficulty of accurately describing it. Therefore, current generative solutions are either low-quality or limited in expressiveness. Diffusion models, which have already shown remarkable generative capabilities in other domains, are promising candidates for human motion due to their many-to-many nature, but they tend to be resource hungry and hard to control. In this paper, we introduce Motion Diffusion Model (MDM), a carefully adapted classifier-free diffusion-based generative model for the human motion domain. MDM is transformer-based, combining insights from motion generation literature. A notable design-choice is the prediction of the sample, rather than the noise, in each diffusion step. This facilitates the use of established geometric losses on the locations and velocities of the motion, such as the foot contact loss. As we demonstrate, MDM is a generic approach, enabling different modes of conditioning, and different generation tasks. We show that our model is trained with lightweight resources and yet achieves state-of-the-art results on leading benchmarks for text-to-motion and action-to-motion.
</details>

### 7. MLD: Motion Latent Diffusion Models 
Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, Jingyi Yu, Gang Yu

(Fudan University, Tencent PCG, ShanghaiTech University)
<details span>
<summary><b>Abstract</b></summary>
We study a challenging task, conditional human motion generation, which produces plausible human motion sequences according to various conditional inputs, such as action classes or textual descriptors. Since human motions are highly diverse and have a property of quite different distribution from conditional modalities, such as textual descriptors in natural languages, it is hard to learn a probabilistic mapping from the desired conditional modality to the human motion sequences. Besides, the raw motion data from the motion capture system might be redundant in sequences and contain noises; directly modeling the joint distribution over the raw motion sequences and conditional modalities would need a heavy computational overhead and might result in artifacts introduced by the captured noises. To learn a better representation of the various human motion sequences, we first design a powerful Variational AutoEncoder (VAE) and arrive at a representative and low-dimensional latent code for a human motion sequence. Then, instead of using a diffusion model to establish the connections between the raw motion sequences and the conditional inputs, we perform a diffusion process on the motion latent space. Our proposed Motion Latent-based Diffusion model (MLD) could produce vivid motion sequences conforming to the given conditional inputs and substantially reduce the computational overhead in both the training and inference stages. Extensive experiments on various human motion generation tasks demonstrate that our MLD achieves significant improvements over the state-of-the-art methods among extensive human motion generation tasks, with two orders of magnitude faster than previous diffusion models on raw motion sequences.
</details>

### 8. MotionMix: Weakly-Supervised Diffusion for Controllable Motion Generation
Nhat M. Hoang, Kehong Gong, Chuan Guo, Michael Bi Mi (Nanyang Technological University, Huawei Technologies Co., Ltd)
<details span>
<summary><b>Abstract</b></summary>
Controllable generation of 3D human motions becomes an important topic as the world embraces digital transformation. Existing works, though making promising progress with the advent of diffusion models, heavily rely on meticulously captured and annotated (e.g., text) high-quality motion corpus, a resource-intensive endeavor in the real world. This motivates our proposed MotionMix, a simple yet effective weakly-supervised diffusion model that leverages both noisy and unannotated motion sequences. Specifically, we separate the denoising objectives of a diffusion model into two stages: obtaining conditional rough motion approximations in the initial T−T∗ steps by learning the noisy annotated motions, followed by the unconditional refinement of these preliminary motions during the last T∗ steps using unannotated motions. Notably, though learning from two sources of imperfect data, our model does not compromise motion generation quality compared to fully supervised approaches that access gold data. Extensive experiments on several benchmarks demonstrate that our MotionMix, as a versatile framework, consistently achieves state-of-the-art performances on text-to-motion, action-to-motion, and music-to-dance tasks. 
</details>

### 9. HumanTOMATO: Text-aligned Whole-body Motion Generation 
Shunlin Lu*, Ling-Hao Chen*, Ailing Zeng, Jing Lin, Ruimao Zhang, Lei Zhang, Heung-Yeung Shum

(Tsinghua University, International Digital Economy Academy (IDEA), School of Data Science CUHK (SZ))
<details span>
<summary><b>Abstract</b></summary>
This work targets a novel text-driven whole-body motion generation task, which takes a given textual description as input and aims at generating high-quality, diverse, and coherent facial expressions, hand gestures, and body motions simultaneously. Previous works on text-driven motion generation tasks mainly have two limitations: they ignore the key role of fine-grained hand and face controlling in vivid whole-body motion generation, and lack a good alignment between text and motion. To address such limitations, we propose a Text-aligned whOle-body Motion generATiOn framework, named HumanTOMATO, which is the first attempt to our knowledge towards applicable holistic motion generation in this research area. To tackle this challenging task, our solution includes two key designs: (1) a Holistic Hierarchical VQ-VAE (aka H²VQ) and a Hierarchical-GPT for fine-grained body and hand motion reconstruction and generation with two structured codebooks; and (2) a pre-trained text-motion-alignment model to help generated motion align with the input textual description explicitly. Comprehensive experiments verify that our model has significant advantages in both the quality of generated motions and their alignment with text.
</details>

### 10. MotionGPT: Human Motion as a Foreign Language  
Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, Tao Chen

(Fudan University, Tencent PCG, ShanghaiTech University)
<details span>
<summary><b>Abstract</b></summary>
Though the advancement of pre-trained large language models unfolds, the exploration of building a unified model for language and other multimodal data, such as motion, remains challenging and untouched so far. Fortunately, human motion displays a semantic coupling akin to human language, often perceived as a form of body language. By fusing language data with large-scale motion models, motion-language pre-training that can enhance the performance of motion-related tasks becomes feasible. Driven by this insight, we propose MotionGPT, a unified, versatile, and user-friendly motion-language model to handle multiple motion-relevant tasks. Specifically, we employ the discrete vector quantization for human motionand transfer 3D motion into motion tokens, similar to the generation process ofword tokens. Building upon this “motion vocabulary”, we perform language modeling on both motion and text in a unified manner, treating human motion as a specific language. Moreover, inspired by prompt learning, we pre-train MotionGPT with a mixture of motion-language data and fine-tune it on prompt-based question-and-answer tasks. Extensive experiments demonstrate that MotionGPT achieves state-of-the-art performances on multiple motion tasks including text-driven motion generation, motion captioning, motion prediction, and motion in-between.
</details>

### 11. Story-to-Motion: Synthesizing Infinite and Controllable Character Animation from Long Text  
Zhongfei Qing, Zhongang Cai, Zhitao Yang, Lei Yang (SenseTime)
<details span>
<summary><b>Abstract</b></summary>
Generating natural human motion from a story has the potential to transform the landscape of animation, gaming, and film industries. A new and challenging task, Story-to-Motion, arises when characters are required to move to various locations and perform specific motions based on a long text description. This task demands a fusion of low-level control (trajectories) and high-level control (motion semantics). Previous works in character control and text-to-motion have addressed related aspects, yet a comprehensive solution remains elusive: character control methods do not handle text description, whereas text-to-motion methods lack position constraints and often produce unstable motions. In light of these limitations, we propose a novel system that generates controllable, infinitely long motions and trajectories aligned with the input text. 1) we leverage contemporary Large Language Models to act as a text-driven motion scheduler to extract a series of (text, position) pairs from long text. 2) we develop a text-driven motion retrieval scheme that incorporates classic motion matching with motion semantic and trajectory constraints. 3) we design a progressive mask transformer that addresses common artifacts in the transition motion such as unnatural pose and foot sliding. Beyond its pioneering role as the first comprehensive solution for Story-to-Motion, our system undergoes evaluation across three distinct sub-tasks: trajectory following, temporal action composition, and motion blending, where it outperforms previous state-of-the-art (SOTA) motion synthesis methods across the board.
</details>

### 12. Plan, Posture and Go: Towards Open-World Text-to-Motion Generation  
Jinpeng Liu, Wenxun Dai, Chunyu Wang, Yiji Cheng, Yansong Tang, Xin Tong 
(Shenzhen International Graudate School Tsinghua University, Microsoft Research Asia)
<details span>
<summary><b>Abstract</b></summary>
Conventional text-to-motion generation methods are usually trained on limited text-motion pairs, making them hard to generalize to open-world scenarios. Some works use the CLIP model to align the motion space and the text space, aiming to enable motion generation from natural language motion descriptions. However, they are still constrained to generate limited and unrealistic in-place motions. To address these issues, we present a divide-and-conquer framework named PRO-Motion, which consists of three modules as motion planner, posture-diffuser and go-diffuser. The motion planner instructs Large Language Models (LLMs) to generate a sequence of scripts describing the key postures in the target motion. Differing from natural languages, the scripts can describe all possible postures following very simple text templates. This significantly reduces the complexity of posture-diffuser, which transforms a script to a posture, paving the way for open-world generation. Finally, go-diffuser, implemented as another diffusion model, estimates whole-body translations and rotations for all postures, resulting in realistic motions. Experimental results have shown the superiority of our method with other counterparts, and demonstrated its capability of generating diverse and realistic motions from complex open-world prompts such as "Experiencing a profound sense of joy". 
</details>

### 13. MoMask: Generative Masked Modeling of 3D Human Motions  
Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, Li Cheng 
(University of Alberta, Canada)
<details span>
<summary><b>Abstract</b></summary>
We introduce MoMask, a novel masked modeling framework for text-driven 3D human motion generation. In MoMask, a hierarchical quantization scheme is employed to represent human motion as multi-layer discrete motion tokens with high-fidelity details. Starting at the base layer, with a sequence of motion tokens obtained by vector quantization, the residual tokens of increasing orders are derived and stored at the subsequent layers of the hierarchy. This is consequently followed by two distinct bidirectional transformers. For the base-layer motion tokens, a Masked Transformer is designated to predict randomly masked motion tokens conditioned on text input at training stage. During generation (i.e. inference) stage, starting from an empty sequence, our Masked Transformer iteratively fills up the missing tokens; Subsequently, a Residual Transformer learns to progressively predict the next-layer tokens based on the results from current layer. Extensive experiments demonstrate that MoMask outperforms the state-of-art methods on the text-to-motion generation task, with an FID of 0.045 (vs e.g. 0.141 of T2M-GPT) on the HumanML3D dataset, and 0.228 (vs 0.514) on KIT-ML, respectively. MoMask can also be seamlessly applied in related tasks without further model fine-tuning, such as text-guided temporal inpainting.
</details>

### 14. Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer
Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, Tali Dekel
(Weizmann Institute of Science)
<details span>
<summary><b>Abstract</b></summary>
We present a new method for text-driven motion transfer - synthesizing a video that complies with an input text prompt describing the target objects and scene while maintaining an input video's motion and scene layout. Prior methods are confined to transferring motion across two subjects within the same or closely related object categories and are applicable for limited domains (e.g., humans). In this work, we consider a significantly more challenging setting in which the target and source objects differ drastically in shape and fine-grained motion characteristics (e.g., translating a jumping dog into a dolphin). To this end, we leverage a pre-trained and fixed text-to-video diffusion model, which provides us with generative and motion priors. The pillar of our method is a new space-time feature loss derived directly from the model. This loss guides the generation process to preserve the overall motion of the input video while complying with the target object in terms of shape and fine-grained motion traits.
</details>

### 15. Self-Correcting Self-Consuming Loops For Generative Model Training
Nate Gillman, Michael Freeman, Daksh Aggarwal, Chia-Hong Hsu, Calvin Luo, Yonglong Tian, Chen Sun (Brown University, Google Research)
<details span>
<summary><b>Abstract</b></summary>
What happens after iteratively training a text-conditioned generative model for human motion synthesis for 50 generations? We simulate a self-consuming loop by creating synthetic data with the latest generative model, and mixing them with the original data to continue training the next generative model. We observe that by self-correcting the synthetic data with a physics simulator, the model can successfully avoid collapse and generate high-quality human motion. Our paper provides theoretical and empirical justification for the self-correcting self-consuming loop.
</details>

### 16. Large Motion Model for Unified Multi-Modal Motion Generation
Mingyuan Zhang, Daisheng Jin, Chenyang Gu, Fangzhou Hong, Zhongang Cai, Jingfang Huang, Chongzhi Zhang, Xinying Guo, Lei Yang, Ying He, Ziwei Liu 

(S-Lab, Nanyang Technological University, SenseTime China)
<details span>
<summary><b>Abstract</b></summary>
Human motion generation, a cornerstone technique in animation and video production, has widespread applications in various tasks like text-to-motion and music-to-dance. Previous works focus on developing specialist models tailored for each task without scalability. In this work, we present Large Motion Model (LMM), a motion-centric, multi-modal framework that unifies mainstream motion generation tasks into a generalist model. A unified motion model is appealing since it can leverage a wide range of motion data to achieve broad generalization beyond a single task. However, it is also challenging due to the heterogeneous nature of substantially different motion data and tasks. LMM tackles these challenges from three principled aspects: 1) Data: We consolidate datasets with different modalities, formats and tasks into a comprehensive yet unified motion generation dataset, MotionVerse, comprising 10 tasks, 16 datasets, a total of 320k sequences, and 100 million frames. 2) Architecture: We design an articulated attention mechanism ArtAttention that incorporates body part-aware modeling into Diffusion Transformer backbone. 3) Pre-Training: We propose a novel pre-training strategy for LMM, which employs variable frame rates and masking forms, to better exploit knowledge from diverse training data. Extensive experiments demonstrate that our generalist LMM achieves competitive performance across various standard motion generation tasks over state-of-the-art specialist models. Notably, LMM exhibits strong generalization capabilities and emerging properties across many unseen tasks. Additionally, our ablation studies reveal valuable insights about training and scaling up large motion models for future research.
</details>

### 17. Generating Human Interaction Motions in Scenes with Text Control
Hongwei Yi, Justus Thies, Michael J. Black, Xue Bin Peng, Davis Rempe

(NVIDIA, Max Planck Institute for Intelligent Systems Tübingen Germany, Technical University of Darmstad, Simon Fraser University)
<details span>
<summary><b>Abstract</b></summary>
We present TeSMo, a method for text-controlled scene-aware motion generation based on denoising diffusion models. Previous text-to-motion methods focus on characters in isolation without considering scenes due to the limited availability of datasets that include motion, text descriptions, and interactive scenes. Our approach begins with pre-training a scene-agnostic text-to-motion diffusion model, emphasizing goal-reaching constraints on large-scale motion-capture datasets. We then enhance this model with a scene-aware component, fine-tuned using data augmented with detailed scene information, including ground plane and object shapes. To facilitate training, we embed annotated navigation and interaction motions within scenes. The proposed method produces realistic and diverse human-object interactions, such as navigation and sitting, in different scenes with various object shapes, orientations, initial body positions, and poses. Extensive experiments demonstrate that our approach surpasses prior techniques in terms of the plausibility of human-scene interactions, as well as the realism and variety of the generated motions. 
</details>

### 18. Move as You Say, Interact as You Can: Language-guided Human Motion Generation with Scene Affordance
Zan Wang, Yixin Chen, Baoxiong Jia, Puhao Li, Jinlu Zhang, Jingze Zhang, Tengyu Liu, Yixin Zhu, Wei Liang, Siyuan Huang

(School of Computer Science & Technology, Beijing Institute of Technology,National Key Laboratory of General Artificial Intelligence, BIGAI, Dept. of Automation, Tsinghua University, CFCS, School of Computer Science, Peking University, Institute for AI, Peking University, Yangtze Delta Region Academy of Beijing Institute of Technology, Jiaxing)
<details span>
<summary><b>Abstract</b></summary>
Despite significant advancements in text-to-motion synthesis, generating language-guided human motion within 3D environments poses substantial challenges. These challenges stem primarily from (i) the absence of powerful generative models capable of jointly modeling natural language, 3D scenes, and human motion, and (ii) the generative models' intensive data requirements contrasted with the scarcity of comprehensive, high-quality, language-scene-motion datasets. To tackle these issues, we introduce a novel two-stage framework that employs scene affordance as an intermediate representation, effectively linking 3D scene grounding and conditional motion generation. Our framework comprises an Affordance Diffusion Model (ADM) for predicting explicit affordance map and an Affordance-to-Motion Diffusion Model (AMDM) for generating plausible human motions. By leveraging scene affordance maps, our method overcomes the difficulty in generating human motion under multimodal condition signals, especially when training with limited data lacking extensive language-scene-motion pairs. Our extensive experiments demonstrate that our approach consistently outperforms all baselines on established benchmarks, including HumanML3D and HUMANISE. Additionally, we validate our model's exceptional generalization capabilities on a specially curated evaluation set featuring previously unseen descriptions and scenes.
</details>

### 19. in2IN:Leveraging individual Information to Generate Human INteractions
Pablo Ruiz Ponce, German Barquero, Cristina Palmero, Sergio Escalera, Jose Garcia-Rodriguez

(University of Alicante, University of Barcelona, Computer Vision Center)
<details span>
<summary><b>Abstract</b></summary>
Generating human-human motion interactions conditioned on textual descriptions is a very useful application in many areas such as robotics, gaming, animation, and the metaverse. Alongside this utility also comes a great difficulty in modeling the highly dimensional inter-personal dynamics. In addition, properly capturing the intra-personal diversity of interactions has a lot of challenges. Current methods generate interactions with limited diversity of intra-person dynamics due to the limitations of the available datasets and conditioning strategies. For this, we introduce in2IN, a novel diffusion model for human-human motion generation which is conditioned not only on the textual description of the overall interaction but also on the individual descriptions of the actions performed by each person involved in the interaction. To train this model, we use a large language model to extend the InterHuman dataset with individual descriptions. As a result, in2IN achieves state-of-the-art performance in the InterHuman dataset. Furthermore, in order to increase the intra-personal diversity on the existing interaction datasets, we propose DualMDM, a model composition technique that combines the motions generated with in2IN and the motions generated by a single-person motion prior pre-trained on HumanML3D. As a result, DualMDM generates motions with higher individual diversity and improves control over the intra-person dynamics while maintaining inter-personal coherence.
</details>

### 20. MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model
Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, Yansong Tang

(Tsinghua University, Shanghai AI Laboratory)
<details span>
<summary><b>Abstract</b></summary>
This work introduces MotionLCM, extending controllable motion generation to a real-time level. Existing methods for spatial control in text-conditioned motion generation suffer from significant runtime inefficiency. To address this issue, we first propose the motion latent consistency model (MotionLCM) for motion generation, building upon the latent diffusion model (MLD). By employing one-step (or few-step) inference, we further improve the runtime efficiency of the motion latent diffusion model for motion generation. To ensure effective controllability, we incorporate a motion ControlNet within the latent space of MotionLCM and enable explicit control signals (e.g., pelvis trajectory) in the vanilla motion space to control the generation process directly, similar to controlling other latent-free diffusion models for motion generation. By employing these techniques, our approach can generate human motions with text and control signals in real-time. Experimental results demonstrate the remarkable generation and controlling capabilities of MotionLCM while maintaining real-time runtime efficiency.
</details>

### 21. SATO: Stable Text-to-Motion Framework
Wenshuo Chen, Hongru Xiao, Erhang Zhang, Lijie Hu, Lei Wang, Mengyuan Liu, Chen Chen

(Shandong University, Tongji University, University of Central Florida, Peking University, King Abdullah University of Science and Technology, Australian National University)
<details span>
<summary><b>Abstract</b></summary>
Is the Text to Motion model robust? Recent advancements in Text to Motion models primarily stem from more accurate predictions of specific actions. However, the text modality typically relies solely on pre-trained Contrastive Language-Image Pretraining (CLIP) models. Our research has uncovered a significant issue with the text-to-motion model: its predictions often exhibit inconsistent outputs, resulting in vastly different or even incorrect poses when presented with semantically similar or identical text inputs. In this paper, we undertake an analysis to elucidate the underlying causes of this instability, establishing a clear link between the unpredictability of model outputs and the erratic attention patterns of the text encoder module. Consequently, we introduce a formal framework aimed at addressing this issue, which we term the Stable Text-to-Motion Framework (SATO). SATO consists of three modules, each dedicated to stable attention, stable prediction, and maintaining a balance between accuracy and robustness trade-off. We present a methodology for constructing an SATO that satisfies the stability of attention and prediction. To verify the stability of the model, we introduced a new textual synonym perturbation dataset based on HumanML3D and KIT-ML. Results show that SATO is significantly more stable against synonyms and other slight perturbations while keeping its high accuracy performance.
</details>


### 22. StableMoFusion: Towards Robust and Efficient Diffusion-based Motion Generation Framework
Yiheng Huang, Hui Yang, Chuanchen Luo, Yuxi Wang, Shibiao Xu, Zhaoxiang Zhang, Man Zhang, Junran Peng

(Beijing University of Posts and Telecommunications, CAIR/HKISI/CAS, Institute of Automation/Chinese Academy of Science)
<details span>
<summary><b>Abstract</b></summary>
Thanks to the powerful generative capacity of diffusion models, recent years have witnessed rapid progress in human motion generation. Existing diffusion-based methods employ disparate network architectures and training strategies. The effect of the design of each component is still unclear. In addition, the iterative denoising process consumes considerable computational overhead, which is prohibitive for real-time scenarios such as virtual characters and humanoid robots. For this reason, we first conduct a comprehensive investigation into network architectures, training strategies, and inference processs. Based on the profound analysis, we tailor each component for efficient high-quality human motion generation. Despite the promising performance, the tailored model still suffers from foot skating which is an ubiquitous issue in diffusion-based solutions. To eliminate footskate, we identify foot-ground contact and correct foot motions along the denoising process. By organically combining these well-designed components together, we present StableMoFusion, a robust and efficient framework for human motion generation. Extensive experimental results show that our StableMoFusion performs favorably against current state-of-the-art methods.
</details>

### 23. Flexible Motion In-betweening with Diffusion Models
Setareh Cohan, Guy Tevet, Daniele Reda, Xue Bin Peng, Michiel van de Panne

(University of British Columbia, Tel-Aviv University, Simon Fraser University, NVIDIA)
<details span>
<summary><b>Abstract</b></summary>
Motion in-betweening, a fundamental task in character animation, consists of generating motion sequences that plausibly interpolate user-provided keyframe constraints. It has long been recognized as a labor-intensive and challenging process. We investigate the potential of diffusion models in generating diverse human motions guided by keyframes. Unlike previous inbetweening methods, we propose a simple unified model capable of generating precise and diverse motions that conform to a flexible range of user-specified spatial constraints, as well as text conditioning. To this end, we propose Conditional Motion Diffusion In-betweening (CondMDI) which allows for arbitrary dense-or-sparse keyframe placement and partial keyframe constraints while generating high-quality motions that are diverse and coherent with the given keyframes. We evaluate the performance of CondMDI on the text-conditioned HumanML3D dataset and demonstrate the versatility and efficacy of diffusion models for keyframe in-betweening. We further explore the use of guidance and imputation-based approaches for inference-time keyframing and compare CondMDI against these methods.
</details>

### Text to Human Motion Paper lists
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2024 | **Synthesizing Moving People with 3D Control**  | Arxiv 2024 |          [Link](https://arxiv.org/abs/2401.10889)          | [Link](https://github.com/Boyiliee/3DHM)   | [Link](https://boyiliee.github.io/3DHM.github.io/)  |
| 2024 | **Multi-Track Timeline Control for Text-Driven 3D Human Motion Generation**  | CVPRW 2024 |          [Link](https://arxiv.org/abs/2401.08559)          | [Link](https://github.com/nv-tlabs/stmc)| [Link](https://mathis.petrovich.fr/stmc/)  |
| 2024 | **FlowMDM: Seamless Human Motion Composition with Blended Positional Encodings**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2402.15509)          | [Link](https://github.com/BarqueroGerman/FlowMDM) | [Link](https://barquerogerman.github.io/FlowMDM/)  |
| 2023 | **EMDM: Efficient Motion Diffusion Model for Fast, High-Quality Human Motion Generation**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2312.02256)          | [Link](https://github.com/Frank-ZY-Dou/EMDM) | [Link](https://frank-zy-dou.github.io/projects/EMDM/index.html)  |
| 2023 | **SinMDM: Single Motion Diffusion**  | ICLR 2024 Spotlight |          [Link](https://arxiv.org/abs/2302.05905)          | [Link](https://github.com/SinMDM/SinMDM)  | [Link](https://sinmdm.github.io/SinMDM-page/)  |
| 2023 | **MDM: Human Motion Diffusion Model**  | ICLR2023 (Top-25%) |          [Link](https://arxiv.org/abs/2209.14916)          | [Link](https://github.com/GuyTevet/motion-diffusion-model)  | [Link](https://guytevet.github.io/mdm-page/)  |
| 2023 | **MLD: Motion Latent Diffusion Models**  | CVPR 2023 |          [Link](https://arxiv.org/abs/2212.04048)          | [Link](https://github.com/ChenFengYe/motion-latent-diffusion)  | [Link](https://chenxin.tech/mld/)  |
| 2023 | **MotionMix: Weakly-Supervised Diffusion for Controllable Motion Generation**  | AAAI 2024 |          [Link](https://arxiv.org/abs/2401.11115)          | [Link](https://github.com/NhatHoang2002/MotionMix/tree/main)  | [Link](https://nhathoang2002.github.io/MotionMix-page/)  |
| 2023 | **HumanTOMATO: Text-aligned Whole-body Motion Generation**  | ICML 2024 |          [Link](https://arxiv.org/abs/2310.12978)          | [Link](https://github.com/IDEA-Research/HumanTOMATO)  | [Link](https://lhchen.top/HumanTOMATO/)  |
| 2023 | **MotionGPT: Human Motion as a Foreign Language**  | NeurIPS 2023 |          [Link](https://arxiv.org/abs/2306.14795)          | [Link](https://github.com/OpenMotionLab/MotionGPT)  | [Link](https://motion-gpt.github.io/)  |
| 2023 | **Story-to-Motion: Synthesizing Infinite and Controllable Character Animation from Long Text**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2311.07446)          | Coming soon! | [Link](https://story2motion.github.io/)  |
| 2023 | **Plan, Posture and Go: Towards Open-World Text-to-Motion Generation**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2312.14828)          | [Link](https://github.com/moonsliu/Pro-Motion) | [Link](https://moonsliu.github.io/Pro-Motion/)  |
| 2023 | **MoMask: Generative Masked Modeling of 3D Human Motions**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2312.00063)          | [Link](https://github.com/EricGuo5513/momask-codes) | [Link](https://ericguo5513.github.io/momask/)  |
| 2023 | **Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer**  | CVPR 2024 |          [Link](https://arxiv.org/abs/2403.18036)          | [Link](https://github.com/diffusion-motion-transfer/diffusion-motion-transfer) | [Link](https://diffusion-motion-transfer.github.io/)  |
| 2024 | **Self-Correcting Self-Consuming Loops for Generative Model Training**  | ICML 2024 |          [Link](https://arxiv.org/abs/2402.07087)          | [Link](https://github.com/nate-gillman/self-correcting-self-consuming) | [Link](https://cs.brown.edu/people/ngillman//sc-sc.html)  |
| 2024 | **Large Motion Model for Unified Multi-Modal Motion Generation**  | Arxiv 2024 |          [Link](https://arxiv.org/abs/2404.01284)          | [Link](https://github.com/mingyuan-zhang/LMM) | [Link](https://mingyuan-zhang.github.io/projects/LMM.html)  |
| 2024 | **Generating Human Interaction Motions in Scenes with Text Control**  | Arxiv 2024 |          [Link](https://arxiv.org/abs/2404.10685)          | -- | [Link](https://research.nvidia.com/labs/toronto-ai/tesmo/)  |
| 2024 | **Move as You Say, Interact as You Can: Language-guided Human Motion Generation with Scene Affordance**  | CVPR 2024 (Highlight) |          [Link](https://arxiv.org/abs/2311.17009)          | [Link](https://github.com/afford-motion/afford-motion) | [Link](https://afford-motion.github.io/)  |
| 2024 | **in2IN: Leveraging individual Information to Generate Human INteractions**  | HuMoGen CVPRW 2024 |          [Link](https://arxiv.org/abs/2404.09988)          | [Link](https://github.com/pabloruizponce/in2IN) | [Link](https://pabloruizponce.github.io/in2IN/)  |
| 2024 | **MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model**  | Arxiv 2024 |          [Link](https://arxiv.org/abs/2404.19759)          | [Link](https://github.com/Dai-Wenxun/MotionLCM) | [Link](https://dai-wenxun.github.io/MotionLCM-page/)  |
| 2024 | **SATO: Stable Text-to-Motion Framework**  | Arxiv 2024 |          [Link](https://arxiv.org/abs/2405.01461)          | [Link](https://github.com/sato-team/Stable-Text-to-motion-Framework) | [Link](https://sato-team.github.io/Stable-Text-to-Motion-Framework/)  |
| 2024 | **StableMoFusion: Towards Robust and Efficient Diffusion-based Motion Generation Framework**  | Arxiv 2024 |          [Link](https://arxiv.org/abs/2405.05691)          | [Link](https://github.com/h-y1heng/StableMoFusion) | [Link](https://h-y1heng.github.io/StableMoFusion-page/)  |
| 2024 | **Flexible Motion In-betweening with Diffusion Models**  |SIGGRAPH 2024|          [Link](https://arxiv.org/abs/2405.11126)          | [Link](https://github.com/setarehc/diffusion-motion-inbetweening) | [Link](https://setarehc.github.io/CondMDI/)  |

### Text to Human Motion Reference
<details close>
<summary>Text to Human Motion</summary>

```
% text to human motion

@article{li20243dhm,
    author = {Li, Boyi and Rajasegaran, Jathushan and Gandelsman, Yossi and Efros, Alexei A. and Malik, Jitendra},
    title = {Synthesizing Moving People with 3D Control},
    journal = {Arxiv},
    year = {2024},
}

@article{petrovich24stmc,
    title     = {{STMC}: Multi-Track Timeline Control for Text-Driven 3D Human Motion Generation},
    author    = {Petrovich, Mathis and Litany, Or and Iqbal, Umar and Black, Michael J. and Varol, G{\"u}l and Peng, Xue Bin and Rempe, Davis}
    journal   = {arXiv:2401.08559},
    year      = {2024}
}

@article{barquero2024seamless,
  title={Seamless Human Motion Composition with Blended Positional Encodings},
  author={Barquero, German and Escalera, Sergio and Palmero, Cristina},
  journal={arXiv preprint arXiv:2402.15509},
  year={2024}
}

@article{zhou2023emdm,
  title={EMDM: Efficient Motion Diffusion Model for Fast, High-Quality Motion Generation},
  author={Zhou, Wenyang and Dou, Zhiyang and Cao, Zeyu and Liao, Zhouyingcheng and Wang, Jingbo and Wang, Wenjia and Liu, Yuan and Komura, Taku and Wang, Wenping and Liu, Lingjie},
  journal={arXiv preprint arXiv:2312.02256},
  year={2023}
}

@article{raab2023single,
            title={Single Motion Diffusion},
            author={Raab, Sigal and Leibovitch, Inbal and Tevet, Guy and Arar, Moab and Bermano, Amit H and Cohen-Or, Daniel},
            journal={arXiv preprint arXiv:2302.05905},
            year={2023}
}

@inproceedings{
tevet2023human,
title={Human Motion Diffusion Model},
author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=SJ1kSyO2jwu}
}

@inproceedings{chen2023executing,
  title={Executing your Commands via Motion Diffusion in Latent Space},
  author={Chen, Xin and Jiang, Biao and Liu, Wen and Huang, Zilong and Fu, Bin and Chen, Tao and Yu, Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18000--18010},
  year={2023}
}

@misc{hoang2024motionmix,
  title={MotionMix: Weakly-Supervised Diffusion for Controllable Motion Generation}, 
  author={Nhat M. Hoang and Kehong Gong and Chuan Guo and Michael Bi Mi},
  year={2024},
  eprint={2401.11115},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@article{humantomato,
  title={HumanTOMATO: Text-aligned Whole-body Motion Generation},
  author={Lu, Shunlin and Chen, Ling-Hao and Zeng, Ailing and Lin, Jing and Zhang, Ruimao and Zhang, Lei and Shum, Heung-Yeung},
  journal={arxiv:2310.12978},
  year={2023}
}

@article{jiang2023motiongpt,
  title={MotionGPT: Human Motion as a Foreign Language},
  author={Jiang, Biao and Chen, Xin and Liu, Wen and Yu, Jingyi and Yu, Gang and Chen, Tao},
  journal={arXiv preprint arXiv:2306.14795},
  year={2023}
}

@misc{qing2023storytomotion,
        title={Story-to-Motion: Synthesizing Infinite and Controllable Character Animation from Long Text}, 
        author={Zhongfei Qing and Zhongang Cai and Zhitao Yang and Lei Yang},
        year={2023},
        eprint={2311.07446},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
}

@article{liu2023plan,
  title={Plan, Posture and Go: Towards Open-World Text-to-Motion Generation},
  author={Liu, Jinpeng and Dai, Wenxun and Wang, Chunyu and Cheng, Yiji and Tang, Yansong and Tong, Xin},
  journal={arXiv preprint arXiv:2312.14828},
  year={2023}
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

@misc{gillman2024selfcorrecting,
  title={Self-Correcting Self-Consuming Loops for Generative Model Training}, 
  author={Nate Gillman and Michael Freeman and Daksh Aggarwal and Chia-Hong Hsu and Calvin Luo and Yonglong Tian and Chen Sun},
  year={2024},
  eprint={2402.07087},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
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

@article{yi2024tesmo,
    author={Yi, Hongwei and Thies, Justus and Black, Michael J. and Peng, Xue Bin and Rempe, Davis},
    title={Generating Human Interaction Motions in Scenes with Text Control},
    journal = {arXiv:2404.10685},
    year={2024}
}

@inproceedings{wang2024move,
  title={Move as You Say, Interact as You Can: Language-guided Human Motion Generation with Scene Affordance},
  author={Wang, Zan and Chen, Yixin and Jia, Baoxiong and Li, Puhao and Zhang, Jinlu and Zhang, Jingze and Liu, Tengyu and Zhu, Yixin and Liang, Wei and Huang, Siyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

@misc{ponce2024in2in,
      title={in2IN: Leveraging individual Information to Generate Human INteractions}, 
      author={Pablo Ruiz Ponce and German Barquero and Cristina Palmero and Sergio Escalera and Jose Garcia-Rodriguez},
      year={2024},
      eprint={2404.09988},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{motionlcm,
      title={MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model},
      author={Wenxun Dai and Ling-Hao Chen and Jingbo Wang and Jinpeng Liu and Bo Dai and Yansong Tang},
      journal={arXiv preprint arXiv:2404.19759},
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

@article{huang2024stablemofusion,
        title={StableMoFusion: Towards Robust and Efficient Diffusion-based Motion Generation Framework},
        author = {Huang, Yiheng and Hui, Yang and Luo, Chuanchen and Wang, Yuxi and Xu, Shibiao and Zhang, Zhaoxiang and Zhang, Man and Peng, Junran},
        journal = {arXiv preprint arXiv: 2405.05691},
        year = {2024}
}

@misc{cohan2024flexible,
      title={Flexible Motion In-betweening with Diffusion Models}, 
      author={Setareh Cohan and Guy Tevet and Daniele Reda and Xue Bin Peng and Michiel van de Panne},
      year={2024},
      eprint={2405.11126},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
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

--------------

## Related Resources
### Dynamic Gaussian Splatting
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

</details>

<details close>
<summary>🔥 Topic 2: AIGC 3D </summary>
 
#### Survey
- [Advances in 3D Generation: A Survey](https://arxiv.org/abs/2401.17807), ArXiv 2024
- [A Comprehensive Survey on 3D Content Generation](https://arxiv.org/abs/2402.01166), ArXiv 2024

#### Awesome Repos
- Resource: [Awesome 3D AIGC 1](https://github.com/mdyao/Awesome-3D-AIGC) and [Awesome 3D AIGC 2](https://github.com/hitcslj/Awesome-AIGC-3D)


#### Benchmars
- text-to-3d generation: [GPT-4V(ision) is a Human-Aligned Evaluator for Text-to-3D Generation](https://arxiv.org/abs/2401.04092), Wu et al., arXiv 2024 | [Code](https://github.com/3DTopia/GPTEval3D)
</details>

<details close>
<summary>🔥 Topic 3: LLM 3D </summary>
 
#### Awesome Repos
- Resource1: [Awesome LLM 3D](https://github.com/ActiveVisionLab/Awesome-LLM-3D)

#### 3D Human
- Survey: [PROGRESS AND PROSPECTS IN 3D GENERATIVE AI: A TECHNICAL OVERVIEW INCLUDING 3D HUMAN](https://arxiv.org/pdf/2401.02620.pdf), ArXiv 2024
- Resource1: [Awesome Digital Human](https://github.com/weihaox/awesome-digital-human)
- Resource2: [Awesome-Avatars](https://github.com/pansanity666/Awesome-Avatars)

</details>

--------------

## License 
Awesome Text2X Resources is released under the [MIT license](./LICENSE).
