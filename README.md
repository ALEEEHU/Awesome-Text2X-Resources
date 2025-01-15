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
* `2024.12.21` adjusted the layouts of several sections and _Happy Winter Solstice_ ⚪🥣.
* `2024.04.05` adjusted the layout and added accepted lists and ArXiv lists to each section.


<div><div align="center">
	<img width="500" height="350" src="media/logo.svg" alt="Awesome"></div>

## Table of Contents

- [Text to 4D](#text-to-4d)
  * [ArXiv Papers](#-4d-arxiv-papers)
- [Text to Video](#text-to-video)
  * [ArXiv Papers](#-t2v-arxiv-papers)
  * [Additional Info](#other-additional-info)
- [Text to Human Motion](#text-to-human-motion)
  * [Accepted Papers](#-motion-accepted-papers)
  * [ArXiv Papers](#-motion-arxiv-papers)
  * [Datasets](#datasets)
- [Text to 3D Human](#text-to-3d-human)
  * [Accepted Papers](#-human-accepted-papers)
  * [ArXiv Papers](#-human-arxiv-papers)
- [Text to 3D Scene](#text-to-scene)
  * [Accepted Papers](#-scene-accepted-papers)
  * [ArXiv Papers](#-scene-arxiv-papers)
- [Related Resources](#related-resources)
  * [Text to Other Tasks](#text-to-other-tasks)
  * [Survey and Awesome Repos](#survey-and-awesome-repos)

## Update Logs
<details span>
<summary><b>2025 Update Logs:</b></summary>
<br>	
	
* `2025.01.09` - update layout.

  
</details>

<details close>
<summary><b>2024 Update Logs:</b></summary>
	
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
(Also, Image/Video to 4D)

### 💡 4D ArXiv Papers

#### 1. AR4D: Autoregressive 4D Generation from Monocular Videos
Hanxin Zhu, Tianyu He, Xiqian Yu, Junliang Guo, Zhibo Chen, Jiang Bian (University of Science and Technology of China, Microsoft Research Asia)
<details span>
<summary><b>Abstract</b></summary>
Recent advancements in generative models have ignited substantial interest in dynamic 3D content creation (\ie, 4D generation). Existing approaches primarily rely on Score Distillation Sampling (SDS) to infer novel-view videos, typically leading to issues such as limited diversity, spatial-temporal inconsistency and poor prompt alignment, due to the inherent randomness of SDS. To tackle these problems, we propose AR4D, a novel paradigm for SDS-free 4D generation. Specifically, our paradigm consists of three stages. To begin with, for a monocular video that is either generated or captured, we first utilize pre-trained expert models to create a 3D representation of the first frame, which is further fine-tuned to serve as the canonical space. Subsequently, motivated by the fact that videos happen naturally in an autoregressive manner, we propose to generate each frame's 3D representation based on its previous frame's representation, as this autoregressive generation manner can facilitate more accurate geometry and motion estimation. Meanwhile, to prevent overfitting during this process, we introduce a progressive view sampling strategy, utilizing priors from pre-trained large-scale 3D reconstruction models. To avoid appearance drift introduced by autoregressive generation, we further incorporate a refinement stage based on a global deformation field and the geometry of each frame's 3D representation. Extensive experiments have demonstrated that AR4D can achieve state-of-the-art 4D generation without SDS, delivering greater diversity, improved spatial-temporal consistency, and better alignment with input prompts.
</details>

#### 2. GS-DiT: Advancing Video Generation with Pseudo 4D Gaussian Fields through Efficient Dense 3D Point Tracking
Weikang Bian, Zhaoyang Huang, Xiaoyu Shi, Yijin Li, Fu-Yun Wang, Hongsheng Li

(The Chinese University of Hong Kong, Centre for Perceptual and Interactive Intelligence, Avolution AI)
<details span>
<summary><b>Abstract</b></summary>
4D video control is essential in video generation as it enables the use of sophisticated lens techniques, such as multi-camera shooting and dolly zoom, which are currently unsupported by existing methods. Training a video Diffusion Transformer (DiT) directly to control 4D content requires expensive multi-view videos. Inspired by Monocular Dynamic novel View Synthesis (MDVS) that optimizes a 4D representation and renders videos according to different 4D elements, such as camera pose and object motion editing, we bring pseudo 4D Gaussian fields to video generation. Specifically, we propose a novel framework that constructs a pseudo 4D Gaussian field with dense 3D point tracking and renders the Gaussian field for all video frames. Then we finetune a pretrained DiT to generate videos following the guidance of the rendered video, dubbed as GS-DiT. To boost the training of the GS-DiT, we also propose an efficient Dense 3D Point Tracking (D3D-PT) method for the pseudo 4D Gaussian field construction. Our D3D-PT outperforms SpatialTracker, the state-of-the-art sparse 3D point tracking method, in accuracy and accelerates the inference speed by two orders of magnitude. During the inference stage, GS-DiT can generate videos with the same dynamic content while adhering to different camera parameters, addressing a significant limitation of current video generation models. GS-DiT demonstrates strong generalization capabilities and extends the 4D controllability of Gaussian splatting to video generation beyond just camera poses. It supports advanced cinematic effects through the manipulation of the Gaussian field and camera intrinsics, making it a powerful tool for creative video production.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **AR4D: Autoregressive 4D Generation from Monocular Videos**  | 3 Jan 2025 |          [Link](https://arxiv.org/abs/2501.01722)          | --  | [Link](https://hanxinzhu-lab.github.io/AR4D/)  |
| 2025 | **GS-DiT: Advancing Video Generation with Pseudo 4D Gaussian Fields through Efficient Dense 3D Point Tracking**  | 5 Jan 2025 |          [Link](https://arxiv.org/abs/2501.02690)          | [Link](https://github.com/wkbian/GS-DiT)  | [Link](https://wkbian.github.io/Projects/GS-DiT/)  |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@misc{zhu2025ar4dautoregressive4dgeneration,
      title={AR4D: Autoregressive 4D Generation from Monocular Videos}, 
      author={Hanxin Zhu and Tianyu He and Xiqian Yu and Junliang Guo and Zhibo Chen and Jiang Bian},
      year={2025},
      eprint={2501.01722},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.01722}, 
}

@article{bian2025gsdit,
  title={GS-DiT: Advancing Video Generation with Pseudo 4D Gaussian Fields through Efficient Dense 3D Point Tracking},
  author={Bian, Weikang and Huang, Zhaoyang and Shi, Xiaoyu and and Li, Yijin and Wang, Fu-Yun and Li, Hongsheng},
  journal={arXiv preprint arXiv:2501.02690},
  year={2025}
}

```
</details>

---

### Previous Papers

### Year 2023
In 2023, tasks classified as text/Image to 4D and video to 4D generally involve producing four-dimensional data from text/Image or video input. For more details, please check the [2023 4D Papers](./docs/4d/4d_2023.md), including 6 accepted papers and 3 arXiv papers.

### Year 2024
For more details, please check the [2024 4D Papers](./docs/4d/4d_2024.md), including 12 accepted papers and 22 arXiv papers.



## Text to Video

### 💡 T2V ArXiv Papers

#### 1. TransPixar: Advancing Text-to-Video Generation with Transparency
Luozhou Wang, Yijun Li, Zhifei Chen, Jui-Hsien Wang, Zhifei Zhang, He Zhang, Zhe Lin, Yingcong Chen

(HKUST(GZ), HKUST, Adobe Research)
<details span>
<summary><b>Abstract</b></summary>
Text-to-video generative models have made significant strides, enabling diverse applications in entertainment, advertising, and education. However, generating RGBA video, which includes alpha channels for transparency, remains a challenge due to limited datasets and the difficulty of adapting existing models. Alpha channels are crucial for visual effects (VFX), allowing transparent elements like smoke and reflections to blend seamlessly into scenes. We introduce TransPixar, a method to extend pretrained video models for RGBA generation while retaining the original RGB capabilities. TransPixar leverages a diffusion transformer (DiT) architecture, incorporating alpha-specific tokens and using LoRA-based fine-tuning to jointly generate RGB and alpha channels with high consistency. By optimizing attention mechanisms, TransPixar preserves the strengths of the original RGB model and achieves strong alignment between RGB and alpha channels despite limited training data. Our approach effectively generates diverse and consistent RGBA videos, advancing the possibilities for VFX and interactive content creation.
</details>

#### 2. Multi-subject Open-set Personalization in Video Generation
Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, Sergey Tulyakov

(Snap Inc., UC Merced, CMU)
<details span>
<summary><b>Abstract</b></summary>
Video personalization methods allow us to synthesize videos with specific concepts such as people, pets, and places. However, existing methods often focus on limited domains, require time-consuming optimization per subject, or support only a single subject. We present Video Alchemist − a video model with built-in multi-subject, open-set personalization capabilities for both foreground objects and background, eliminating the need for time-consuming test-time optimization. Our model is built on a new Diffusion Transformer module that fuses each conditional reference image and its corresponding subject-level text prompt with cross-attention layers. Developing such a large model presents two main challenges: dataset and evaluation. First, as paired datasets of reference images and videos are extremely hard to collect, we sample selected video frames as reference images and synthesize a clip of the target video. However, while models can easily denoise training videos given reference frames, they fail to generalize to new contexts. To mitigate this issue, we design a new automatic data construction pipeline with extensive image augmentations. Second, evaluating open-set video personalization is a challenge in itself. To address this, we introduce a personalization benchmark that focuses on accurate subject fidelity and supports diverse personalization scenarios. Finally, our extensive experiments show that our method significantly outperforms existing personalization methods in both quantitative and qualitative evaluations.
</details>

| Year | Title                                                        | ArXiv Time  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2025 | **TransPixar: Advancing Text-to-Video Generation with Transparency**  | 6 Jan 2025 |          [Link](https://arxiv.org/abs/2501.03006)          | [Link](https://github.com/wileewang/TransPixar)  | [Link](https://wileewang.github.io/TransPixar/)  |
| 2025 | **Multi-subject Open-set Personalization in Video Generation**  | 10 Jan 2025 |          [Link](https://arxiv.org/abs/2501.06187)          | -- | [Link](https://snap-research.github.io/open-set-video-personalization/)  |

<details close>
<summary>ArXiv Papers References</summary>

```
%axiv papers

@misc{wang2025transpixar,
     title={TransPixar: Advancing Text-to-Video Generation with Transparency}, 
     author={Luozhou Wang and Yijun Li and Zhifei Chen and Jui-Hsien Wang and Zhifei Zhang and He Zhang and Zhe Lin and Yingcong Chen},
     year={2025},
     eprint={2501.03006},
     archivePrefix={arXiv},
     primaryClass={cs.CV},
     url={https://arxiv.org/abs/2501.03006}, 
}

@misc{chen2025multisubjectopensetpersonalizationvideo,
      title={Multi-subject Open-set Personalization in Video Generation}, 
      author={Tsai-Shien Chen and Aliaksandr Siarohin and Willi Menapace and Yuwei Fang and Kwot Sin Lee and Ivan Skorokhodov and Kfir Aberman and Jun-Yan Zhu and Ming-Hsuan Yang and Sergey Tulyakov},
      year={2025},
      eprint={2501.06187},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.06187}, 
}

```
</details>


---

### Other Additional Info

### Previous Papers

### Year 2024
For more details, please check the [2024 T2V Papers](./docs/video/t2v_2024.md), including 7 accepted papers and 19 arXiv papers.

- OSS video generation models: [Mochi 1](https://github.com/genmoai/models) preview is an open state-of-the-art video generation model with high-fidelity motion and strong prompt adherence.
- Survey: The Dawn of Video Generation: Preliminary Explorations with SORA-like Models, [arXiv](https://arxiv.org/abs/2410.05227), [Project Page](https://ailab-cvc.github.io/VideoGen-Eval/), [GitHub Repo](https://github.com/AILab-CVC/VideoGen-Eval)

### 📚 Dataset Works

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

#### 16. ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model
Shunlin Lu, Jingbo Wang, Zeyu Lu, Ling-Hao Chen, Wenxun Dai, Junting Dong, Zhiyang Dou, Bo Dai, Ruimao Zhang

(Sun Yat-sen University, The Chinese University of Hongkong Shenzhen, Shanghai AI Laboratory, Tsinghua University, Shanghai Jiao Tong University, The University of Hong Kong)
<details span>
<summary><b>Abstract</b></summary>
The scaling law has been validated in various domains, such as natural language processing (NLP) and massive computer vision tasks; however, its application to motion generation remains largely unexplored. In this paper, we introduce a scalable motion generation framework that includes the motion tokenizer Motion FSQ-VAE and a text-prefix autoregressive transformer. Through comprehensive experiments, we observe the scaling behavior of this system. For the first time, we confirm the existence of scaling laws within the context of motion generation. Specifically, our results demonstrate that the normalized test loss of our prefix autoregressive models adheres to a logarithmic law in relation to compute budgets. Furthermore, we also confirm the power law between Non-Vocabulary Parameters, Vocabulary Parameters, and Data Tokens with respect to compute budgets respectively. Leveraging the scaling law, we predict the optimal transformer size, vocabulary size, and data requirements for a compute budget of 1e18. The test loss of the system, when trained with the optimal model size, vocabulary size, and required data, aligns precisely with the predicted test loss, thereby validating the scaling law.
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
| 2024 | **ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model**  | 19 Dec 2024 |          [Link](https://arxiv.org/abs/2412.14559)          | [Link](https://github.com/shunlinlu/ScaMo_code) | [Link](https://shunlinlu.github.io/ScaMo/)  |

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

@misc{lu2024scamoexploringscalinglaw,
      title={ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model}, 
      author={Shunlin Lu and Jingbo Wang and Zeyu Lu and Ling-Hao Chen and Wenxun Dai and Junting Dong and Zhiyang Dou and Bo Dai and Ruimao Zhang},
      year={2024},
      eprint={2412.14559},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.14559}, 
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

#### 7. MixedGaussianAvatar: Realistically and Geometrically Accurate Head Avatar via Mixed 2D-3D Gaussian Splatting
Peng Chen, Xiaobao Wei, Qingpo Wuwu, Xinyi Wang, Xingyu Xiao, Ming Lu

(Institute of Software Chinese Academy of Sciences, University of Chinese Academy of Sciences, Intel Labs China, Tsinghua University, Nankai University, Peking University)
<details span>
<summary><b>Abstract</b></summary>
Reconstructing high-fidelity 3D head avatars is crucial in various applications such as virtual reality. The pioneering methods reconstruct realistic head avatars with Neural Radiance Fields (NeRF), which have been limited by training and rendering speed. Recent methods based on 3D Gaussian Splatting (3DGS) significantly improve the efficiency of training and rendering. However, the surface inconsistency of 3DGS results in subpar geometric accuracy; later, 2DGS uses 2D surfels to enhance geometric accuracy at the expense of rendering fidelity. To leverage the benefits of both 2DGS and 3DGS, we propose a novel method named MixedGaussianAvatar for realistically and geometrically accurate head avatar reconstruction. Our main idea is to utilize 2D Gaussians to reconstruct the surface of the 3D head, ensuring geometric accuracy. We attach the 2D Gaussians to the triangular mesh of the FLAME model and connect additional 3D Gaussians to those 2D Gaussians where the rendering quality of 2DGS is inadequate, creating a mixed 2D-3D Gaussian representation. These 2D-3D Gaussians can then be animated using FLAME parameters. We further introduce a progressive training strategy that first trains the 2D Gaussians and then fine-tunes the mixed 2D-3D Gaussians. We demonstrate the superiority of MixedGaussianAvatar through comprehensive experiments.
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
| 2024 | **MixedGaussianAvatar: Realistically and Geometrically Accurate Head Avatar via Mixed 2D-3D Gaussian Splatting**  | 6 Dec 2024  |    [Link](https://arxiv.org/abs/2412.04955)       | [Link](https://github.com/ChenVoid/MGA/)| [Link](https://chenvoid.github.io/MGA/) |

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

@misc{chen2024mixedgaussianavatarrealisticallygeometricallyaccurate,
      title={MixedGaussianAvatar: Realistically and Geometrically Accurate Head Avatar via Mixed 2D-3D Gaussian Splatting}, 
      author={Peng Chen and Xiaobao Wei and Qingpo Wuwu and Xinyi Wang and Xingyu Xiao and Ming Lu},
      year={2024},
      eprint={2412.04955},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.04955}, 
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

#### 12. WonderWorld: Interactive 3D Scene Generation from a Single Image
Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T. Freeman, Jiajun Wu

(Stanford University, MIT)
<details span>
<summary><b>Abstract</b></summary>
We present WonderWorld, a novel framework for interactive 3D scene generation that enables users to interactively specify scene contents and layout and see the created scenes in low latency. The major challenge lies in achieving fast generation of 3D scenes. Existing scene generation approaches fall short of speed as they often require (1) progressively generating many views and depth maps, and (2) time-consuming optimization of the scene geometry representations. We introduce the Fast Layered Gaussian Surfels (FLAGS) as our scene representation and an algorithm to generate it from a single view. Our approach does not need multiple views, and it leverages a geometry-based initialization that significantly reduces optimization time. Another challenge is generating coherent geometry that allows all scenes to be connected. We introduce the guided depth diffusion that allows partial conditioning of depth estimation. WonderWorld generates connected and diverse 3D scenes in less than 10 seconds on a single A6000 GPU, enabling real-time user interaction and exploration. We demonstrate the potential of WonderWorld for user-driven content creation and exploration in virtual environments. We will release full code and software for reproducibility.
</details>

#### 13. Semantic Score Distillation Sampling for Compositional Text-to-3D Generation
Ling Yang, Zixiang Zhang, Junlin Han, Bohan Zeng, Runjia Li, Philip Torr, Wentao Zhang

(Peking University, University of Oxford)
<details span>
<summary><b>Abstract</b></summary>
Generating high-quality 3D assets from textual descriptions remains a pivotal challenge in computer graphics and vision research. Due to the scarcity of 3D data, state-of-the-art approaches utilize pre-trained 2D diffusion priors, optimized through Score Distillation Sampling (SDS). Despite progress, crafting complex 3D scenes featuring multiple objects or intricate interactions is still difficult. To tackle this, recent methods have incorporated box or layout guidance. However, these layout-guided compositional methods often struggle to provide fine-grained control, as they are generally coarse and lack expressiveness. To overcome these challenges, we introduce a novel SDS approach, Semantic Score Distillation Sampling (SemanticSDS), designed to effectively improve the expressiveness and accuracy of compositional text-to-3D generation. Our approach integrates new semantic embeddings that maintain consistency across different rendering views and clearly differentiate between various objects and parts. These embeddings are transformed into a semantic map, which directs a region-specific SDS process, enabling precise optimization and compositional generation. By leveraging explicit semantic guidance, our method unlocks the compositional capabilities of existing pre-trained diffusion models, thereby achieving superior quality in 3D content generation, particularly for complex objects and scenes. Experimental results demonstrate that our SemanticSDS framework is highly effective for generating state-of-the-art complex 3D content.
</details>

#### 14. The Scene Language: Representing Scenes with Programs, Words, and Embeddings
Yunzhi Zhang, Zizhang Li, Matt Zhou, Shangzhe Wu, Jiajun Wu (Stanford University, UC Berkeley)
<details span>
<summary><b>Abstract</b></summary>
We introduce the Scene Language, a visual scene representation that concisely and precisely describes the structure, semantics, and identity of visual scenes. It represents a scene with three key components: a program that specifies the hierarchical and relational structure of entities in the scene, words in natural language that summarize the semantic class of each entity, and embeddings that capture the visual identity of each entity. This representation can be inferred from pre-trained language models via a training-free inference technique, given text or image inputs. The resulting scene can be rendered into images using traditional, neural, or hybrid graphics renderers. Together, this forms a robust, automated system for high-quality 3D and 4D scene generation. Compared with existing representations like scene graphs, our proposed Scene Language generates complex scenes with higher fidelity, while explicitly modeling the scene structures to enable precise control and editing.
</details>

#### 15. Prometheus: 3D-Aware Latent Diffusion Models for Feed-Forward Text-to-3D Scene Generation
Yuanbo Yang, Jiahao Shao, Xinyang Li, Yujun Shen, Andreas Geiger, Yiyi Liao

(Zhejiang University, Xiamen University, Ant Group, University of Tübingen)
<details span>
<summary><b>Abstract</b></summary>
In this work, we introduce Prometheus, a 3D-aware latent diffusion model for text-to-3D generation at both object and scene levels in seconds. We formulate 3D scene generation as multi-view, feed-forward, pixel-aligned 3D Gaussian generation within the latent diffusion paradigm. To ensure generalizability, we build our model upon pre-trained text-to-image generation model with only minimal adjustments, and further train it using a large number of images from both single-view and multi-view datasets. Furthermore, we introduce an RGB-D latent space into 3D Gaussian generation to disentangle appearance and geometry information, enabling efficient feed-forward generation of 3D Gaussians with better fidelity and geometry. Extensive experimental results demonstrate the effectiveness of our method in both feed-forward 3D Gaussian reconstruction and text-to-3D generation. 
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
| 2024 | **WonderWorld: Interactive 3D Scene Generation from a Single Image**  | 10 Sep 2024 |          [Link](https://arxiv.org/abs/2406.09394)          | [Link](https://github.com/KovenYu/WonderWorld)  | [Link](https://kovenyu.com/wonderworld/) |
| 2024 | **Semantic Score Distillation Sampling for Compositional Text-to-3D Generation**  | 11 Oct 2024 |          [Link](https://arxiv.org/abs/2410.09009)          | [Link](https://github.com/YangLing0818/SemanticSDS-3D)  | -- |
| 2024 | **The Scene Language: Representing Scenes with Programs, Words, and Embeddings**  | 22 Oct 2024 |          [Link](https://arxiv.org/abs/2410.16770)          | [Link](https://github.com/zzyunzhi/scene-language)  | [Link](https://ai.stanford.edu/~yzzhang/projects/scene-language/)|
| 2024 | **Prometheus: 3D-Aware Latent Diffusion Models for Feed-Forward Text-to-3D Scene Generation**  | 30 Dec 2024 |          [Link](https://arxiv.org/abs/2412.21117)          | [Link](https://github.com/XDimLab/Prometheus)  | [Link](https://freemty.github.io/project-prometheus/)|

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

@article{yu2024wonderworld,
    title={WonderWorld: Interactive 3D Scene Generation from a Single Image},
    author={Hong-Xing Yu and Haoyi Duan and Charles Herrmann and William T. Freeman and Jiajun Wu},
    journal={arXiv:2406.09394},
    year={2024}
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

@article{yang2024prometheus,
      title={Prometheus: 3D-Aware Latent Diffusion Models for Feed-Forward Text-to-3D Scene Generation}, 
      author={Yuanbo, Yang and Jiahao, Shao and Xinyang, Li and Yujun, Shen and Andreas, Geiger and Yiyi, Liao},
      year={2024},
      journal= {arxiv:2412.21117},
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
