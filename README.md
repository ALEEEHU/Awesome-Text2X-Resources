# Awesome Text2X Resources
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FALEEEHU%2FAwesome-Text2X-Resources%2F&count_bg=%23EAA8EA&title_bg=%233D2549&icon=react.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=true)](https://hits.seeyoufarm.com)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-pink) ![Stars](https://img.shields.io/github/stars/ALEEEHU/Awesome-Text2X-Resources)

This is an open collection of state-of-the-art (SOTA), novel **Text to X (X can be everything)** methods (papers, codes and datasets), intended to keep pace with the anticipated surge of research in the coming months. 

⭐ If you find this repository useful to your research or work, it is really appreciated to star this repository. 

:heart: Any additions or suggestions, feel free to contribute and contact hyqale1024@gmail.com. Additional resources like blog posts, videos, etc. are also welcome.

## Table of contents

- [Text to 4D](#text-to-4d)
  * [Paper lists](#text-to-4d-paper-lists)
  * [Reference](#text-to-4d-reference)
- [Text to Scene](#text-to-scene)
  * [Paper lists](#text-to-scene-paper-lists)
  * [Reference](#text-to-scene-reference)
- [Text to 3D Human](#text-to-3d-human)
  * [Paper lists](#text-to-3d-human-paper-lists)
  * [Reference](#text-to-3d-human-reference)
  * [Additional Info](#additional-info)
- [Text to Human Motion](#text-to-human-motion)
  * [Paper lists](#text-to-human-motion-paper-lists)
  * [Reference](#text-to-human-motion-reference)
  * [Datasets](#datasets)
- [Related Resources](#related-resources)
  * [Survey and Awesome Repos](#survey-and-awesome-repos)


## Text to 4D
(Also, Image to 4D)

### 1. Text-To-4D Dynamic Scene Generation
Uriel Singer*, Shelly Sheynin*, Adam Polyak*, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, Yaniv Taigman

(Meta AI)
<details span>
<summary><b>Abstract</b></summary>
We present MAV3D (Make-A-Video3D), a method for generating three-dimensional dynamic scenes from text descriptions. Our approach uses a 4D dynamic Neural Radiance Field (NeRF), which is optimized for scene appearance, density, and motion consistency by querying a Text-to-Video (T2V) diffusion-based model. The dynamic video output generated from the provided text can be viewed from any camera location and angle, and can be composited into any 3D environment. MAV3D does not require any 3D or 4D data and the T2V model is trained only on Text-Image pairs and unlabeled videos. We demonstrate the effectiveness of our approach using comprehensive quantitative and qualitative experiments and show an improvement over previously established internal baselines. To the best of our knowledge, our method is the first to generate 3D dynamic scenes given a text description.
</details>

### 2. 4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling
Bahmani, Sherwin, Ivan, Skorokhodov, Victor, Rong, Gordon, Wetzstein, Leonidas, Guibas, Peter, Wonka, Sergey, Tulyakov, Jeong Joon, Park, Andrea, Tagliasacchi, David B., Lindell.

(University of Toronto, Vector Institute, KAUST, Snap Inc., Stanford University, University of Michigan, SFU, Google)
<details span>
<summary><b>Abstract</b></summary>
Recent breakthroughs in text-to-4D generation rely on pre-trained text-to-image and text-to-video models to generate dynamic 3D scenes. However, current text-to-4D methods face a three-way tradeoff between the quality of scene appearance, 3D structure, and motion. For example, text-to-image models and their 3D-aware variants are trained on internet-scale image datasets and can be used to produce scenes with realistic appearance and 3D structure-but no motion. Text-to-video models are trained on relatively smaller video datasets and can produce scenes with motion, but poorer appearance and 3D structure. While these models have complementary strengths, they also have opposing weaknesses, making it difficult to combine them in a way that alleviates this three-way tradeoff. Here, we introduce hybrid score distillation sampling, an alternating optimization procedure that blends supervision signals from multiple pre-trained diffusion models and incorporates benefits of each for high-fidelity text-to-4D generation. Using hybrid SDS, we demonstrate synthesis of 4D scenes with compelling appearance, 3D structure, and motion.
</details>

### 3. A Unified Approach for Text- and Image-guided 4D Scene Generation
Yufeng Zheng, Xueting Li, Koki Nagano, Sifei Liu, Karsten Kreis, Otmar Hilliges, Shalini De Mello

(NVIDIA, ETH Zurich, Max Planck Institute for Intelligent Systems)
<details span>
<summary><b>Abstract</b></summary>
Large-scale diffusion generative models are greatly simplifying image, video and 3D asset creation from user-provided text prompts and images. However, the challenging problem of text-to-4D dynamic 3D scene generation with diffusion guidance remains largely unexplored. We propose Dream-in-4D, which features a novel two-stage approach for text-to-4D synthesis, leveraging (1) 3D and 2D diffusion guidance to effectively learn a high-quality static 3D asset in the first stage; (2) a deformable neural radiance field that explicitly disentangles the learned static asset from its deformation, preserving quality during motion learning; and (3) a multi-resolution feature grid for the deformation field with a displacement total variation loss to effectively learn motion with video diffusion guidance in the second stage. Through a user preference study, we demonstrate that our approach significantly advances image and motion quality, 3D consistency and text fidelity for text-to-4D generation compared to baseline approaches. Thanks to its motion-disentangled representation, Dream-in-4D can also be easily adapted for controllable generation where appearance is defined by one or multiple images, without the need to modify the motion learning stage. Thus, our method offers, for the first time, a unified approach for text-to-4D, image-to-4D and personalized 4D generation tasks.
</details>

### 4. Animate124: Animating One Image to 4D Dynamic Scene
Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, Gim Hee Lee

(National University of Singapore, Huawei Noah's Ark Lab)
<details span>
<summary><b>Abstract</b></summary>
We introduce Animate124 (Animate-one-image-to-4D), the first work to animate a single in-the-wild image into 3D video through textual motion descriptions, an underexplored problem with significant applications. Our 4D generation leverages an advanced 4D grid dynamic Neural Radiance Field (NeRF) model, optimized in three distinct stages using multiple diffusion priors. Initially, a static model is optimized using the reference image, guided by 2D and 3D diffusion priors, which serves as the initialization for the dynamic NeRF. Subsequently, a video diffusion model is employed to learn the motion specific to the subject. However, the object in the 3D videos tends to drift away from the reference image over time. This drift is mainly due to the misalignment between the text prompt and the reference image in the video diffusion model. In the final stage, a personalized diffusion prior is therefore utilized to address the semantic drift. As the pioneering image-text-to-4D generation framework, our method demonstrates significant advancements over existing baselines, evidenced by comprehensive quantitative and qualitative assessments.
</details>

### 5. 4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency
Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, Yunchao Wei

(Beijing Jiaotong University, University of Texas at Austin)
<details span>
<summary><b>Abstract</b></summary>
Aided by text-to-image and text-to-video diffusion models, existing 4D content creation pipelines utilize score distillation sampling to optimize the entire dynamic 3D scene. However, as these pipelines generate 4D content from text or image inputs, they incur significant time and effort in prompt engineering through trial and error. This work introduces 4DGen, a novel, holistic framework for grounded 4D content creation that decomposes the 4D generation task into multiple stages. We identify static 3D assets and monocular video sequences as key components in constructing the 4D content. Our pipeline facilitates conditional 4D generation, enabling users to specify geometry (3D assets) and motion (monocular videos), thus offering superior control over content creation. Furthermore, we construct our 4D representation using dynamic 3D Gaussians, which permits efficient, high-resolution supervision through rendering during training, thereby facilitating high-quality 4D generation. Additionally, we employ spatial-temporal pseudo labels on anchor frames, along with seamless consistency priors implemented through 3D-aware score distillation sampling and smoothness regularizations. Compared to existing baselines, our approach yields competitive results in faithfully reconstructing input signals and realistically inferring renderings from novel viewpoints and timesteps. Most importantly, our method supports grounded generation, offering users enhanced control, a feature difficult to achieve with previous methods.
</details>

### 6. Fast Dynamic 3D Object Generation from a Single-view Video
Zijie Pan, Zeyu Yang, Xiatian Zhu, Li Zhang (Fudan University, University of Surrey)
<details span>
<summary><b>Abstract</b></summary>
Generating dynamic three-dimensional (3D) object from a single-view video is challenging due to the lack of 4D labeled data. Existing methods extend text-to-3D pipelines by transferring off-the-shelf image generation models such as score distillation sampling, but they are slow and expensive to scale (e.g., 150 minutes per object) due to the need for back-propagating the information-limited supervision signals through a large pretrained model. To address this limitation, we propose an efficient video-to-4D object generation framework called Efficient4D. It generates high-quality spacetime-consistent images under different camera views, and then uses them as labeled data to directly train a novel 4D Gaussian splatting model with explicit point cloud geometry, enabling real-time rendering under continuous camera trajectories. Extensive experiments on synthetic and real videos show that Efficient4D offers a remarkable 10-fold increase in speed when compared to prior art alternatives while preserving the same level of innovative view synthesis quality. For example, Efficient4D takes only 14 minutes to model a dynamic object.
</details>

### 7. Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models
Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, Karsten Kreis

(NVIDIA, ETH Zurich, Max Planck Institute for Intelligent Systems)
<details span>
<summary><b>Abstract</b></summary>
Text-guided diffusion models have revolutionized image and video generation and have also been successfully used for optimization-based 3D object synthesis. Here, we instead focus on the underexplored text-to-4D setting and synthesize dynamic, animated 3D objects using score distillation methods with an additional temporal dimension. Compared to previous work, we pursue a novel compositional generation-based approach, and combine text-to-image, text-to-video, and 3D-aware multiview diffusion models to provide feedback during 4D object optimization, thereby simultaneously enforcing temporal consistency, high-quality visual appearance and realistic geometry. Our method, called Align Your Gaussians (AYG), leverages dynamic 3D Gaussian Splatting with deformation fields as 4D representation. Crucial to AYG is a novel method to regularize the distribution of the moving 3D Gaussians and thereby stabilize the optimization and induce motion. We also propose a motion amplification mechanism as well as a new autoregressive synthesis scheme to generate and combine multiple 4D sequences for longer generation. These techniques allow us to synthesize vivid dynamic scenes, outperform previous work qualitatively and quantitatively and achieve state-of-the-art text-to-4D performance. Due to the Gaussian 4D representation, different 4D animations can be seamlessly combined, as we demonstrate. AYG opens up promising avenues for animation, simulation and digital content creation as well as synthetic data generation.
</details>

### 8. Control4D: Efficient 4D Portrait Editing with Text
Ruizhi Shao, Jingxiang Sun, Cheng Peng, Zerong Zheng, Boyao Zhou, Hongwen Zhang, Yebin Liu (Tsinghua University)
<details span>
<summary><b>Abstract</b></summary>
We introduce Control4D, an innovative framework for editing dynamic 4D portraits using text instructions. Our method addresses the prevalent challenges in 4D editing, notably the inefficiencies of existing 4D representations and the inconsistent editing effect caused by diffusion-based editors. We first propose GaussianPlanes, a novel 4D representation that makes Gaussian Splatting more structured by applying plane-based decomposition in 3D space and time. This enhances both efficiency and robustness in 4D editing. Furthermore, we propose to leverage a 4D generator to learn a more continuous generation space from inconsistent edited images produced by the diffusion-based editor, which effectively improves the consistency and quality of 4D editing. Comprehensive evaluation demonstrates the superiority of Control4D, including significantly reduced training time, high-quality rendering, and spatial-temporal consistency in 4D portrait editing.
</details>

### 9. DreamGaussian4D: Generative 4D Gaussian Splatting
Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, Ziwei Liu

(S-Lab, Nanyang Technological University, Shanghai AI Laboratory, Peking University, University of Michigan)
<details span>
<summary><b>Abstract</b></summary>
Remarkable progress has been made in 4D content generation recently. However, existing methods suffer from long optimization time, lack of motion controllability, and a low level of detail. In this paper, we introduce DreamGaussian4D, an efficient 4D generation framework that builds on 4D Gaussian Splatting representation. Our key insight is that the explicit modeling of spatial transformations in Gaussian Splatting makes it more suitable for the 4D generation setting compared with implicit representations. DreamGaussian4D reduces the optimization time from several hours to just a few minutes, allows flexible control of the generated 3D motion, and produces animated meshes that can be efficiently rendered in 3D engines.
</details>



### Text to 4D Paper lists
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **Text-To-4D Dynamic Scene Generation**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2301.11280)          | -  | [Link](https://make-a-video3d.github.io/)  |
| 2023 | **4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2311.17984)          | [link](https://github.com/sherwinbahmani/4dfy)  | [Link](https://sherwinbahmani.github.io/4dfy/)  |
| 2023 | **A Unified Approach for Text- and Image-guided 4D Scene Generation**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2311.16854)          | -  | [Link](https://research.nvidia.com/labs/nxp/dream-in-4d/)  |
| 2023 | **Animate124: Animating One Image to 4D Dynamic Scene**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2311.14603)          | [link](https://github.com/HeliosZhao/Animate124)  | [Link](https://animate124.github.io/)  |
| 2023 | **4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2312.17225)          | [link](https://github.com/VITA-Group/4DGen)  | [Link](https://vita-group.github.io/4DGen/)  |
| 2024 | **Fast Dynamic 3D Object Generation from a Single-view Video**  | Arxiv 2024 |          [Link](https://arxiv.org/abs/2401.08742)          | [link](https://github.com/fudan-zvg/Efficient4D)  | [Link](https://fudan-zvg.github.io/Efficient4D/)  |
| 2023 | **Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2312.13763)          | [link](https://github.com/fudan-zvg/Efficient4D)  | [Link](https://research.nvidia.com/labs/toronto-ai/AlignYourGaussians/)  |
| 2023 | **Control4D: Efficient 4D Portrait Editing with Text**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2305.20082)          | -  | [Link](https://control4darxiv.github.io./)  |
| 2023 | **DreamGaussian4D:Generative 4D Gaussian Splatting**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2312.17142)          | [link](https://github.com/jiawei-ren/dreamgaussian4d)  | [Link](https://jiawei-ren.github.io/projects/dreamgaussian4d/)  |

### Text to 4D Reference


<details close>
<summary>Text to 4D</summary>

```
%text to 4D

@article{singer2023text4d,
  author = {Singer, Uriel and Sheynin, Shelly and Polyak, Adam and Ashual, Oron and
           Makarov, Iurii and Kokkinos, Filippos and Goyal, Naman and Vedaldi, Andrea and
           Parikh, Devi and Johnson, Justin and Taigman, Yaniv},
  title = {Text-To-4D Dynamic Scene Generation},
  journal = {arXiv:2301.11280},
  year = {2023},
}

@article{bah20234dfy,
  author = {Bahmani, Sherwin and Skorokhodov, Ivan and Rong, Victor and Wetzstein, Gordon and Guibas, Leonidas and Wonka, Peter and Tulyakov, Sergey and Park, Jeong Joon and Tagliasacchi, Andrea and Lindell, David B.},
  title = {4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling},
  journal = {arXiv},
  year = {2023},
}

@article{zheng2023unified,
  title={A Unified Approach for Text- and Image-guided 4D Scene Generation},
  author={Yufeng Zheng and Xueting Li and Koki Nagano and Sifei Liu and Karsten Kreis and Otmar Hilliges and Shalini De Mello},
  journal = {arXiv:2311.16854},
  year={2023}
}

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

@article{pan2024fast,
  title={Fast Dynamic 3D Object Generation from a Single-view Video},
  author={Pan, Zijie and Yang, Zeyu and Zhu, Xiatian and Zhang, Li},
  journal={arXiv preprint arXiv 2401.08742},
  year={2024}
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
booktitle = {arxiv},
year = {2023}
}

@article{ren2023dreamgaussian4d,
  title={DreamGaussian4D: Generative 4D Gaussian Splatting},
  author={Ren, Jiawei and Pan, Liang and Tang, Jiaxiang and Zhang, Chi and Cao, Ang and Zeng, Gang and Liu, Ziwei},
  journal={arXiv preprint arXiv:xxxx.xxxx},
  year={2023}
}
```
</details>

--------------

## Text to Scene
### 1. SceneScape: Text-Driven Consistent Scene Generation 
Rafail Fridman, Amit Abecasis, Yoni Kasten, Tali Dekel (Weizmann Institute of Science, NVIDIA Research)
<details span>
<summary><b>Abstract</b></summary>
We present a method for text-driven perpetual view generation -- synthesizing long-term videos of various scenes solely, given an input text prompt describing the scene and camera poses. We introduce a novel framework that generates such videos in an online fashion by combining the generative power of a pre-trained text-to-image model with the geometric priors learned by a pre-trained monocular depth prediction model. To tackle the pivotal challenge of achieving 3D consistency, i.e., synthesizing videos that depict geometrically-plausible scenes, we deploy an online test-time training to encourage the predicted depth map of the current frame to be geometrically consistent with the synthesized scene. The depth maps are used to construct a unified mesh representation of the scene, which is progressively constructed along the video generation process. In contrast to previous works, which are applicable only to limited domains, our method generates diverse scenes, such as walkthroughs in spaceships, caves, or ice castles.
</details>

### 2. Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models 
Lukas Höllein, Ang Cao, Andrew Owens, Justin Johnson, Matthias Nießner (Technical University of Munich, University of Michigan)
<details span>
<summary><b>Abstract</b></summary>
We present Text2Room, a method for generating room-scale textured 3D meshes from a given text prompt as input. To this end, we leverage pre-trained 2D text-to-image models to synthesize a sequence of images from different poses. In order to lift these outputs into a consistent 3D scene representation, we combine monocular depth estimation with a text-conditioned inpainting model. The core idea of our approach is a tailored viewpoint selection such that the content of each image can be fused into a seamless, textured 3D mesh. More specifically, we propose a continuous alignment strategy that iteratively fuses scene frames with the existing geometry to create a seamless mesh. Unlike existing works that focus on generating single objects or zoom-out trajectories from text, our method generates complete 3D scenes with multiple objects and explicit 3D geometry. We evaluate our approach using qualitative and quantitative metrics, demonstrating it as the first method to generate room-scale 3D geometry with compelling textures from only text as input.
</details>

### 3. Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints 
Chuan Fang, Xiaotao Hu, Kunming Luo, Ping Tan 

(Hong Kong University of Science and Technology, Light Illusions, Nankai University)
<details span>
<summary><b>Abstract</b></summary>
Text-driven 3D indoor scene generation could be useful for gaming, film industry, and AR/VR applications. However, existing methods cannot faithfully capture the room layout, nor do they allow flexible editing of individual objects in the room. To address these problems, we present Ctrl-Room, which is able to generate convincing 3D rooms with designer-style layouts and high-fidelity textures from just a text prompt. Moreover, Ctrl-Room enables versatile interactive editing operations such as resizing or moving individual furniture items. Our key insight is to separate the modeling of layouts and appearance. %how to model the room that takes into account both scene texture and geometry at the same time. To this end, Our proposed method consists of two stages, a `Layout Generation Stage' and an `Appearance Generation Stage'. The `Layout Generation Stage' trains a text-conditional diffusion model to learn the layout distribution with our holistic scene code parameterization. Next, the `Appearance Generation Stage' employs a fine-tuned ControlNet to produce a vivid panoramic image of the room guided by the 3D scene layout and text prompt. In this way, we achieve a high-quality 3D room with convincing layouts and lively textures. Benefiting from the scene code parameterization, we can easily edit the generated room model through our mask-guided editing module, without expensive editing-specific training. Extensive experiments on the Structured3D dataset demonstrate that our method outperforms existing methods in producing more reasonable, view-consistent, and editable 3D rooms from natural language prompts.
</details>

### 4. ShowRoom3D: Text to High-Quality 3D Room Generation Using 3D Priors
Weijia Mao, Yan-Pei Cao, Jia-Wei Liu, Zhongcong Xu, Mike Zheng Shou

(Show Lab National University of Singapore, ARC Lab Tencent PCG)
<details span>
<summary><b>Abstract</b></summary>
We introduce ShowRoom3D, a three-stage approach for generating high-quality 3D room-scale scenes from texts. Previous methods using 2D diffusion priors to optimize neural radiance fields for generating room-scale scenes have shown unsatisfactory quality. This is primarily attributed to the limitations of 2D priors lacking 3D awareness and constraints in the training methodology. In this paper, we utilize a 3D diffusion prior, MVDiffusion, to optimize the 3D room-scale scene. Our contributions are in two aspects. Firstly, we propose a progressive view selection process to optimize NeRF. This involves dividing the training process into three stages, gradually expanding the camera sampling scope. Secondly, we propose the pose transformation method in the second stage. It will ensure MVDiffusion provide the accurate view guidance. As a result, ShowRoom3D enables the generation of rooms with improved structural integrity, enhanced clarity from any view, reduced content repetition, and higher consistency across different perspectives. Extensive experiments demonstrate that our method, significantly outperforms state-of-the-art approaches by a large margin in terms of user study.
</details>

### 5. SceneWiz3D: Towards Text-guided 3D Scene Composition 
Qihang Zhang, Chaoyang Wang, Aliaksandr Siarohin, Peiye Zhuang, Yinghao Xu, Ceyuan Yang, Dahua Lin, Bolei Zhou, Sergey Tulyakov, Hsin-Ying Lee

(The Chinese University of Hong Kong, Snap Inc., Stanford University, University of California Los Angeles)
<details span>
<summary><b>Abstract</b></summary>
We are witnessing significant breakthroughs in the technology for generating 3D objects from text. Existing approaches either leverage large text-to-image models to optimize a 3D representation or train 3D generators on object-centric datasets. Generating entire scenes, however, remains very challenging as a scene contains multiple 3D objects, diverse and scattered. In this work, we introduce SceneWiz3D, a novel approach to synthesize high-fidelity 3D scenes from text. We marry the locality of objects with globality of scenes by introducing a hybrid 3D representation: explicit for objects and implicit for scenes. Remarkably, an object, being represented explicitly, can be either generated from text using conventional text-to-3D approaches, or provided by users. To configure the layout of the scene and automatically place objects, we apply the Particle Swarm Optimization technique during the optimization process. Furthermore, it is difficult for certain parts of the scene (e.g., corners, occlusion) to receive multi-view supervision, leading to inferior geometry. We incorporate an RGBD panorama diffusion model to mitigate it, resulting in high-quality geometry. Extensive evaluation supports that our approach achieves superior quality over previous approaches, enabling the generation of detailed and view-consistent 3D scenes.
</details>

### 6. Detailed Human-Centric Text Description-Driven Large Scene Synthesis
Gwanghyun Kim, Dong Un Kang, Hoigi Seo, Hayeon Kim, Se Young Chun

(Dept. of Electrical and Computer Engineering, INMC & IPAI, Seoul National University Republic of Korea)
<details span>
<summary><b>Abstract</b></summary>
Text-driven large scene image synthesis has made significant progress with diffusion models, but controlling it is challenging. While using additional spatial controls with corresponding texts has improved the controllability of large scene synthesis, it is still challenging to faithfully reflect detailed text descriptions without user-provided controls. Here, we propose DetText2Scene, a novel text-driven large-scale image synthesis with high faithfulness, controllability, and naturalness in a global context for the detailed human-centric text description. Our DetText2Scene consists of 1) hierarchical keypoint-box layout generation from the detailed description by leveraging large language model (LLM), 2) view-wise conditioned joint diffusion process to synthesize a large scene from the given detailed text with LLM-generated grounded keypoint-box layout and 3) pixel perturbation-based pyramidal interpolation to progressively refine the large scene for global coherence. Our DetText2Scene significantly outperforms prior arts in text-to-large scene synthesis qualitatively and quantitatively, demonstrating strong faithfulness with detailed descriptions, superior controllability, and excellent naturalness in a global context.
</details>

### 7. Text2Immersion: Generative Immersive Scene with 3D Gaussians
Hao Ouyang, Kathryn Heal, Stephen Lombardi, Tiancheng Sun (HKUST, Google)
<details span>
<summary><b>Abstract</b></summary>
We introduce Text2Immersion, an elegant method for producing high-quality 3D immersive scenes from text prompts. Our proposed pipeline initiates by progressively generating a Gaussian cloud using pre-trained 2D diffusion and depth estimation models. This is followed by a refining stage on the Gaussian cloud, interpolating and refining it to enhance the details of the generated scene. Distinct from prevalent methods that focus on single object or indoor scenes, or employ zoom-out trajectories, our approach generates diverse scenes with various objects, even extending to the creation of imaginary scenes. Consequently, Text2Immersion can have wide-ranging implications for various applications such as virtual reality, game development, and automated content creation. Extensive evaluations demonstrate that our system surpasses other methods in rendering quality and diversity, further progressing towards text-driven 3D scene generation. 
</details>

### 8. ReplaceAnything3D:Text-Guided 3D Scene Editing with Compositional Neural Radiance Fields
Edward Bartrum, Thu Nguyen-Phuoc, Chris Xie, Zhengqin Li, Numair Khan, Armen Avetisyan, Douglas Lanman, Lei Xiao

(University College London, Alan Turing Institute, Reality Labs Research Meta)
<details span>
<summary><b>Abstract</b></summary>
We introduce ReplaceAnything3D model (RAM3D), a novel text-guided 3D scene editing method that enables the replacement of specific objects within a scene. Given multi-view images of a scene, a text prompt describing the object to replace, and a text prompt describing the new object, our Erase-and-Replace approach can effectively swap objects in the scene with newly generated content while maintaining 3D consistency across multiple viewpoints. We demonstrate the versatility of ReplaceAnything3D by applying it to various realistic 3D scenes, showcasing results of modified foreground objects that are well-integrated with the rest of the scene without affecting its overall integrity.
</details>

### 9. GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guidedGenerative Gaussian Splatting
Xiaoyu Zhou, Xingjian Ran, Yajiao Xiong, Jinlin He, Zhiwei Lin, Yongtao Wang, Deqing Sun, Ming-Hsuan Yang

(Wangxuan Institute of Computer Technology Peking University, Google Research, University of California Merced)
<details span>
<summary><b>Abstract</b></summary>
We present GALA3D, generative 3D GAussians with LAyout-guided control, for effective compositional text-to-3D generation. We first utilize large language models (LLMs) to generate the initial layout and introduce a layout-guided 3D Gaussian representation for 3D content generation with adaptive geometric constraints. We then propose an object-scene compositional optimization mechanism with conditioned diffusion to collaboratively generate realistic 3D scenes with consistent geometry, texture, scale, and accurate interactions among multiple objects while simultaneously adjusting the coarse layout priors extracted from the LLMs to align with the generated scene. Experiments show that GALA3D is a user-friendly, end-to-end framework for state-of-the-art scene-level 3D content generation and controllable editing while ensuring the high fidelity of object-level entities within the scene. 
</details>

### Text to Scene Paper lists
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **SceneScape: Text-Driven Consistent Scene Generation**  | NeurIPS 2023 |          [Link](https://arxiv.org/abs/2302.01133)          | [Link](https://github.com/RafailFridman/SceneScape)  | [Link](https://scenescape.github.io/)  |
| 2023 | **Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models**  |  ICCV 2023 (Oral) |          [Link](https://arxiv.org/abs/2303.11989)          | [Link](https://github.com/lukasHoel/text2room)  | [Link](https://lukashoel.github.io/text-to-room/)  |
| 2023 | **Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2310.03602)          | Coming soon!   | [Link](https://fangchuan.github.io/ctrl-room.github.io/)  |
| 2023 | **ShowRoom3D: Text to High-Quality 3D Room Generation Using 3D Priors**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2312.13324)          | [Link](https://github.com/showlab/ShowRoom3D)  | [Link](https://showroom3d.github.io/)  |
| 2023 | **SceneWiz3D: Towards Text-guided 3D Scene Composition**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2312.08885)          | [Link](https://github.com/zqh0253/SceneWiz3D)   | [Link](https://zqh0253.github.io/SceneWiz3D/)  |
| 2023 | **Detailed Human-Centric Text Description-Driven Large Scene Synthesis**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2311.18654)          | --  |-- |
| 2023 | **Text2Immersion: Generative Immersive Scene with 3D Gaussians**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2312.09242)          | Coming soon!   | [Link](https://ken-ouyang.github.io/text2immersion/index.html)  |
| 2024 | **ReplaceAnything3D:Text-Guided 3D Scene Editing with Compositional Neural Radiance Fields**  | Arxiv 2024 |          [Link](https://arxiv.org/abs/2401.17895)          | --  |[Link](https://replaceanything3d.github.io/) |
| 2024 | **GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guidedGenerative Gaussian Splatting**  | Arxiv 2024 |          [Link](https://arxiv.org/abs/2402.07207)          | [Link](https://github.com/VDIGPKU/GALA3D)  |[Link](https://gala3d.github.io/) |

### Text to Scene Reference
<details close>
<summary>Text to Scene</summary>

```
% text to scene

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

@article{fang2023ctrl,
      title={Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints},
      author={Fang, Chuan and Hu, Xiaotao and Luo, Kunming and Tan, Ping},
      journal={arXiv preprint arXiv:2310.03602},
      year={2023}
}

@article{mao2023showroom3d,
  title={ShowRoom3D: Text to High-Quality 3D Room Generation Using 3D Priors},
  author={Mao, Weijia and Cao, Yan-Pei and Liu, Jia-Wei and Xu, Zhongcong and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2312.13324},
  year={2023}
}

@inproceedings{zhang2023scenewiz3d,
        author = {Qihang Zhang and Chaoyang Wang and Aliaksandr Siarohin and Peiye Zhuang and Yinghao Xu and Ceyuan Yang and Dahua Lin and Bo Dai and Bolei Zhou and Sergey Tulyakov and Hsin-Ying Lee},
        title = {{SceneWiz3D}: Towards Text-guided {3D} Scene Composition},
        booktitle = {arXiv},
        year = {2023}
}

@article{kim2023detailed,
  title={Detailed Human-Centric Text Description-Driven Large Scene Synthesis},
  author={Kim, Gwanghyun and Kang, Dong Un and Seo, Hoigi and Kim, Hayeon and Chun, Se Young},
  journal={arXiv preprint arXiv:2311.18654},
  year={2023}
}

@article{ouyang2023text,
  author    = {Ouyang, Hao and Sun, Tiancheng and Lombardi, Stephen and Heal, Kathryn},
  title     = {Text2Immersion: Generative Immersive Scene with 3D Gaussians},
  journal   = {Arxiv},
  year      = {2023},
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

@article{zhou2024gala3d,
  title={GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting},
  author={Zhou, Xiaoyu and Ran, Xingjian and Xiong, Yajiao and He, Jinlin and Lin, Zhiwei and Wang, Yongtao and Sun, Deqing and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2402.07207},
  year={2024}
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

### 7. Guide3D: Create 3D Avatars from Text and Image Guidance  
Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, Kwan-Yee K. Wong (HKU, ARC Lab Tencent PCG)
<details span>
<summary><b>Abstract</b></summary>
Recently, text-to-image generation has exhibited remarkable advancements, with the ability to produce visually impressive results. In contrast, text-to-3D generation has not yet reached a comparable level of quality. Existing methods primarily rely on text-guided score distillation sampling (SDS), and they encounter difficulties in transferring 2D attributes of the generated images to 3D content. In this work, we aim to develop an effective 3D generative model capable of synthesizing high-resolution textured meshes by leveraging both textual and image information. To this end, we introduce Guide3D, a zero-shot text-and-image-guided generative model for 3D avatar generation based on diffusion models. Our model involves (1) generating sparse-view images of a text-consistent character using diffusion models, and (2) jointly optimizing multi-resolution differentiable marching tetrahedral grids with pixel-aligned image features. We further propose a similarity-aware feature fusion strategy for efficiently integrating features from different views. Moreover, we introduce two novel training objectives as an alternative to calculating SDS, significantly enhancing the optimization process. We thoroughly evaluate the performance and components of our framework, which outperforms the current state-of-the-art in producing topologically and structurally correct geometry and high-resolution textures. Guide3D enables the direct transfer of 2D-generated images to the 3D space. Our code will be made publicly available.
</details>

### 8. AvatarVerse: High-quality & Stable 3D Avatar Creation from Text and Pose  
Huichao Zhang, Bowen Chen, Hao Yang, Liao Qu, Xu Wang, Li Chen, Chao Long, Feida Zhu, Kang Du, Min Zheng (ByteDance, CMU)
<details span>
<summary><b>Abstract</b></summary>
Creating expressive, diverse and high-quality 3D avatars from highly customized text descriptions and pose guidance is a challenging task, due to the intricacy of modeling and texturing in 3D that ensure details and various styles (realistic, fictional, etc). We present AvatarVerse, a stable pipeline for generating expressive high-quality 3D avatars from nothing but text descriptions and pose guidance. In specific, we introduce a 2D diffusion model conditioned on DensePose signal to establish 3D pose control of avatars through 2D images, which enhances view consistency from partially observed scenarios. It addresses the infamous Janus Problem and significantly stablizes the generation process. Moreover, we propose a progressive high-resolution 3D synthesis strategy, which obtains substantial improvement over the quality of the created 3D avatars. To this end, the proposed AvatarVerse pipeline achieves zero-shot 3D modeling of 3D avatars that are not only more expressive, but also in higher quality and fidelity than previous works. Rigorous qualitative evaluations and user studies showcase AvatarVerse's superiority in synthesizing high-fidelity 3D avatars, leading to a new standard in high-quality and stable 3D avatar creation.
</details>

### 9. AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars 
Fangzhou Hong, Mingyuan Zhang, Liang Pan, Zhongang Cai, Lei Yang, Ziwei Liu 

(S-Lab NTU, SenseTime Research, Shanghai AI Laboratory)
<details span>
<summary><b>Abstract</b></summary>
3D avatar creation plays a crucial role in the digital age. However, the whole production process is prohibitively time-consuming and labor-intensive. To democratize this technology to a larger audience, we propose AvatarCLIP, a zero-shot text-driven framework for 3D avatar generation and animation. Unlike professional software that requires expert knowledge, AvatarCLIP empowers layman users to customize a 3D avatar with the desired shape and texture, and drive the avatar with the described motions using solely natural languages. Our key insight is to take advantage of the powerful vision-language model CLIP for supervising neural human generation, in terms of 3D geometry, texture and animation. Specifically, driven by natural language descriptions, we initialize 3D human geometry generation with a shape VAE network. Based on the generated 3D human shapes, a volume rendering model is utilized to further facilitate geometry sculpting and texture generation. Moreover, by leveraging the priors learned in the motion VAE, a CLIP-guided reference-based motion synthesis method is proposed for the animation of the generated 3D avatar. Extensive qualitative and quantitative experiments validate the effectiveness and generalizability of AvatarCLIP on a wide range of avatars. Remarkably, AvatarCLIP can generate unseen 3D avatars with novel animations, achieving superior zero-shot capability.
</details>

### 10. SEEAvatar: Photorealistic Text-to-3D Avatar Generation with Constrained Geometry and Appearance
Yuanyou Xu, Zongxin Yang, Yi Yang

(ReLER, CCAI, Zhejiang University)
<details span>
<summary><b>Abstract</b></summary>
Powered by large-scale text-to-image generation models, text-to-3D avatar generation has made promising progress. However, most methods fail to produce photorealistic results, limited by imprecise geometry and low-quality appearance. Towards more practical avatar generation, we present SEEAvatar, a method for generating photorealistic 3D avatars from text with SElf-Evolving constraints for decoupled geometry and appearance. For geometry, we propose to constrain the optimized avatar in a decent global shape with a template avatar. The template avatar is initialized with human prior and can be updated by the optimized avatar periodically as an evolving template, which enables more flexible shape generation. Besides, the geometry is also constrained by the static human prior in local parts like face and hands to maintain the delicate structures. For appearance generation, we use diffusion model enhanced by prompt engineering to guide a physically based rendering pipeline to generate realistic textures. The lightness constraint is applied on the albedo texture to suppress incorrect lighting effect. Experiments show that our method outperforms previous methods on both global and local geometry and appearance quality by a large margin. Since our method can produce high-quality meshes and textures, such assets can be directly applied in classic graphics pipeline for realistic rendering under any lighting condition.
</details>

### 11. HeadArtist: Text-conditioned 3D Head Generation with Self Score Distillation
Hongyu Liu, Xuan Wang, Ziyu Wan, Yujun Shen, Yibing Song, Jing Liao, Qifeng Chen

(HKUST, Ant Group, City University of HongKong AI Institute, Fudan University)
<details span>
<summary><b>Abstract</b></summary>
This work presents HeadArtist for 3D head generation from text descriptions. With a landmark-guided ControlNet serving as the generative prior, we come up with an efficient pipeline that optimizes a parameterized 3D head model under the supervision of the prior distillation itself. We call such a process self score distillation (SSD). In detail, given a sampled camera pose, we first render an image and its corresponding landmarks from the head model, and add some particular level of noise onto the image. The noisy image, landmarks, and text condition are then fed into the frozen ControlNet twice for noise prediction. Two different classifier-free guidance (CFG) weights are applied during these two predictions, and the prediction difference offers a direction on how the rendered image can better match the text of interest. Experimental results suggest that our approach delivers high-quality 3D head sculptures with adequate geometry and photorealistic appearance, significantly outperforming state-ofthe-art methods. We also show that the same pipeline well supports editing the generated heads, including both geometry deformation and appearance change.
</details>

### 12. HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting
Zhenglin Zhou, Fan Ma, Hehe Fan, Yi Yang (ReLER, CCAI, Zhejiang University)
<details span>
<summary><b>Abstract</b></summary>
Creating digital avatars from textual prompts has long been a desirable yet challenging task. Despite the promising outcomes obtained through 2D diffusion priors in recent works, current methods face challenges in achieving high-quality and animated avatars effectively. In this paper, we present HeadStudio, a novel framework that utilizes 3D Gaussian splatting to generate realistic and animated avatars from text prompts. Our method drives 3D Gaussians semantically to create a flexible and achievable appearance through the intermediate FLAME representation. Specifically, we incorporate the FLAME into both 3D representation and score distillation: 1) FLAME-based 3D Gaussian splatting, driving 3D Gaussian points by rigging each point to a FLAME mesh. 2) FLAME-based score distillation sampling, utilizing FLAME-based fine-grained control signal to guide score distillation from the text prompt. Extensive experiments demonstrate the efficacy of HeadStudio in generating animatable avatars from textual prompts, exhibiting visually appealing appearances. The avatars are capable of rendering high-quality real-time (≥40 fps) novel views at a resolution of 1024. They can be smoothly controlled by real-world speech and video. We hope that HeadStudio can advance digital avatar creation and that the present method can widely be applied across various domains.
</details>

### 13. En3D: An Enhanced Generative Model for Sculpting 3D Humans from 2D Synthetic Data
Yifang Men, Biwen Lei, Yuan Yao, Miaomiao Cui, Zhouhui Lian, Xuansong Xie

(Institute for Intelligent Computing Alibaba Group, Peking University)
<details span>
<summary><b>Abstract</b></summary>
We present En3D, an enhanced generative scheme for sculpting high-quality 3D human avatars. Unlike previous works that rely on scarce 3D datasets or limited 2D collections with imbalanced viewing angles and imprecise pose priors, our approach aims to develop a zero-shot 3D generative scheme capable of producing visually realistic, geometrically accurate and content-wise diverse 3D humans without relying on pre-existing 3D or 2D assets. To address this challenge, we introduce a meticulously crafted workflow that implements accurate physical modeling to learn the enhanced 3D generative model from synthetic 2D data. During inference, we integrate optimization modules to bridge the gap between realistic appearances and coarse 3D shapes. Specifically, En3D comprises three modules: a 3D generator that accurately models generalizable 3D humans with realistic appearance from synthesized balanced, diverse, and structured human images; a geometry sculptor that enhances shape quality using multi-view normal constraints for intricate human anatomy; and a texturing module that disentangles explicit texture maps with fidelity and editability, leveraging semantical UV partitioning and a differentiable rasterizer. Experimental results show that our approach significantly outperforms prior works in terms of image quality, geometry accuracy and content diversity. We also showcase the applicability of our generated avatars for animation and editing, as well as the scalability of our approach for content-style free adaptation.
</details>

### Text to 3D Human Paper lists
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting**  | arXiv  |          [Link](https://arxiv.org/abs/2311.17061)          | [Link](https://github.com/alvinliu0/HumanGaussian)  | [Link](https://alvinliu0.github.io/projects/HumanGaussian)  | 
| 2023 | **HumanNorm: Learning Normal Diffusion Model for High-quality and Realistic 3D Human Generation**  | arXiv  |          [Link](https://arxiv.org/abs/2310.01406)          | [Link](https://github.com/xhuangcv/humannorm)  | [Link](https://humannorm.github.io/)  |
| 2023 | **TeCH: Text-guided Reconstruction of Lifelike Clothed Humans**  | 3DV 2024  |          [Link](https://arxiv.org/abs/2308.08545)          | [Link](https://github.com/huangyangyi/TeCH)  | [Link](https://huangyangyi.github.io/TeCH/)  |
| 2023 | **TADA! Text to Animatable Digital Avatars**  | 3DV 2024  |          [Link](https://arxiv.org/abs/2308.10899)          | [Link](https://github.com/TingtingLiao/TADA)  | [Link](https://tada.is.tue.mpg.de/)  |
| 2023 | **DreamWaltz: Make a Scene with Complex 3D Animatable Avatars**  | NeurIPS 2023  |          [Link](https://arxiv.org/abs/2305.12529)          | [Link](https://github.com/IDEA-Research/DreamWaltz)  | [Link](https://idea-research.github.io/DreamWaltz/)  |
| 2023 | **DreamHuman: Animatable 3D Avatars from Text**  | arXiv  |          [Link](https://arxiv.org/abs/2306.09329)          |  Coming soon!  | [Link](https://dream-human.github.io/)  |
| 2023 | **Guide3D: Create 3D Avatars from Text and Image Guidance**  | arXiv  |          [Link](https://arxiv.org/abs/2308.09705)          |  [Link](https://github.com/yukangcao/Guide3D) | -- |
| 2023 | **AvatarVerse: High-quality & Stable 3D Avatar Creation from Text and Pose**  | AAAI2024  |          [Link](https://arxiv.org/abs/2308.03610)          |  [Link](https://github.com/bytedance/AvatarVerse)  | [Link](https://avatarverse3d.github.io/)  |
| 2022 | **AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars**  | SIGGRAPH 2022 (Journal Track)  |          [Link](https://arxiv.org/abs/2205.08535)          | [Link](https://github.com/hongfz16/AvatarCLIP)  | [Link](https://hongfz16.github.io/projects/AvatarCLIP.html)  |
| 2023 | **SEEAvatar: Photorealistic Text-to-3D Avatar Generation with Constrained Geometry and Appearance**  | arXiv  |          [Link](https://arxiv.org/abs/2312.08889)          | Coming Soon!  | [Link](https://seeavatar3d.github.io/)  |
| 2023 | **HeadArtist: Text-conditioned 3D Head Generation with Self Score Distillation**  | arXiv  |          [Link](https://arxiv.org/abs/2312.07539)          |  [Link](https://github.com/KumapowerLIU/HeadArtist)  | [Link](https://kumapowerliu.github.io/HeadArtist/)  |
| 2024 | **HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting**  | arXiv 2024  |          [Link](https://arxiv.org/abs/2402.06149)          |  [Link](https://github.com/ZhenglinZhou/HeadStudio/)  | [Link](https://zhenglinzhou.github.io/HeadStudio-ProjectPage/)  |
| 2024 | **En3D: An Enhanced Generative Model for Sculpting 3D Humans from 2D Synthetic Data**  | arXiv 2024  |          [Link](https://arxiv.org/abs/2401.01173)          |  [Link](https://github.com/menyifang/En3D)  | [Link](https://menyifang.github.io/projects/En3D/index.html)  |

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

### 2. EMDM: Efficient Motion Diffusion Model for Fast, High-Quality Human Motion Generation  
Wenyang Zhou, Zhiyang Dou, Zeyu Cao, Zhouyingcheng Liao, Jingbo Wang, Wenjia Wang, Yuan Liu, Taku Komura, Wenping Wang, Lingjie Liu
(University of Cambridge, University of Hong Kong, Shanghai AI Laboratory, Texas A&M University, University of Pennsylvania)
<details span>
<summary><b>Abstract</b></summary>
We introduce Efficient Motion Diffusion Model (EMDM) for fast and high-quality human motion generation. Although previous motion diffusion models have shown impressive results, they struggle to achieve fast generation while maintaining high-quality human motions. Motion latent diffusion has been proposed for efficient motion generation. However, effectively learning a latent space can be non-trivial in such a two-stage manner. Meanwhile, accelerating motion sampling by increasing the step size, e.g., DDIM, typically leads to a decline in motion quality due to the inapproximation of complex data distributions when naively increasing the step size. In this paper, we propose EMDM that allows for much fewer sample steps for fast motion generation by modeling the complex denoising distribution during multiple sampling steps. Specifically, we develop a Conditional Denoising Diffusion GAN to capture multimodal data distributions conditioned on both control signals, i.e., textual description and denoising time step. By modeling the complex data distribution, a larger sampling step size and fewer steps are achieved during motion synthesis, significantly accelerating the generation process. To effectively capture the human dynamics and reduce undesired artifacts, we employ motion geometric loss during network training, which improves the motion quality and training efficiency. As a result, EMDM achieves a remarkable speed-up at the generation stage while maintaining high-quality motion generation in terms of fidelity and diversity.
</details>

### 3. SinMDM: Single Motion Diffusion 
Sigal Raab, Inbal Leibovitch, Guy Tevet, Moab Arar, Amit H. Bermano, Daniel Cohen-Or
(Tel Aviv University, Israel)
<details span>
<summary><b>Abstract</b></summary>
Synthesizing realistic animations of humans, animals, and even imaginary creatures, has long been a goal for artists and computer graphics professionals. Compared to the imaging domain, which is rich with large available datasets, the number of data instances for the motion domain is limited, particularly for the animation of animals and exotic creatures (e.g., dragons), which have unique skeletons and motion patterns. In this work, we present a Single Motion Diffusion Model, dubbed SinMDM, a model designed to learn the internal motifs of a single motion sequence with arbitrary topology and synthesize motions of arbitrary length that are faithful to them. We harness the power of diffusion models and present a denoising network explicitly designed for the task of learning from a single input motion. SinMDM is designed to be a lightweight architecture, which avoids overfitting by using a shallow network with local attention layers that narrow the receptive field and encourage motion diversity. SinMDM can be applied in various contexts, including spatial and temporal in-betweening, motion expansion, style transfer, and crowd animation. Our results show that SinMDM outperforms existing methods both in quality and time-space efficiency. Moreover, while current approaches require additional training for different applications, our work facilitates these applications at inference time.
</details>

### 4. MDM: Human Motion Diffusion Model 
Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, Amit H. Bermano (Tel Aviv University, Israel)
<details span>
<summary><b>Abstract</b></summary>
Natural and expressive human motion generation is the holy grail of computer animation. It is a challenging task, due to the diversity of possible motion, human perceptual sensitivity to it, and the difficulty of accurately describing it. Therefore, current generative solutions are either low-quality or limited in expressiveness. Diffusion models, which have already shown remarkable generative capabilities in other domains, are promising candidates for human motion due to their many-to-many nature, but they tend to be resource hungry and hard to control. In this paper, we introduce Motion Diffusion Model (MDM), a carefully adapted classifier-free diffusion-based generative model for the human motion domain. MDM is transformer-based, combining insights from motion generation literature. A notable design-choice is the prediction of the sample, rather than the noise, in each diffusion step. This facilitates the use of established geometric losses on the locations and velocities of the motion, such as the foot contact loss. As we demonstrate, MDM is a generic approach, enabling different modes of conditioning, and different generation tasks. We show that our model is trained with lightweight resources and yet achieves state-of-the-art results on leading benchmarks for text-to-motion and action-to-motion.
</details>

### 5. MLD: Motion Latent Diffusion Models 
Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, Jingyi Yu, Gang Yu

(Fudan University, Tencent PCG, ShanghaiTech University)
<details span>
<summary><b>Abstract</b></summary>
We study a challenging task, conditional human motion generation, which produces plausible human motion sequences according to various conditional inputs, such as action classes or textual descriptors. Since human motions are highly diverse and have a property of quite different distribution from conditional modalities, such as textual descriptors in natural languages, it is hard to learn a probabilistic mapping from the desired conditional modality to the human motion sequences. Besides, the raw motion data from the motion capture system might be redundant in sequences and contain noises; directly modeling the joint distribution over the raw motion sequences and conditional modalities would need a heavy computational overhead and might result in artifacts introduced by the captured noises. To learn a better representation of the various human motion sequences, we first design a powerful Variational AutoEncoder (VAE) and arrive at a representative and low-dimensional latent code for a human motion sequence. Then, instead of using a diffusion model to establish the connections between the raw motion sequences and the conditional inputs, we perform a diffusion process on the motion latent space. Our proposed Motion Latent-based Diffusion model (MLD) could produce vivid motion sequences conforming to the given conditional inputs and substantially reduce the computational overhead in both the training and inference stages. Extensive experiments on various human motion generation tasks demonstrate that our MLD achieves significant improvements over the state-of-the-art methods among extensive human motion generation tasks, with two orders of magnitude faster than previous diffusion models on raw motion sequences.
</details>

### 6. HumanTOMATO: Text-aligned Whole-body Motion Generation 
Shunlin Lu*, Ling-Hao Chen*, Ailing Zeng, Jing Lin, Ruimao Zhang, Lei Zhang, Heung-Yeung Shum

(Tsinghua University, International Digital Economy Academy (IDEA), School of Data Science CUHK (SZ))
<details span>
<summary><b>Abstract</b></summary>
This work targets a novel text-driven whole-body motion generation task, which takes a given textual description as input and aims at generating high-quality, diverse, and coherent facial expressions, hand gestures, and body motions simultaneously. Previous works on text-driven motion generation tasks mainly have two limitations: they ignore the key role of fine-grained hand and face controlling in vivid whole-body motion generation, and lack a good alignment between text and motion. To address such limitations, we propose a Text-aligned whOle-body Motion generATiOn framework, named HumanTOMATO, which is the first attempt to our knowledge towards applicable holistic motion generation in this research area. To tackle this challenging task, our solution includes two key designs: (1) a Holistic Hierarchical VQ-VAE (aka H²VQ) and a Hierarchical-GPT for fine-grained body and hand motion reconstruction and generation with two structured codebooks; and (2) a pre-trained text-motion-alignment model to help generated motion align with the input textual description explicitly. Comprehensive experiments verify that our model has significant advantages in both the quality of generated motions and their alignment with text.
</details>

### 7. MotionGPT: Human Motion as a Foreign Language  
Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, Tao Chen

(Fudan University, Tencent PCG, ShanghaiTech University)
<details span>
<summary><b>Abstract</b></summary>
Though the advancement of pre-trained large language models unfolds, the exploration of building a unified model for language and other multimodal data, such as motion, remains challenging and untouched so far. Fortunately, human motion displays a semantic coupling akin to human language, often perceived as a form of body language. By fusing language data with large-scale motion models, motion-language pre-training that can enhance the performance of motion-related tasks becomes feasible. Driven by this insight, we propose MotionGPT, a unified, versatile, and user-friendly motion-language model to handle multiple motion-relevant tasks. Specifically, we employ the discrete vector quantization for human motionand transfer 3D motion into motion tokens, similar to the generation process ofword tokens. Building upon this “motion vocabulary”, we perform language modeling on both motion and text in a unified manner, treating human motion as a specific language. Moreover, inspired by prompt learning, we pre-train MotionGPT with a mixture of motion-language data and fine-tune it on prompt-based question-and-answer tasks. Extensive experiments demonstrate that MotionGPT achieves state-of-the-art performances on multiple motion tasks including text-driven motion generation, motion captioning, motion prediction, and motion in-between.
</details>

### 8. Story-to-Motion: Synthesizing Infinite and Controllable Character Animation from Long Text  
Zhongfei Qing, Zhongang Cai, Zhitao Yang, Lei Yang (SenseTime)
<details span>
<summary><b>Abstract</b></summary>
Generating natural human motion from a story has the potential to transform the landscape of animation, gaming, and film industries. A new and challenging task, Story-to-Motion, arises when characters are required to move to various locations and perform specific motions based on a long text description. This task demands a fusion of low-level control (trajectories) and high-level control (motion semantics). Previous works in character control and text-to-motion have addressed related aspects, yet a comprehensive solution remains elusive: character control methods do not handle text description, whereas text-to-motion methods lack position constraints and often produce unstable motions. In light of these limitations, we propose a novel system that generates controllable, infinitely long motions and trajectories aligned with the input text. 1) we leverage contemporary Large Language Models to act as a text-driven motion scheduler to extract a series of (text, position) pairs from long text. 2) we develop a text-driven motion retrieval scheme that incorporates classic motion matching with motion semantic and trajectory constraints. 3) we design a progressive mask transformer that addresses common artifacts in the transition motion such as unnatural pose and foot sliding. Beyond its pioneering role as the first comprehensive solution for Story-to-Motion, our system undergoes evaluation across three distinct sub-tasks: trajectory following, temporal action composition, and motion blending, where it outperforms previous state-of-the-art (SOTA) motion synthesis methods across the board.
</details>

### 9. Plan, Posture and Go: Towards Open-World Text-to-Motion Generation  
Jinpeng Liu, Wenxun Dai, Chunyu Wang, Yiji Cheng, Yansong Tang, Xin Tong 
(Shenzhen International Graudate School Tsinghua University, Microsoft Research Asia)
<details span>
<summary><b>Abstract</b></summary>
Conventional text-to-motion generation methods are usually trained on limited text-motion pairs, making them hard to generalize to open-world scenarios. Some works use the CLIP model to align the motion space and the text space, aiming to enable motion generation from natural language motion descriptions. However, they are still constrained to generate limited and unrealistic in-place motions. To address these issues, we present a divide-and-conquer framework named PRO-Motion, which consists of three modules as motion planner, posture-diffuser and go-diffuser. The motion planner instructs Large Language Models (LLMs) to generate a sequence of scripts describing the key postures in the target motion. Differing from natural languages, the scripts can describe all possible postures following very simple text templates. This significantly reduces the complexity of posture-diffuser, which transforms a script to a posture, paving the way for open-world generation. Finally, go-diffuser, implemented as another diffusion model, estimates whole-body translations and rotations for all postures, resulting in realistic motions. Experimental results have shown the superiority of our method with other counterparts, and demonstrated its capability of generating diverse and realistic motions from complex open-world prompts such as "Experiencing a profound sense of joy". 
</details>

### 10. MoMask: Generative Masked Modeling of 3D Human Motions  
Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, Li Cheng 
(University of Alberta, Canada)
<details span>
<summary><b>Abstract</b></summary>
We introduce MoMask, a novel masked modeling framework for text-driven 3D human motion generation. In MoMask, a hierarchical quantization scheme is employed to represent human motion as multi-layer discrete motion tokens with high-fidelity details. Starting at the base layer, with a sequence of motion tokens obtained by vector quantization, the residual tokens of increasing orders are derived and stored at the subsequent layers of the hierarchy. This is consequently followed by two distinct bidirectional transformers. For the base-layer motion tokens, a Masked Transformer is designated to predict randomly masked motion tokens conditioned on text input at training stage. During generation (i.e. inference) stage, starting from an empty sequence, our Masked Transformer iteratively fills up the missing tokens; Subsequently, a Residual Transformer learns to progressively predict the next-layer tokens based on the results from current layer. Extensive experiments demonstrate that MoMask outperforms the state-of-art methods on the text-to-motion generation task, with an FID of 0.045 (vs e.g. 0.141 of T2M-GPT) on the HumanML3D dataset, and 0.228 (vs 0.514) on KIT-ML, respectively. MoMask can also be seamlessly applied in related tasks without further model fine-tuning, such as text-guided temporal inpainting.
</details>

### 11. Self-Correcting Self-Consuming Loops For Generative Model Training
Nate Gillman, Michael Freeman, Daksh Aggarwal, Chia-Hong Hsu, Calvin Luo, Yonglong Tian, Chen Sun (Brown University, Google Research)
<details span>
<summary><b>Abstract</b></summary>
What happens after iteratively training a text-conditioned generative model for human motion synthesis for 50 generations? We simulate a self-consuming loop by creating synthetic data with the latest generative model, and mixing them with the original data to continue training the next generative model. We observe that by self-correcting the synthetic data with a physics simulator, the model can successfully avoid collapse and generate high-quality human motion. Our paper provides theoretical and empirical justification for the self-correcting self-consuming loop.
</details>



### Text to Human Motion Paper lists
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2024 | **Synthesizing Moving People with 3D Control**  | Arxiv 2024 |          [Link](https://arxiv.org/abs/2401.10889)          | [Link](https://github.com/Boyiliee/3DHM)   | [Link](https://boyiliee.github.io/3DHM.github.io/)  |
| 2024 | **Multi-Track Timeline Control for Text-Driven 3D Human Motion Generation**  | Arxiv 2024 |          [Link](https://arxiv.org/abs/2401.08559)          | Coming Soon! | [Link](https://mathis.petrovich.fr/stmc/)  |
| 2023 | **EMDM: Efficient Motion Diffusion Model for Fast, High-Quality Human Motion Generation**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2312.02256)          | [Link](https://github.com/Frank-ZY-Dou/EMDM) | [Link](https://frank-zy-dou.github.io/projects/EMDM/index.html)  |
| 2023 | **SinMDM: Single Motion Diffusion**  | ICLR 2024 Spotlight |          [Link](https://arxiv.org/abs/2302.05905)          | [Link](https://github.com/SinMDM/SinMDM)  | [Link](https://sinmdm.github.io/SinMDM-page/)  |
| 2023 | **MDM: Human Motion Diffusion Model**  | ICLR2023 (Top-25%) |          [Link](https://arxiv.org/abs/2209.14916)          | [Link](https://github.com/GuyTevet/motion-diffusion-model)  | [Link](https://guytevet.github.io/mdm-page/)  |
| 2023 | **MLD: Motion Latent Diffusion Models**  | CVPR 2023 |          [Link](https://arxiv.org/abs/2212.04048)          | [Link](https://github.com/ChenFengYe/motion-latent-diffusion)  | [Link](https://chenxin.tech/mld/)  |
| 2023 | **HumanTOMATO: Text-aligned Whole-body Motion Generation**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2310.12978)          | [Link](https://github.com/IDEA-Research/HumanTOMATO)  | [Link](https://lhchen.top/HumanTOMATO/)  |
| 2023 | **MotionGPT: Human Motion as a Foreign Language**  | NeurIPS 2023 |          [Link](https://arxiv.org/abs/2306.14795)          | [Link](https://github.com/OpenMotionLab/MotionGPT)  | [Link](https://motion-gpt.github.io/)  |
| 2023 | **Story-to-Motion: Synthesizing Infinite and Controllable Character Animation from Long Text**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2311.07446)          | Coming soon! | [Link](https://story2motion.github.io/)  |
| 2023 | **Plan, Posture and Go: Towards Open-World Text-to-Motion Generation**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2312.14828)          | [Link](https://github.com/moonsliu/Pro-Motion) | [Link](https://moonsliu.github.io/Pro-Motion/)  |
| 2023 | **MoMask: Generative Masked Modeling of 3D Human Motions**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2312.00063)          | [Link](https://github.com/EricGuo5513/momask-codes) | [Link](https://ericguo5513.github.io/momask/)  |
| 2024 | **Self-Correcting Self-Consuming Loops for Generative Model Training**  | Arxiv 2024 |          [Link](https://arxiv.org/abs/2402.07087)          | [Link](https://github.com/nate-gillman/self-correcting-self-consuming) | [Link](https://cs.brown.edu/people/ngillman//sc-sc.html)  |

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

@misc{gillman2024selfcorrecting,
  title={Self-Correcting Self-Consuming Loops for Generative Model Training}, 
  author={Nate Gillman and Michael Freeman and Daksh Aggarwal and Chia-Hong Hsu and Calvin Luo and Yonglong Tian and Chen Sun},
  year={2024},
  eprint={2402.07087},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
</details>


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
### Survey and Awesome Repos 
<details close>
<summary>🔥 Topic 1: 3D Gaussian Splatting</summary>
 
#### Survey
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
- Resource1: [Awesome 3D AIGC](https://github.com/mdyao/Awesome-3D-AIGC)

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

</details>

--------------

## License 
Awesome Text2X Resources is released under the [MIT license](./LICENSE).
