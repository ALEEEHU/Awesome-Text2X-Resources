# Awesome Text2X Resources
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FALEEEHU%2FAwesome-Text2X-Resources%2F&count_bg=%23EAA8EA&title_bg=%233D2549&icon=react.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=true)](https://hits.seeyoufarm.com)
[![GitHub](https://img.shields.io/github/stars/ALEEEHU/Awesome-Text2X-Resources?style=social)](https://github.com/ALEEEHU/Awesome-Text2X-Resources)

This is an open collection of state-of-the-art (SOTA), novel **Text to X (X can be everything)** methods (papers, codes and datasets), intended to keep pace with the anticipated surge of research in the coming months. 

⭐ If you find this repository useful to your research or work, it is really appreciated to star this repository. 

:heart: Any additions or suggestions, feel free to contribute and contact hyqale1024@gmail.com. Additional resources like blog posts, videos, etc. are also welcome.

## Table of contents

- [Text to 3D Human](#text-to-3d-human)
  * [Paper lists](#paper-lists)
  * [Pretrained Models](#pretrained-models)
- [Text to Human Motion](#text-to-human-motion)
  * [Paper lists](#paper-lists)
  * [Datasets](#datasets)
- [Text to Texture](#text-to-texture)
  * [Paper lists](#paper-lists)
- [Text to Video](#text-to-video)
  * [Paper lists](#paper-lists)
- [Text to 4D](#text-to-4d)
  * [Paper lists](#paper-lists)
- [Others](#others)
  * [Reference](#reference)
  * [Other Related Awesome Repository](#other-related-awesome-repository)

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

### Paper lists
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting**  | arXiv  |          [Link](https://arxiv.org/abs/2311.17061)          | [Link](https://github.com/alvinliu0/HumanGaussian)  | [Link](https://alvinliu0.github.io/projects/HumanGaussian)  | 
| 2023 | **HumanNorm: Learning Normal Diffusion Model for High-quality and Realistic 3D Human Generation**  | arXiv  |          [Link](https://arxiv.org/abs/2310.01406)          | [Link](https://github.com/xhuangcv/humannorm)  | [Link](https://humannorm.github.io/)  |
| 2023 | **TeCH: Text-guided Reconstruction of Lifelike Clothed Humans**  | 3DV 2024  |          [Link](https://arxiv.org/abs/2308.08545)          | [Link](https://github.com/huangyangyi/TeCH)  | [Link](https://huangyangyi.github.io/TeCH/)  |
| 2023 | **TADA! Text to Animatable Digital Avatars**  | 3DV 2024  |          [Link](https://arxiv.org/abs/2308.10899)          | [Link](https://github.com/TingtingLiao/TADA)  | [Link](https://tada.is.tue.mpg.de/)  |
| 2023 | **DreamWaltz: Make a Scene with Complex 3D Animatable Avatars**  | NeurIPS 2023  |          [Link](https://arxiv.org/abs/2305.12529)          | [Link](https://github.com/IDEA-Research/DreamWaltz)  | [Link](https://idea-research.github.io/DreamWaltz/)  |
| 2023 | **DreamHuman: Animatable 3D Avatars from Text**  | arXiv  |          [Link](https://arxiv.org/abs/2306.09329)          |  Coming soon!  | [Link](https://dream-human.github.io/)  |
| 2023 | **Guide3D: Create 3D Avatars from Text and Image Guidance**  | arXiv  |          [Link](https://arxiv.org/abs/2308.09705)          |  [Link](https://github.com/yukangcao/Guide3D) | -- |
| 2023 | **AvatarVerse: High-quality & Stable 3D Avatar Creation from Text and Pose**  | arXiv  |          [Link](https://arxiv.org/abs/2308.03610)          |  Coming soon!  | [Link](https://avatarverse3d.github.io/)  |
| 2022 | **AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars**  | SIGGRAPH 2022 (Journal Track)  |          [Link](https://arxiv.org/abs/2205.08535)          | [Link](https://github.com/hongfz16/AvatarCLIP)  | [Link](https://hongfz16.github.io/projects/AvatarCLIP.html)  |
| 2023 | **SEEAvatar: Photorealistic Text-to-3D Avatar Generation with Constrained Geometry and Appearance**  | arXiv  |          [Link](https://arxiv.org/abs/2312.08889)          | Coming Soon!  | [Link](https://seeavatar3d.github.io/)  |
| 2023 | **HeadArtist: Text-conditioned 3D Head Generation with Self Score Distillation**  | arXiv  |          [Link](https://arxiv.org/abs/2312.07539)          |  [Link](https://github.com/KumapowerLIU/HeadArtist)  | [Link](https://kumapowerliu.github.io/HeadArtist/)  |

### Pretrained Models 

   | Pretrained Models (human body) | Info |                              URL                              |
   | :-----: | :-----: | :----------------------------------------------------------: |
   |  SMPL  |  smpl model (smpl weights) | [Link](https://smpl.is.tue.mpg.de/) |
   |  SMPL-X  |  smpl model (smpl weights)  | [Link](https://smpl-x.is.tue.mpg.de/) |
   |  human_body_prior  |  vposer model (smpl weights)  | [Link](https://github.com/nghorbani/human_body_prior) |

#### Additional Info
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

--------------

## Text to Human Motion
### 1. HumanTOMATO: Text-aligned Whole-body Motion Generation 
Shunlin Lu*, Ling-Hao Chen*, Ailing Zeng, Jing Lin, Ruimao Zhang, Lei Zhang, Heung-Yeung Shum

(Tsinghua University, International Digital Economy Academy (IDEA), School of Data Science CUHK (SZ))
<details span>
<summary><b>Abstract</b></summary>
This work targets a novel text-driven whole-body motion generation task, which takes a given textual description as input and aims at generating high-quality, diverse, and coherent facial expressions, hand gestures, and body motions simultaneously. Previous works on text-driven motion generation tasks mainly have two limitations: they ignore the key role of fine-grained hand and face controlling in vivid whole-body motion generation, and lack a good alignment between text and motion. To address such limitations, we propose a Text-aligned whOle-body Motion generATiOn framework, named HumanTOMATO, which is the first attempt to our knowledge towards applicable holistic motion generation in this research area. To tackle this challenging task, our solution includes two key designs: (1) a Holistic Hierarchical VQ-VAE (aka H²VQ) and a Hierarchical-GPT for fine-grained body and hand motion reconstruction and generation with two structured codebooks; and (2) a pre-trained text-motion-alignment model to help generated motion align with the input textual description explicitly. Comprehensive experiments verify that our model has significant advantages in both the quality of generated motions and their alignment with text.
</details>


### 2. MotionGPT: Human Motion as a Foreign Language  
Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, Tao Chen

(Fudan University, Tencent PCG, ShanghaiTech University)
<details span>
<summary><b>Abstract</b></summary>
Though the advancement of pre-trained large language models unfolds, the exploration of building a unified model for language and other multimodal data, such as motion, remains challenging and untouched so far. Fortunately, human motion displays a semantic coupling akin to human language, often perceived as a form of body language. By fusing language data with large-scale motion models, motion-language pre-training that can enhance the performance of motion-related tasks becomes feasible. Driven by this insight, we propose MotionGPT, a unified, versatile, and user-friendly motion-language model to handle multiple motion-relevant tasks. Specifically, we employ the discrete vector quantization for human motionand transfer 3D motion into motion tokens, similar to the generation process ofword tokens. Building upon this “motion vocabulary”, we perform language modeling on both motion and text in a unified manner, treating human motion as a specific language. Moreover, inspired by prompt learning, we pre-train MotionGPT with a mixture of motion-language data and fine-tune it on prompt-based question-and-answer tasks. Extensive experiments demonstrate that MotionGPT achieves state-of-the-art performances on multiple motion tasks including text-driven motion generation, motion captioning, motion prediction, and motion in-between.
</details>

### 3. MLD: Motion Latent Diffusion Models 
Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, Jingyi Yu, Gang Yu

(Fudan University, Tencent PCG, ShanghaiTech University)
<details span>
<summary><b>Abstract</b></summary>
We study a challenging task, conditional human motion generation, which produces plausible human motion sequences according to various conditional inputs, such as action classes or textual descriptors. Since human motions are highly diverse and have a property of quite different distribution from conditional modalities, such as textual descriptors in natural languages, it is hard to learn a probabilistic mapping from the desired conditional modality to the human motion sequences. Besides, the raw motion data from the motion capture system might be redundant in sequences and contain noises; directly modeling the joint distribution over the raw motion sequences and conditional modalities would need a heavy computational overhead and might result in artifacts introduced by the captured noises. To learn a better representation of the various human motion sequences, we first design a powerful Variational AutoEncoder (VAE) and arrive at a representative and low-dimensional latent code for a human motion sequence. Then, instead of using a diffusion model to establish the connections between the raw motion sequences and the conditional inputs, we perform a diffusion process on the motion latent space. Our proposed Motion Latent-based Diffusion model (MLD) could produce vivid motion sequences conforming to the given conditional inputs and substantially reduce the computational overhead in both the training and inference stages. Extensive experiments on various human motion generation tasks demonstrate that our MLD achieves significant improvements over the state-of-the-art methods among extensive human motion generation tasks, with two orders of magnitude faster than previous diffusion models on raw motion sequences.
</details>

### 4. Story-to-Motion: Synthesizing Infinite and Controllable Character Animation from Long Text  
Zhongfei Qing, Zhongang Cai, Zhitao Yang, Lei Yang (SenseTime)
<details span>
<summary><b>Abstract</b></summary>
Generating natural human motion from a story has the potential to transform the landscape of animation, gaming, and film industries. A new and challenging task, Story-to-Motion, arises when characters are required to move to various locations and perform specific motions based on a long text description. This task demands a fusion of low-level control (trajectories) and high-level control (motion semantics). Previous works in character control and text-to-motion have addressed related aspects, yet a comprehensive solution remains elusive: character control methods do not handle text description, whereas text-to-motion methods lack position constraints and often produce unstable motions. In light of these limitations, we propose a novel system that generates controllable, infinitely long motions and trajectories aligned with the input text. 1) we leverage contemporary Large Language Models to act as a text-driven motion scheduler to extract a series of (text, position) pairs from long text. 2) we develop a text-driven motion retrieval scheme that incorporates classic motion matching with motion semantic and trajectory constraints. 3) we design a progressive mask transformer that addresses common artifacts in the transition motion such as unnatural pose and foot sliding. Beyond its pioneering role as the first comprehensive solution for Story-to-Motion, our system undergoes evaluation across three distinct sub-tasks: trajectory following, temporal action composition, and motion blending, where it outperforms previous state-of-the-art (SOTA) motion synthesis methods across the board.
</details>

### 5. Plan, Posture and Go: Towards Open-World Text-to-Motion Generation  
Jinpeng Liu, Wenxun Dai, Chunyu Wang, Yiji Cheng, Yansong Tang, Xin Tong 
(Shenzhen International Graudate School Tsinghua University, Microsoft Research Asia)
<details span>
<summary><b>Abstract</b></summary>
Conventional text-to-motion generation methods are usually trained on limited text-motion pairs, making them hard to generalize to open-world scenarios. Some works use the CLIP model to align the motion space and the text space, aiming to enable motion generation from natural language motion descriptions. However, they are still constrained to generate limited and unrealistic in-place motions. To address these issues, we present a divide-and-conquer framework named PRO-Motion, which consists of three modules as motion planner, posture-diffuser and go-diffuser. The motion planner instructs Large Language Models (LLMs) to generate a sequence of scripts describing the key postures in the target motion. Differing from natural languages, the scripts can describe all possible postures following very simple text templates. This significantly reduces the complexity of posture-diffuser, which transforms a script to a posture, paving the way for open-world generation. Finally, go-diffuser, implemented as another diffusion model, estimates whole-body translations and rotations for all postures, resulting in realistic motions. Experimental results have shown the superiority of our method with other counterparts, and demonstrated its capability of generating diverse and realistic motions from complex open-world prompts such as "Experiencing a profound sense of joy". 
</details>

### Paper lists
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **HumanTOMATO: Text-aligned Whole-body Motion Generation**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2310.12978)          | [Link](https://github.com/IDEA-Research/HumanTOMATO)  | [Link](https://lhchen.top/HumanTOMATO/)  |
| 2023 | **MotionGPT: Human Motion as a Foreign Language**  | NeurIPS 2023 |          [Link](https://arxiv.org/abs/2306.14795)          | [Link](https://github.com/OpenMotionLab/MotionGPT)  | [Link](https://motion-gpt.github.io/)  |
| 2023 | **MLD: Motion Latent Diffusion Models**  | CVPR 2023 |          [Link](https://arxiv.org/abs/2212.04048)          | [Link](https://github.com/ChenFengYe/motion-latent-diffusion)  | [Link](https://chenxin.tech/mld/)  |
| 2023 | **Story-to-Motion: Synthesizing Infinite and Controllable Character Animation from Long Text**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2311.07446)          | Coming soon! | [Link](https://story2motion.github.io/)  |
| 2023 | **Plan, Posture and Go: Towards Open-World Text-to-Motion Generation**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2312.14828)          | [Link](https://github.com/moonsliu/Pro-Motion) | [Link](https://moonsliu.github.io/Pro-Motion/)  |

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

## Text to Texture
### 1. Paint-it: Text-to-Texture Synthesis via Deep Convolutional Texture Map Optimization and Physically-Based Rendering 
Kim Youwang, Tae-Hyun Oh, Gerard Pons-Moll

(University of Tübingen, Tübingen AI Center Germany, Max Planck Institute for Informatics Germany, Dept. of Electrical Engineering POSTECH, Grad. School of AI POSTECH, Institute for Convergence Research and Education in Advanced Technology Yonsei University)
<details span>
<summary><b>Abstract</b></summary>
We present Paint-it, a text-driven high-fidelity texture map synthesis method for 3D meshes via neural re-parameterized texture optimization. Paint-it synthesizes texture maps from a text description by synthesis-through-optimization, exploiting the Score-Distillation Sampling (SDS). We observe that directly applying SDS yields undesirable texture quality due to its noisy gradients. We reveal the importance of texture parameterization when using SDS. Specifically, we propose Deep Convolutional Physically-Based Rendering (DC-PBR) parameterization, which re-parameterizes the physically-based rendering (PBR) texture maps with randomly initialized convolution-based neural kernels, instead of a standard pixel-based parameterization. We show that DC-PBR inherently schedules the optimization curriculum according to texture frequency and naturally filters out the noisy signals from SDS. In experiments, Paint-it obtains remarkable quality PBR texture maps within 15 min., given only a text description. We demonstrate the generalizability and practicality of Paint-it by synthesizing high-quality texture maps for large-scale mesh datasets and showing test-time applications such as relighting and material control using a popular graphics engine.
</details>

### Paper lists
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **Paint-it: Text-to-Texture Synthesis via Deep Convolutional Texture Map Optimization and Physically-Based Rendering**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2312.11360)          | [Link](https://github.com/postech-ami/paint-it)  | [Link](https://kim-youwang.github.io/paint-it)  |

--------------

## Text to Video
### 1. Follow Your Pose: Pose-Guided Text-to-Video Generation using Pose-Free Videos 
Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Ying Shan, Xiu Li, Qifeng Chen

(Tsinghua University(Tsinghua Shenzhen International Graduate School), HKUST, Tencent AI Lab)
<details span>
<summary><b>Abstract</b></summary>
Generating text-editable and pose-controllable character videos have an imperious demand in creating various digital human. Nevertheless, this task has been restricted by the absence of a comprehensive dataset featuring paired video-pose captions and the generative prior models for videos. In this work, we design a novel two-stage training scheme that can utilize easily obtained datasets (i.e.,image pose pair and pose-free video) and the pre-trained text-to-image (T2I) model to obtain the pose-controllable character videos. Specifically, in the first stage, only the keypoint-image pairs are used only for a controllable text-to-image generation. We learn a zero-initialized convolu- tional encoder to encode the pose information. In the second stage, we finetune the motion of the above network via a pose-free video dataset by adding the learnable temporal self-attention and reformed cross-frame self-attention blocks. Powered by our new designs, our method successfully generates continuously pose-controllable character videos while keeps the editing and concept composition ability of the pre-trained T2I model.
</details>

### Paper lists
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **Follow Your Pose: Pose-Guided Text-to-Video Generation using Pose-Free Videos**  | AAAI 2024 |          [Link](https://arxiv.org/abs/2304.01186)          | [Link](https://github.com/mayuelala/FollowYourPose)  | [Link](https://follow-your-pose.github.io/)  |

--------------

## Text to 4D
(Text to 3D Video)

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


### 5. Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models
Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, Karsten Kreis

(NVIDIA, ETH Zurich, Max Planck Institute for Intelligent Systems)
<details span>
<summary><b>Abstract</b></summary>
Text-guided diffusion models have revolutionized image and video generation and have also been successfully used for optimization-based 3D object synthesis. Here, we instead focus on the underexplored text-to-4D setting and synthesize dynamic, animated 3D objects using score distillation methods with an additional temporal dimension. Compared to previous work, we pursue a novel compositional generation-based approach, and combine text-to-image, text-to-video, and 3D-aware multiview diffusion models to provide feedback during 4D object optimization, thereby simultaneously enforcing temporal consistency, high-quality visual appearance and realistic geometry. Our method, called Align Your Gaussians (AYG), leverages dynamic 3D Gaussian Splatting with deformation fields as 4D representation. Crucial to AYG is a novel method to regularize the distribution of the moving 3D Gaussians and thereby stabilize the optimization and induce motion. We also propose a motion amplification mechanism as well as a new autoregressive synthesis scheme to generate and combine multiple 4D sequences for longer generation. These techniques allow us to synthesize vivid dynamic scenes, outperform previous work qualitatively and quantitatively and achieve state-of-the-art text-to-4D performance. Due to the Gaussian 4D representation, different 4D animations can be seamlessly combined, as we demonstrate. AYG opens up promising avenues for animation, simulation and digital content creation as well as synthetic data generation.
</details>

### 6. Control4D: Efficient 4D Portrait Editing with Text
Ruizhi Shao, Jingxiang Sun, Cheng Peng, Zerong Zheng, Boyao Zhou, Hongwen Zhang, Yebin Liu (Tsinghua University)
<details span>
<summary><b>Abstract</b></summary>
We introduce Control4D, an innovative framework for editing dynamic 4D portraits using text instructions. Our method addresses the prevalent challenges in 4D editing, notably the inefficiencies of existing 4D representations and the inconsistent editing effect caused by diffusion-based editors. We first propose GaussianPlanes, a novel 4D representation that makes Gaussian Splatting more structured by applying plane-based decomposition in 3D space and time. This enhances both efficiency and robustness in 4D editing. Furthermore, we propose to leverage a 4D generator to learn a more continuous generation space from inconsistent edited images produced by the diffusion-based editor, which effectively improves the consistency and quality of 4D editing. Comprehensive evaluation demonstrates the superiority of Control4D, including significantly reduced training time, high-quality rendering, and spatial-temporal consistency in 4D portrait editing.
</details>



### Paper lists
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **Text-To-4D Dynamic Scene Generation**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2301.11280)          | -  | [Link](https://make-a-video3d.github.io/)  |
| 2023 | **4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2311.17984)          | [link](https://github.com/sherwinbahmani/4dfy)  | [Link](https://sherwinbahmani.github.io/4dfy/)  |
| 2023 | **A Unified Approach for Text- and Image-guided 4D Scene Generation**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2311.16854)          | -  | [Link](https://research.nvidia.com/labs/nxp/dream-in-4d/)  |
| 2023 | **Animate124: Animating One Image to 4D Dynamic Scene**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2311.14603)          | [link](https://github.com/HeliosZhao/Animate124)  | [Link](https://animate124.github.io/)  |
| 2023 | **Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2312.13763)          | -  | [Link](https://research.nvidia.com/labs/toronto-ai/AlignYourGaussians/)  |
| 2023 | **Control4D: Efficient 4D Portrait Editing with Text**  | Arxiv 2023 |          [Link](https://arxiv.org/abs/2305.20082)          | Coming soon!  | [Link](https://control4darxiv.github.io./)  |

--------------

## Others
### Reference
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

% text to human motion

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

@inproceedings{chen2023executing,
  title={Executing your Commands via Motion Diffusion in Latent Space},
  author={Chen, Xin and Jiang, Biao and Liu, Wen and Huang, Zilong and Fu, Bin and Chen, Tao and Yu, Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18000--18010},
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

% text to texture

@misc{youwang2023paintit,
      title={Paint-it: Text-to-Texture Synthesis via Deep Convolutional Texture Map Optimization and Physically-Based Rendering},
      author={Kim Youwang and Tae-Hyun Oh and Gerard Pons-Moll},
      year={2023},
      eprint={2312.11360},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

% text to video

@article{ma2023follow,
  title={Follow Your Pose: Pose-Guided Text-to-Video Generation using Pose-Free Videos},
  author={Ma, Yue and He, Yingqing and Cun, Xiaodong and Wang, Xintao and Shan, Ying and Li, Xiu and Chen, Qifeng},
  journal={arXiv preprint arXiv:2304.01186},
  year={2023}
}

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

```

### Other Related Awesome Repository
- 🔥 Topic 1 : 3DGS [Awesome 3D Gaussian Splatting Resources](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)
- 🔥 Topic 2 : Awesome-LLM-3D. [Links](https://github.com/ActiveVisionLab/Awesome-LLM-3D)
- 🔥 text-to-3d object generation **benchmark** [T3Bench](https://github.com/THU-LYJ-Lab/T3Bench)
- 🔥 A growing curation of Text-to-3D, Diffusion-to-3D works. [Links](https://github.com/yyeboah/Awesome-Text-to-3D)
