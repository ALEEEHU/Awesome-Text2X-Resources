[stars-img]: https://img.shields.io/github/stars/ALEEEHU/Awesome-Text2X-Resources?color=yellow
[stars-url]: https://github.com/ALEEEHU/Awesome-Text2X-Resources/stargazers
[fork-img]: https://img.shields.io/github/forks/ALEEEHU/Awesome-Text2X-Resources?color=blue&label=fork
[fork-url]: https://github.com/ALEEEHU/Awesome-Text2X-Resources/network/members

# Awesome Text2X Resources
[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FALEEEHU%2FAwesome-Text2X-Resources%2F&labelColor=%23d9e3f0&countColor=%23f47373&style=flat)

This is an open collection of state-of-the-art (SOTA), novel **Text to X (X can be everything)** methods (papers, codes and datasets), intended to keep pace with the anticipated surge of research in the coming months. 

⭐ If you find this repository useful to your research or work, it is really appreciated to star this repository. 

:heart: Any additions or suggestions, feel free to contribute. Additional resources like blog posts, videos, etc. are also welcome.

## Table of contents

- [Text to 3D Human](#text-to-3d-human)
  * [Paper lists](#paper-lists)
  * [Datasets](#datasets)
- [Text to Human Motion](#text-to-human-motion)
  * [Paper lists](#paper-lists)
  * [Datasets](#datasets)
- [Text to Scene](#text-to-scene)
  * [Paper lists](#paper-lists)
  * [Datasets](#datasets)
- [Others](#others)
  * [Reference](#reference)
  * [Other Related Awesome Repository](#other-related-awesome-repository)

--------------

## Text to 3D Human

### 1. HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting  
Xian Liu, Xiaohang Zhan, Jiaxiang Tang, Ying Shan, Gang Zeng, Dahua Lin, Xihui Liu, Ziwei Liu
<details span>
<summary><b>Abstract</b></summary>
Realistic 3D human generation from text prompts is a desirable yet challenging task. Existing methods optimize 3D representations like mesh or neural fields via score distillation sampling (SDS), which suffers from inadequate fine details or excessive training time. In this paper, we propose an efficient yet effective framework, HumanGaussian, that generates high-quality 3D humans with fine-grained geometry and realistic appearance. Our key insight is that 3D Gaussian Splatting is an efficient renderer with periodic Gaussian shrinkage or growing, where such adaptive density control can be naturally guided by intrinsic human structures. Specifically, 1) we first propose a Structure-Aware SDS that simultaneously optimizes human appearance and geometry. The multi-modal score function from both RGB and depth space is leveraged to distill the Gaussian densification and pruning process. 2) Moreover, we devise an Annealed Negative Prompt Guidance by decomposing SDS into a noisier generative score and a cleaner classifier score, which well addresses the over-saturation issue. The floating artifacts are further eliminated based on Gaussian size in a prune-only phase to enhance generation smoothness. Extensive experiments demonstrate the superior efficiency and competitive quality of our framework, rendering vivid 3D humans under diverse scenarios.
</details>

### 2. HumanNorm: Learning Normal Diffusion Model for High-quality and Realistic 3D Human Generation  
Xin Huang, Ruizhi Shao, Qi Zhang, Hongwen Zhang, Ying Feng, Yebin Liu, Qing Wang
<details span>
<summary><b>Abstract</b></summary>
Recent text-to-3D methods employing diffusion models have made significant advancements in 3D human generation. However, these approaches face challenges due to the limitations of text-to-image diffusion models, which lack an understanding of 3D structures. Consequently, these methods struggle to achieve high-quality human generation, resulting in smooth geometry and cartoon-like appearances. In this paper, we propose HumanNorm, a novel approach for high-quality and realistic 3D human generation. The main idea is to enhance the model's 2D perception of 3D geometry by learning a normal-adapted diffusion model and a normal-aligned diffusion model. The normal-adapted diffusion model can generate high-fidelity normal maps corresponding to user prompts with view-dependent and body-aware text. The normal-aligned diffusion model learns to generate color images aligned with the normal maps, thereby transforming physical geometry details into realistic appearance. Leveraging the proposed normal diffusion model, we devise a progressive geometry generation strategy and a multi-step Score Distillation Sampling (SDS) loss to enhance the performance of 3D human generation. Comprehensive experiments substantiate HumanNorm's ability to generate 3D humans with intricate geometry and realistic appearances. HumanNorm outperforms existing text-to-3D methods in both geometry and texture quality.
</details>

### 3. TeCH: Text-guided Reconstruction of Lifelike Clothed Humans  
Yangyi Huang, Hongwei Yi, Yuliang Xiu, Tingting Liao, Jiaxiang Tang, Deng Cai, Justus Thies
<details span>
<summary><b>Abstract</b></summary>
Despite recent research advancements in reconstructing clothed humans from a single image, accurately restoring the "unseen regions" with high-level details remains an unsolved challenge that lacks attention. Existing methods often generate overly smooth back-side surfaces with a blurry texture. But how to effectively capture all visual attributes of an individual from a single image, which are sufficient to reconstruct unseen areas (e.g., the back view)? Motivated by the power of foundation models, TeCH reconstructs the 3D human by leveraging 1) descriptive text prompts (e.g., garments, colors, hairstyles) which are automatically generated via a garment parsing model and Visual Question Answering (VQA), 2) a personalized fine-tuned Text-to-Image diffusion model (T2I) which learns the "indescribable" appearance. To represent high-resolution 3D clothed humans at an affordable cost, we propose a hybrid 3D representation based on DMTet, which consists of an explicit body shape grid and an implicit distance field. Guided by the descriptive prompts + personalized T2I diffusion model, the geometry and texture of the 3D humans are optimized through multi-view Score Distillation Sampling (SDS) and reconstruction losses based on the original observation. TeCH produces high-fidelity 3D clothed humans with consistent & delicate texture, and detailed full-body geometry. Quantitative and qualitative experiments demonstrate that TeCH outperforms the state-of-the-art methods in terms of reconstruction accuracy and rendering quality. 
</details>

### 4. DreamWaltz: Make a Scene with Complex 3D Animatable Avatars  
Yukun Huang, Jianan Wang, Ailing Zeng, He Cao, Xianbiao Qi, Yukai Shi, Zheng-Jun Zha, Lei Zhang
<details span>
<summary><b>Abstract</b></summary>
We present DreamWaltz, a novel framework for generating and animating complex 3D avatars given text guidance and parametric human body prior. While recent methods have shown encouraging results for text-to-3D generation of common objects, creating high-quality and animatable 3D avatars remains challenging. To create high-quality 3D avatars, DreamWaltz proposes 3D-consistent occlusion-aware Score Distillation Sampling (SDS) to optimize implicit neural representations with canonical poses. It provides view-aligned supervision via 3D-aware skeleton conditioning which enables complex avatar generation without artifacts and multiple faces. For animation, our method learns an animatable 3D avatar representation from abundant image priors of diffusion model conditioned on various poses, which could animate complex non-rigged avatars given arbitrary poses without retraining. Extensive evaluations demonstrate that DreamWaltz is an effective and robust approach for creating 3D avatars that can take on complex shapes and appearances as well as novel poses for animation. The proposed framework further enables the creation of complex scenes with diverse compositions, including avatar-avatar, avatar-object and avatar-scene interactions.
</details>

### Paper lists
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      | Project Page                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
| 2023 | **HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting**  | arXiv  |          [Link](https://arxiv.org/abs/2311.17061)          | [Link](https://github.com/alvinliu0/HumanGaussian)  | [Link](https://alvinliu0.github.io/projects/HumanGaussian)  | 
| 2023 | **HumanNorm: Learning Normal Diffusion Model for High-quality and Realistic 3D Human Generation**  | arXiv  |          [Link](https://arxiv.org/abs/2310.01406)          | Coming soon!  | [Link](https://humannorm.github.io/)  |
| 2023 | **TeCH: Text-guided Reconstruction of Lifelike Clothed Humans**  | 3DV 2024  |          [Link](https://arxiv.org/abs/2308.08545)          | [Link](https://github.com/huangyangyi/TeCH)  | [Link](https://huangyangyi.github.io/TeCH/)  |
| 2023 | **DreamWaltz: Make a Scene with Complex 3D Animatable Avatars**  | arXiv  |          [Link](https://arxiv.org/abs/2305.12529)          | [Link](https://github.com/IDEA-Research/DreamWaltz)  | [Link](https://idea-research.github.io/DreamWaltz/)  |

### Datasets
<!--
   | Dataset | Samples |                              URL                              |
   | :-----: | :-----: | :----------------------------------------------------------: |
   |  USPS   |  9298   | [usps.zip](https://drive.google.com/file/d/19oBkSeIluW3A5kcV7W0UM1Bt6V9Q62e-/view?usp=sharing) |-->

--------------

## Text to Human Motion
### Paper lists
### Datasets

--------------

## Text to Scene
### Paper lists
### Datasets

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

% text to human motion

% text to scene

```

### Other Related Awesome Repository
- 🔥 topic 1 : 3DGS [Awesome 3D Gaussian Splatting Resources](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)
