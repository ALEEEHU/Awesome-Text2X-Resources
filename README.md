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

### Paper lists
| Year | Title                                                        | Venue  |                           Paper                            |                      Code                      |
| ---- | ------------------------------------------------------------ | :----: | :--------------------------------------------------------: | :--------------------------------------------: |
| 2023 | **HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting**  | arXiv  |          [Link](https://arxiv.org/abs/2311.17061)          | [Link](https://github.com/alvinliu0/HumanGaussian)  |

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
