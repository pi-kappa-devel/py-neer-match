# Neer Match


<a href="https://py-neer-match.pikappa.eu" style="float:right;margin-left:10px;"><img src="docs/source/_static/img/hex-logo.png" align="right" height="139" alt="neermatch website" /></a>

<!-- badges: start -->

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
<!-- badges: end -->

The package `neermatch` provides a set of tools for neural-symbolic
entity reasoning and matching. It is designed to support easy set-up,
training, and inference of entity matching models using deep learning,
symbolic learning, and a hybrid approach combining both deep and
symbolic learning. Moreover, the package provides automated fuzzy logic
reasoning (by refutation) functionality that can be used to examine the
significance of particular associations between fields in an entity
matching task.

The project is financially supported by the [Deutsche
Forschungsgemeinschaft](https://www.dfg.de/de) (DFG) under Grant
539465691 as part of the Infrastructure Priority Programme “[New Data
Spaces for the Social Sciences](https://www.new-data-spaces.de/en-us/)”
(SPP 2431).

The package has also an `R` implementation available at
[r-neer-match](https://github.com/pi-kappa-devel/r-neer-match).

## Features

The package is built on the concept of similarity maps. Similarity maps
are concise representations of potential associations between fields in
two datasets. Entities from two datasets can be matched using one or
more pairs of fields (one from each dataset). Each field pair can have
one or more ways to compute the similarity between the values of the
fields.

Similarity maps are used to automate the construction of entity matching
models and to facilitate the reasoning capabilities of the package. More
details on the concept of similarity maps and an early implementation of
the package’s functionality (without neural-symbolic components) are
given by (Karapanagiotis and Liebald 2023).

The training loops for both deep and symbolic learning models are
implemented in [tensorflow](https://www.tensorflow.org) (Abadi et al.
2015). The pure deep learning model inherits from the
[keras](https://keras.io) model class (Chollet et al. 2015). The
neural-symbolic model is implemented using the logic tensor network
([LTN](https://pypi.org/project/ltn/)) framework (Badreddine et al.
2022). Pure neural-symbolic and hybrid models do not inherit directly
from the [keras](https://keras.io) model class, but they emulate the
behavior by providing custom `compile`, `fit`, `evaluate`, and
`predict`methods, so that all model classes in `neermatch` have a
uniform calling interface.

## Auxiliary Features

In addition, the package offers explainability functionality customized
for the needs of matching problems. The default explainability behavior
is built on the information provided by the similarity map. From a
global explainability aspect, the package can be used to calculate
partial matching dependencies and accumulated local effects on
similarities. From a local explainability aspect, the package can be
used to calculate local interpretable model-agnostic matching
explanations and Shapley matching values.

# Basic Usage

Implementing matching models using `neermatch` is a three-step process:

1.  Instantiate a model with a similarity map.
2.  Compile the model.
3.  Train the model.

To train the model you need to provide three datasets. Two datasets
should contain records representing the entities to be matched. By
convention, the first dataset is called Left and the second dataset is
called Right dataset in the package’s documentation. The third dataset
should contain the ground truth labels for the matching entities. The
ground truth dataset should have two columns, one for the index of the
entity in the Left dataset and one for the index of the entity in the
Right dataset.

``` python
from neer_match.similarity_map import SimilarityMap
from neer_match.matching_model import NSMatchingModel
import tensorflow as tf

# 0) replace this with your own data preprocessing function
left, right, matches = prepare_data()

# 1) customize according to the fields in your data
smap = SimilarityMap(
    {
        "title": ["jaro", "levenshtein"],
        "developer~dev": ["jaro_winkler"],
        "platform": ["lcsseq"],
        "year": ["gaussian"],
    }
)
model = NSMatchingModel(smap)

# 2) compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

# 3) train
model.fit(
    left, right, matches,
    epochs=10, batch_size=16,
    log_mod_n=1,
)
```

    | Epoch      | BCE        | Recall     | Precision  | F1         | Sat        |
    | 0          | 7.0392     | 0.5897     | 0.2054     | 0.3046     | 0.7343     |
    | 1          | 6.6445     | 0.0000     | nan        | nan        | 0.7451     |
    | 2          | 6.4059     | 0.0000     | nan        | nan        | 0.7520     |
    | 3          | 6.1831     | 0.0000     | nan        | nan        | 0.7591     |
    | 4          | 5.9807     | 0.0000     | nan        | nan        | 0.7664     |
    | 5          | 5.8045     | 0.0000     | nan        | nan        | 0.7736     |
    | 6          | 5.6597     | 0.0000     | nan        | nan        | 0.7807     |
    | 7          | 5.5508     | 0.0000     | nan        | nan        | 0.7874     |
    | 8          | 5.4804     | 0.0000     | nan        | nan        | 0.7938     |
    | 9          | 5.4492     | 0.0000     | nan        | nan        | 0.7997     |
    Training finished at Epoch 9 with DL loss 5.4492 and Sat 0.7997

# Installation

## From Source

You can obtain the sources for the development version of `neermatch`
from its github
[repository](https://github.com/pi-kappa-devel/py-neer-match).

    git clone https://github.com/pi-kappa-devel/py-neer-match

To build and install the package locally, from the project’s root path,
execute

    python -m build
    python -m pip install dist/$(basename `ls -Art dist | tail -n 1` -py3-none-any.whl).tar.gz

# Documentation

Online documentation is available for the
[release](https://py-neer-match.pikappa.eu) version of the package.

## Reproducing Documentation from Source

Make sure to build and install the package with the latest modifications
before building the documentation. The documentation website is using
[sphinx](https://www.sphinx-doc.org/). The build the documentation, from
`<project-root>/docs`, execute

    make html

# Development Notes

## Logo

The logo was designed using [Microsoft
Designer](https://designer.microsoft.com/) and [GNU Image Manipulation
Program (GIMP)](https://www.gimp.org/). The hexagon version of the logo
was generated with the R package
[hexSticker](https://github.com/GuangchuangYu/hexSticker). It uses the
[Philosopher](https://fonts.google.com/specimen/Philosopher) font.

# Alternative Software

Several state-of-the-art entity matching (EM) systems have been developed in recent years, utilizing different methodologies to address the challenges of EM tasks. Below, we highlight some of the most recent, best-performing and/or most recognized EM systems:

- [**HierGAT**](https://github.com/CGCL-codes/HierGAT) ([Yao et al., 2022](https://dl.acm.org/doi/10.1145/3514221.3517872)): HierGAT introduces a Hierarchical Graph Attention Transformer Network to model and leverage interdependence between EM decisions and attributes. It uses a graph attention mechanism to identify discriminative words and attributes, combined with contextual embeddings to enrich word representations, enabling a more nuanced and interconnected approach to EM.

- [**Ditto**](https://github.com/megagonlabs/ditto) ([Li et al., 2020](https://dl.acm.org/doi/10.14778/3421424.3421431)): Ditto leverages pre-trained Transformer-based language models to cast EM as a sequence-pair classification task, enhancing matching quality through fine-tuning. It incorporates optimizations such as domain-specific highlighting, string summarization to retain essential information, and advanced data augmentation to improve training, making it both efficient and effective for large-scale EM tasks.

- **CorDEL** ([Wang et al., 2020](https://ieeexplore.ieee.org/document/9338287)): CorDEL employs a contrastive deep learning framework that moves beyond twin-network architectures by focusing on both syntactic and semantic matching signals while emphasizing critical subtle differences. The approach includes a simple yet effective variant, CorDEL-Sum, to enhance the model's ability to discern nuanced relationships in data.

- [**DAEM**](https://github.com/nju-websoft/DAEM) ([Huang et al., 2023](https://dl.acm.org/doi/abs/10.1007/s00778-022-00745-1)): This approach combines a deep neural network for EM with adversarial active learning, enabling the automatic completion of missing textual values and the modeling of both similarities and differences between records. It integrates active learning to curate high-quality labeled examples, adversarial learning for augmented stability, and a dynamic blocking method for scalable database handling, ensuring efficient and robust EM performance.

- [**AdaMEL**](https://github.com/DerekDiJin/AdaMEL-supplementary) ([Jin et al., 2021](https://dl.acm.org/doi/10.14778/3494124.3494131)): AdaMEL introduces a deep transfer learning framework for multi-source entity linkage, addressing challenges of incremental data and source variability by learning high-level generic knowledge. It employs an attribute-level self-attention mechanism to model attribute importance and leverages domain adaptation to utilize unlabeled data from new sources, enabling source-agnostic EM while accommodating additional labeled data for enhanced accuracy.

- [**DeepMatcher**](https://github.com/anhaidgroup/deepmatcher) ([Mudgal et al., 2018](https://dl.acm.org/doi/10.1145/3183713.3196926)): This framework is one of the first to introduce deep learning (DL) to entity matching, categorizing learning approaches into SIF, RNN, Attention, and Hybrid models based on their representational power. It highlights DL's strengths in handling textual and dirty EM tasks while identifying its limitations in structured EM, offering valuable insights for both researchers and practitioners.

- **SETEM** ([Ding et al., 2024](https://dl.acm.org/doi/10.1016/j.knosys.2024.111708)): SETEM introduces a self-ensemble training method for EM to overcome challenges in real-world scenarios, such as small datasets, hard negatives, and unseen entities, where traditional Pre-trained Language Model (PLM)-based methods often struggle due to their reliance on large labeled datasets and overlapping benchmarks. By leveraging the stability and generalization of ensemble models, SETEM effectively addresses these limitations while maintaining low memory consumption and high label efficiency. Additionally, it incorporates a faster training method designed for low-resource applications, ensuring adaptability and scalability for practical EM tasks.

- **AttendEM** ([Low et al., 2024](https://www.sciencedirect.com/science/article/pii/S0950705124003137?dgcid=rss_sd_all)): AttendEM introduces a novel framework for entity matching (EM) that enhances transformer architectures through intra-transformer ensembling, distinct text rearrangements, additional aggregator tokens, and extra self-attention layers. Departing from the focus on text cleaning and data augmentation in existing solutions, AttendEM innovates within the base model design, offering a distinct approach to pairwise duplicate identification across databases.

These systems represent significant advancements in the EM field, offering various approaches such as graph neural networks, attention mechanisms, transformers, and data augmentation. Depending on your project's requirements and data characteristics, they can serve as effective alternative solutions.

# Contributors

[Pantelis Karapanagiotis](https://www.pikappa.eu) (maintainer)

[Marius Liebald](https://www.marius-liebald.de) (contributor)

Feel free to share, modify, and distribute. If you implement new
features that might be of general interest, please consider contributing
them back to the project.

# License

The package is distributed under the [MIT license](LICENSE.txt).

# References

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-abadi2015" class="csl-entry">
    Abadi, Martín, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, et al. 2015. “TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems.” <https://www.tensorflow.org/>.
</div>
<br>

<div id="ref-badreddine2022" class="csl-entry">
    Badreddine, Samy, Artur d’Avila Garcez, Luciano Serafini, and Michael Spranger. 2022. “Logic Tensor Networks.” *Artificial Intelligence* 303: 103649. <https://doi.org/10.1016/j.artint.2021.103649>.
</div>
<br>

<div id="ref-chollet2015" class="csl-entry">
    Chollet, François et al. 2015. “Keras.” <https://keras.io>.
</div>
<br>

<div id="ref-ding2024" class="csl-entry">
    Ding, Huahua, Chaofan Dai, Yahui Wu, Wubin Ma, and Haohao Zhou. 2024. “SETEM: Self-ensemble Training with Pre-trained Language Models for Entity Matching.” <https://dl.acm.org/doi/10.1016/j.knosys.2024.111708>.
</div>
<br>

<div id="ref-huang2022" class="csl-entry">
    Huang, Jiacheng, Wei Hu, Zhifeng Bao, Qijin Chen, and Yuzhong Qu. 2022. “Deep Entity Matching with Adversarial Active Learning.” <https://dl.acm.org/doi/abs/10.1007/s00778-022-00745-1>.
</div>
<br>

<div id="ref-jin2021" class="csl-entry">
    Jin, Di, Bunyamin Sisman, Hao Wei, Xin Luna Dong, and Danai Koutra. 2021. “Deep Transfer Learning for Multi-Source Entity Linkage via Domain Adaptation.” <https://dl.acm.org/doi/10.14778/3494124.3494131>.
</div>
<br>

<div id="ref-karapanagiotis2023" class="csl-entry">
    Karapanagiotis, Pantelis, and Marius Liebald. 2023. “Entity Matching with Similarity Encoding: A Supervised Learning Recommendation Framework for Linking (Big) Data.” <http://dx.doi.org/10.2139/ssrn.4541376>.
</div>
<br>

<div id="ref-li2020" class="csl-entry">
    Li, Yuliang, Jinfeng Li, Yoshihiko Suhara, AnHai Doan, and Wang-Chiew Ta. 2020. “Deep Entity Matching with Pre-Trained Language Models.” <https://dl.acm.org/doi/10.14778/3421424.3421431>.
</div>
<br>

<div id="ref-low2024" class="csl-entry">
    Low, Jwen Fai, Benjamin C.M. Fung, and Pulei Xiong. 2024. “Better Entity Matching with Transformers Through Ensembles.” <https://www.sciencedirect.com/science/article/pii/S0950705124003137?dgcid=rss_sd_all>.
</div>
<br>

<div id="ref-mudgal2018" class="csl-entry">
    Mudgal, Sidharth, Han Li, Theodoros Rekatsinas, AnHai Doan, Youngchoon Park, Ganesh Krishnan, Rohit Deep, Esteban Arcaute, and Vijay Raghavendra. 2018. “Deep Learning for Entity Matching: A Design Space Exploration.” <https://dl.acm.org/doi/10.1145/3183713.3196926>.
</div>
<br>

<div id="ref-wang2020" class="csl-entry">
    Wang, Zhengyang, Bunyamin Sisman, Hao Wei, Xin Luna Dong, and Shuiwang Ji. 2020. “CorDEL: A Contrastive Deep Learning Approach for Entity Linkage.” <https://ieeexplore.ieee.org/document/9338287>.
</div>
<br>

<div id="ref-yao2022" class="csl-entry">
    Yao, Dezhong, Yuhong Gu, Gao Cong, Hai Jin, and Xinqiao Lv. 2022. “Entity Resolution with Hierarchical Graph Attention Networks.” <https://dl.acm.org/doi/10.1145/3514221.3517872>.
</div>
<br>

</div>
