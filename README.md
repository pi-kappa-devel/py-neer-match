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
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# 3) train
model.fit(
    left, right, matches,
    epochs=161, batch_size=16,
    log_mod_n=20,
)
```

    | Epoch      | BCE        | Recall     | Precision  | F1         | Sat        |
    | 0          | 7.0906     | 0.5897     | 0.1643     | 0.2570     | 0.7334     |
    | 20         | 8.5553     | 0.0000     | nan        | nan        | 0.8391     |
    | 40         | 12.4309    | 0.0000     | nan        | nan        | 0.8469     |
    | 60         | 16.1034    | 0.0000     | nan        | nan        | 0.8489     |
    | 80         | 18.4529    | 0.0000     | nan        | nan        | 0.8496     |
    | 100        | 19.7995    | 0.0000     | nan        | nan        | 0.8497     |
    | 120        | 20.8475    | 0.0000     | nan        | nan        | 0.8498     |
    | 140        | 21.7300    | 0.0000     | nan        | nan        | 0.8498     |
    | 160        | 22.4964    | 0.0000     | nan        | nan        | 0.8498     |
    Training finished at Epoch 160 with DL loss 22.4964 and Sat 0.8498

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

TODO

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

<div id="ref-tensorflow2015" class="csl-entry">

Abadi, Martín, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen,
Craig Citro, Greg S. Corrado, et al. 2015. “TensorFlow: Large-Scale
Machine Learning on Heterogeneous Systems.”
<https://www.tensorflow.org/>.

</div>

<div id="ref-badreddine2022" class="csl-entry">

Badreddine, Samy, Artur d’Avila Garcez, Luciano Serafini, and Michael
Spranger. 2022. “Logic Tensor Networks.” *Artificial Intelligence* 303:
103649. <https://doi.org/10.1016/j.artint.2021.103649>.

</div>

<div id="ref-keras2015" class="csl-entry">

Chollet, François et al. 2015. “Keras.” <https://keras.io>.

</div>

<div id="ref-karapanagiotis2023" class="csl-entry">

Karapanagiotis, Pantelis, and Marius Liebald. 2023. “Entity Matching
with Similarity Encoding: A Supervised Learning Recommendation Framework
for Linking (Big) Data.” <http://dx.doi.org/10.2139/ssrn.4541376>.

</div>

</div>
