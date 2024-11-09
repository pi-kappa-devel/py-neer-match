# Neer Match <a href="https://py-neer-match.pikappa.eu"><img src="docs/source/_static/img/hex-logo.png" align="right" height="139" alt="neermatch website" /></a>

<!-- badges: start -->
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
<!-- badges: end -->

The package `neermatch` provides a set of tools for neural-symbolic entity reasoning and matching. It is designed to support easy set-up, training, and inference of entity matching models using deep learning, symbolic learning, and a hybrid approach combining both deep and symbolic learning. Moreover, the package provides automated fuzzy logic reasoning (by refutation) functionality that can be used to examine the significance of particular associations between fields in an entity matching task.

The project is financially supported by the [Deutsche Forschungsgemeinschaft](https://www.dfg.de/de) (DFG) under Grant 539465691 as part of the Infrastructure Priority Programme "[New Data Spaces for the Social Sciences](https://www.new-data-spaces.de/en-us/)" (SPP 2431).

The package has also an `R` implementation available at [r-neer-match](https://github.com/pi-kappa-devel/r-neer-match).

## Features

The package is built on the concept of similarity maps. Similarity maps are concise representations of potential associations between fields in two datasets. Entities from two datasets can be matched using one or more pairs of fields (one from each dataset). Each field pair can have one or more ways to compute the similarity between the values of the fields.

Similarity maps are used to automate the construction of entity matching models and to facilitate the reasoning capabilities of the package. More details on the concept of similarity maps and an early implementation of the package’s functionality (without neural-symbolic components) are given by (Karapanagiotis and Liebald 2023).

The training loops for both deep and symbolic learning models are implemented in [tensorflow](https://www.tensorflow.org) (see Abadi et al. 2015). The pure deep learning model inherits from the [keras](https://keras.io) model class (Chollet et al. 2015). The neural-symbolic model is implemented using the logic tensor network ([LTN](https://pypi.org/project/ltn/)) framework (Badreddine et al. 2022). Pure neural-symbolic and hybrid models do not inherit directly from the (Chollet et al. 2015) model class, but they emulate the behavior by providing custom `compile`, `fit`, `evaluate`, and `predict`methods, so that all model classes in `neermatch` have a uniform calling interface.

## Auxiliary Features
In addition, the package offers explainability functionality customized for the needs of matching problems. The default explainability behavior is built on the information provided by the similarity map. From a global explainability aspect, the package can be used to calculate partial matching dependencies and accumulated local effects on similarities. From a local explainability aspect, the package can be used to calculate local interpretable model-agnostic matching explanations and Shapley matching values.

# Basic Usage

Implementing matching models using `neermatch` is a three-step process:

1.  Instantiate a model with a similarity map.
2.  Compile the model.
3.  Train the model.

To train the model you need to provide three datasets. Two datasets should contain records representing the entities to be matched. By convention, the first dataset is called Left and the second dataset is called Right dataset in the package’s documentation. The third dataset should contain the ground truth labels for the matching entities. The ground truth dataset should have two columns, one for the index of the entity in the Left dataset and one for the index of the entity in the Right dataset.

``` python
from neer_match.similarity_map import SimilarityMap
from neer_match.matching_model import NSMatchingModel
import tensorflow as tf

# 0) replace this with your own data preprocessing function
from neer_match.examples import games

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
    games.left,
    games.right,
    games.matches,
    epochs=100,
    batch_size=16,
    log_mod_n=10,
)
#>>> | Epoch      | BCE        | Recall     | Precision  | F1         | Sat        |
#>>> | 0          | 5.2150     | 1.0000     | 0.3333     | 0.5000     | 0.7245     |
#>>> | 10         | 6.9364     | 0.0000     | nan        | nan        | 0.7806     |
#>>> | 20         | 9.4707     | 0.0000     | nan        | nan        | 0.7853     |
#>>> | 30         | 8.9746     | 0.0000     | nan        | nan        | 0.7857     |
#>>> | 40         | 1.9495     | 0.0000     | nan        | nan        | 0.8339     |
#>>> | 50         | 0.7654     | 1.0000     | 0.8919     | 0.9429     | 0.8853     |
#>>> | 60         | 0.3452     | 1.0000     | 0.9429     | 0.9706     | 0.9083     |
#>>> | 70         | 1.2782     | 1.0000     | 0.8462     | 0.9167     | 0.8718     |
#>>> | 80         | 0.6670     | 1.0000     | 0.9167     | 0.9565     | 0.9039     |
#>>> | 90         | 0.8415     | 1.0000     | 0.9167     | 0.9565     | 0.9002     |
#>>> Training finished at Epoch 99 with DL loss 0.9324 and Sat 0.9020
```

# Installation

## From Source

You can obtain the sources for the development version of `neermatch` from its github [repository](https://github.com/pi-kappa-devel/py-neer-match).

``` bash
git clone https://github.com/pi-kappa-devel/py-neer-match
```

To build and install the package locally, from the project's root path, execute
```bash
python -m build
python -m pip install dist/$(basename `ls -Art dist | tail -n 1` -py3-none-any.whl).tar.gz
```

# Documentation

Online documentation is available for the [release](https://py-neer-match.pikappa.eu) version of the package.

## Reproducing Documentation from Source

Make sure to build and install the package with the latest modifications before building the documentation.  The documentation website is using [sphinx](https://www.sphinx-doc.org/). The build the documentation, from `<project-root>/docs`, execute 
```bash
make html
```

# Development Notes

## Logo
The logo was designed using [Microsoft Designer](https://designer.microsoft.com/) and [GNU Image Manipulation Program (GIMP)](https://www.gimp.org/). The hexagon version of the logo was generated with the R package [hexSticker](https://github.com/GuangchuangYu/hexSticker). It uses the [Philosopher](https://fonts.google.com/specimen/Philosopher) font.

# Alternative Software

TODO

# Contributors

[Pantelis Karapanagiotis](https://www.pikappa.eu) (maintainer)

[Marius Liebald](https://www.marius-liebald.de) (contributor)

Feel free to share, modify, and distribute. If you implement new features that might be of general interest, please consider contributing them back to the project.

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

<div id="ref-badreddine2022logic" class="csl-entry">

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
