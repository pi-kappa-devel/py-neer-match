# Neer Match

<img src="docs/source/_static/img/hex-logo.png" align="right" height="192"/>
NEurosymbolic Entity Reasoning and Matching.

# Features

## Auxiliary Features
In addition, the package offers explainability functionality customized for the needs of matching problems. The default explainability behavior is built on the information provided by the similarity map. In particular, the package can be used to calculate
- partial matching dependencies on similarities.

# Documentation
Documentation is available for the latest version of the package [online]()

# Development Notes
To build and install the package locally, from the project's root path, execute
```bash
python -m build && \
	python -m pip install dist/$(basename `ls -Art dist | tail -n 1` -py3-none-any.whl).tar.gz
```

## Documentation
Make sure to build and install the package with the latest modifications before building the documentation. The build the documentation, from `<project-root>/docs`, execute 
```bash
make html
```

## Logo
The logo was designed using [Microsoft Designer](https://designer.microsoft.com/) and [GNU Image Manipulation Program (GIMP)](https://www.gimp.org/). The hexagon version of the logo was generated with the R package [hexSticker](https://github.com/GuangchuangYu/hexSticker). It uses the [Philosopher](https://fonts.google.com/specimen/Philosopher) font.

# Contributors

[Pantelis Karapanagiotis](https://www.pikappa.eu)

[Marius Liebald](https://www.marius-liebald.de/)

Feel free to share and distribute. If you would like to contribute, please send a pull request.

# License

The code is distributed under the Expat [License](LICENSE.txt).

# References
