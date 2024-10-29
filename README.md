# Neer Match

<img src="docs/source/_static/img/hex-logo.png" align="right" height="48"/>
NEurosymbolic Entity Reasoning and Matching.

## Development notes
To build and install the package locally, from the project's root path, execute
```bash
python -m build && \
	python -m pip install dist/$(basnename `ls -Art dist | tail -n 1` -py3-none-any.whl).tar.gz
```

### Documentation
Make sure to build and install the package with the latest modifications before building the documentation. The build the documentation, from `<project-root>/docs`, execute 
```bash
make
```

# Contributors

[Pantelis Karapanagiotis](https://www.pikappa.eu)

[Marius Liebald](https://www.marius-liebald.de/)

Feel free to share and distribute. If you would like to contribute, please send a pull request.

# License

The code is distributed under the Expat [License](LICENSE.txt).

# References
