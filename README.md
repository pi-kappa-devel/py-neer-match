# Development notes
To build and install the package locally, from the project's root path, execute
```bash
python -m build && \
	python -m pip install dist/$(basename `ls -Art dist | tail -n 1` -py3-none-any.whl).tar.gz
```

## Documentation
Make sure to build and install the package with the latest modifications before building the documentation. The build the documentation, from `<project-root>/docs`, execute 
```bash
make
```
