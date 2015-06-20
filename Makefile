
all:


test:
	nosetests --with-coverage --cover-package=pipeline_grid_search --cover-html

.PHONY: test
