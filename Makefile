init:
	pip install -e .[tests]

test:
	py.test tests/ --cov keras_rcnn/ --cov-report term-missing

pep8:
	py.test --pep8 -m pep8

.PHONY: init test pep8
