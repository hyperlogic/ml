DATA_FILES = train-points.np train-categories.np test-points.np test-categories.np

all: $(DATA_FILES)

$(DATA_FILES): triangle_data.py
	python triangle_data.py

.PHONY: clean
clean:
	rm -rf $(DATA)