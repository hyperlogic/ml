DATA_FILES = train-points.npy train-categories.npy test-points.npy test-categories.npy

all: $(DATA_FILES)

$(DATA_FILES): triangle_data.py
	python triangle_data.py

.PHONY: clean
clean:
	rm -rf $(DATA)