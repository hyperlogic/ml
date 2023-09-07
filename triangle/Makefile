DATA_FILES = train-points.npy train-categories.npy test-points.npy test-categories.npy

C_OBJ = triangle_inference.o math_util.o
C_DEPENDS = math_util.h
C_TARGET = triangle_inference.exe
GCC = gcc
CFLAGS = -Wall -g
LFLAGS =

all: $(C_TARGET)

data: $(DATA_FILES)

$(DATA_FILES): triangle_data.py
	python triangle_data.py

$(C_TARGET): $(C_OBJ)
	$(GCC) $(C_OBJ) -o $(C_TARGET) $(LFLAGS)

%.o: %.c
	$(GCC) $(CFLAGS) -c $< -o $@

$(C_OBJ): $(C_DEPENDS)

.PHONY: clean clean_data
clean:
	rm -rf $(C_OBJ) $(C_TARGET)
clean_data:
	rm -rf $(DATA)
