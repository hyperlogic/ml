DATA_FILES = train-points.npy train-categories.npy test-points.npy test-categories.npy

C_OBJ = triangle_inference.o
C_DEPENDS =
C_TARGET = triangle_inference.exe
GCC = gcc
CFLAGS = -Wall
LFLAGS =

all: $(DATA_FILES) $(C_TARGET)

$(DATA_FILES): triangle_data.py
	python triangle_data.py

$(C_TARGET): $(C_OBJ)
	$(GCC) $(C_OBJ) -o $(C_TARGET) $(LFLAGS)

%.o: %.c
	$(GCC) $(CFLAGS) -c $< -o $@

$(C_OBJ): $(C_DEPENDS)

.PHONY: clean
clean:
	rm -rf $(DATA)