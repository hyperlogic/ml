OBJ = dataset.o mnist_eval.o math_util.o

DEPENDS = dataset.h math_util.h

GCC = gcc
CFLAGS = -Wall
LFLAGS =
TARGET = mnist_eval.exe

$(TARGET): $(OBJ)
	$(GCC) $(OBJ) -o $(TARGET) $(LFLAGS)

%.o: %.c
	$(GCC) $(CFLAGS) -c $< -o $@

$(OBJ): $(DEPENDS)

.PHONY: clean
clean:
	rm -rf $(OBJ) $(TARGET)
