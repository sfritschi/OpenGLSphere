CC=gcc
CFLAGS=-Wall -Wextra -Wpedantic -std=gnu11

TARGET=sphere
.PHONY: all, clean
all=$(TARGET)

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) -o $@ $^ -lm -lGL -lglut -lGLEW

clean:
	$(RM) $(TARGET)
