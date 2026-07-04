CC = gcc

CFLAGS = -I/usr/local/include
LDFLAGS = -L/usr/local/lib -lblis -lm

TARGET = kernals.so
SRC = attention.c softmax.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -shared -fPIC $(SRC) $(CFLAGS) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)