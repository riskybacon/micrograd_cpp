CPP = g++
CFLAGS = -Wall -g --std=c++20

TARGET = main

SRCS = main.cpp
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CPP) $(CFLAGS) -o $@ $^

%.o: %.cpp
	$(CPP) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)