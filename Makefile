CPP = g++
EXTRA_CXXFLAGS =
CXXFLAGS = -Wall -g --std=c++20 -I. $(EXTRA_CXXFLAGS)

TARGETS = main test

SRCS = main.cpp test.cpp
OBJS = $(SRCS:.c=.o)

all: $(TARGETS)

$(TARGET): $(OBJS)
	$(CPP) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CPP) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS) *.dot *.png