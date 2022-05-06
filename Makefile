PROG = "main"

SRC_FILES = $(wildcard *.cu)
OBJ_FILES := $(patsubst %.cu,%.o,$(SRC_FILES))

NVCCFLAGS = -MD -arch=sm_37 -Wno-deprecated-gpu-targets -I. -rdc=true
LDFLAGS = -lm -g -G -arch=sm_37 -Wno-deprecated-gpu-targets

all: $(OBJ_FILES)
	nvcc $(LDFLAGS) $(OBJ_FILES) -o $(PROG)

%.o: %.cu
	nvcc $(NVCCFLAGS) -dc $< -o $@

clean:
	rm *.o
	rm *.d
	rm $(PROG)

.PHONY: all clean

-include $(OBJ_FILES:.o=.d)
