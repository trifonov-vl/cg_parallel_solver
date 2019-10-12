CXX = gcc
CXXFLAGS = -g -Wall -O3 -Werror -Wl,-z,defs -Wextra -fopenmp

# Directories with source code
SRC_DIR = src
INCLUDE_DIR = include

BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin

# Link libraries gcc flag: library will be searched with prefix "lib".
LDFLAGS = -lm

# Add headers dirs to gcc search path
CXXFLAGS += -I $(INCLUDE_DIR)

# Helper macros
# subst is sensitive to leading spaces in arguments.
make_path = $(addsuffix $(1), $(basename $(subst $(2), $(3), $(4))))
# Takes path list with source files and returns pathes to related objects.
src_to_obj = $(call make_path,.o, $(SRC_DIR), $(OBJ_DIR), $(1))

# All source files in our project that must be built into movable object code.
CXXFILES := $(wildcard $(SRC_DIR)/*.c)
OBJFILES := $(call src_to_obj, $(CXXFILES))

# Default target (make without specified target).
.DEFAULT_GOAL := all

# Alias to make all targets.
.PHONY: all
all: $(BIN_DIR)/main

# Suppress makefile rebuilding.
Makefile: ;

# Rules for compiling targets
$(BIN_DIR)/main: $(OBJFILES)
	$(CXX) $(CXXFLAGS) $(filter %.o, $^) -o $@ $(LDFLAGS)

# Pattern for compiling object files (*.o)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	mkdir -p $(OBJ_DIR) $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c -o $(call src_to_obj, $<) $<

# Fictive target
.PHONY: clean
# Delete all temprorary and binary files
clean:
	rm -rf $(BUILD_DIR)
