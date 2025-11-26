# Simple multi-file build driven by PETSc make rules
TARGET := heat_linear
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build

SRCS := $(SRC_DIR)/main.c \
        $(SRC_DIR)/stiffness.c \
        $(SRC_DIR)/residual.c \
        $(SRC_DIR)/jacobian.c

OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

CFLAGS   += -I$(INC_DIR) -O3 -g -std=c99
CPPFLAGS += -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
LDLIBS   += -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc -lm

.PHONY: all clean dirs
all: dirs $(TARGET)

$(TARGET): $(OBJS)
	$(LINK.C) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | dirs
	$(COMPILE.c) $(OUTPUT_OPTION) $<

dirs:
	mkdir -p $(BUILD_DIR)

clean::
	rm -rf $(BUILD_DIR) $(TARGET)



# PETSc make system
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
