# author Brooks Mershon

models := $(shell find models/ -name *.off)

GENERATED_FILES = \
	build/build.log

all: $(GENERATED_FILES)

.PHONY: clean all

clean:
	rm -rf -- $(GENERATED_FILES) build

# Build unit sphere meshes with EGI color coding
build/%.off: 
	mkdir -p $(dir $@)
	for f in models_off/*.off; do python test/EGI-test.py models_off/{f} build/EGI/{f}; done