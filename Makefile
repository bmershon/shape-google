# author Brooks Mershon

MODELS := $(shell find models_off/ -name *.off )

GENERATED_FILES = \
	$(MODELS)

all: $(GENERATED_FILES)

.PHONY: clean all

clean:
	rm -rf -- $(GENERATED_FILES) build

# Build unit sphere meshes with EGI color coding
EGI: 
	mkdir -p build/EGI
	@$(foreach m,$(MODELS),python test/EGI-test.py models_off/$(m) build/EGI/$(m);)