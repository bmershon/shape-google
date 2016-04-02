# author Brooks Mershon

GENERATED_FILES = \
	build/similarity/D2/D2.png \
	build/similarity/A3/A3.png \
	build/similarity/EGI/EGI.png \
	build/similarity/random/random.png \
	build/precision-recall/compare/precision-recall.png \
	build/precision-recall/shell/precision-recall-shell.png \
	build/precision-recall/D2/precision-recall-D2.png 

all: $(GENERATED_FILES)

.PHONY: clean all

clean:
	rm -rf -- $(GENERATED_FILES) build

# Create Unit Sphere mesh (.off) with Extended Gaussian Image color coding
# for a given .off file
build/EGI/%.off: 
	mkdir -p $(dir $@)
	python test/EGI-test.py models_off/$(notdir $@) $@;

# Create point cloud with points and normals for 
# for a given .off file
build/cloud/%.off: 
	mkdir -p $(dir $@)
	python test/sample-test.py models_off/$(notdir $@) $@.pts;

# Output .png of self-similarity matrix
build/similarity/D2/%.png:
	mkdir -p $(dir $@)
	python test/D2-self-similarity-test.py $@;

# Output .png of self-similarity matrix
build/similarity/A3/%.png:
	mkdir -p $(dir $@)
	python test/A3-self-similarity-test.py $@;

# Output .png of self-similarity matrix
build/similarity/EGI/%.png:
	mkdir -p $(dir $@)
	python test/EGI-self-similarity-test.py $@;

# Output .png of self-similarity matrix
build/similarity/random/%.png:
	mkdir -p $(dir $@)
	python test/random-self-similarity-test.py $@;

build/precision-recall/compare/%.png:
	mkdir -p $(dir $@)
	python test/precision-recall-test.py $@;

build/precision-recall/shell/%.png:
	mkdir -p $(dir $@)
	python test/precision-recall-shell-test.py $@;

build/precision-recall/D2/%.png:
	mkdir -p $(dir $@)
	python test/precision-recall-D2-test.py $@;