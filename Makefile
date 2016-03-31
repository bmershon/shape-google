# author Brooks Mershon
# Type `make all` to create all the generated files

GENERATED_FILES = \
	build/cloud/biplane0.off \
	build/EGI/biplane0.off \
	build/EGI/biplane1.off \
	build/EGI/biplane2.off \
	build/EGI/sword0.off \
	build/EGI/sword1.off \
	build/EGI/sword2.off \
	build/EGI/fighter_jet4.off \
	build/EGI/fighter_jet5.off \
	build/EGI/fighter_jet8.off \
	build/EGI/fish0.off \
	build/EGI/fish2.off \
	build/EGI/fish7.off \
	build/EGI/guitar0.off \
	build/EGI/guitar5.off \
	build/EGI/guitar9.off 


all: $(GENERATED_FILES)

.PHONY: clean all

clean:
	rm -rf -- $(GENERATED_FILES) build

# Create Unit Sphere mesh (.off) with Extended Gaussian Image color coding
# for a given .off file
build/EGI/%.off: 
	mkdir -p build/EGI
	python test/EGI-test.py models_off/$(notdir $@) $@;

# Create point cloud with points and normals for 
# for a given .off file
build/cloud/%.off: 
	mkdir -p build/cloud
	python test/sample-test.py models_off/$(notdir $@) $@.pts;