.PHONY: all install build safe_build clean source black
.DEFAULT_GOAL := all

all:
	$(MAKE) install
	$(MAKE) build

install:
	scripts/install.sh 

build:
	scripts/build.sh

safe_build:
	scripts/safe_build.sh

clean:
	rm -rf ./sim_ws/build ./sim_ws/install ./sim_ws/devel
	rm -rf ./rob_ws/build ./rob_ws/install ./rob_ws/devel

black:
	scripts/black.sh

source:
	scripts/source.sh