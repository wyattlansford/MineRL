# Makefile to build & run docker images
# Assumes you have setup docker for local user:
# https://docs.docker.com/engine/security/rootless/

VER=0.1
## make build TAG=_<tag> [Default: ]
## 	Builds docker image tagged MineRL:v<VER><tag>
TAG=
build :
	docker build -t MineRL:v$(VER)$(TAG) .

## make run-dev IMAGE= [Default: MineRL:v<VER>]
##	Starts a docker container with specified image and mounts directory for development
IMAGE=MineRL:v$(VER)
run-dev :
	docker run -it --rm --gpus all --network="host" \
	    --shm-size="5g" \
	    -v ~/Documents/MineRL/:/home/MineRL -v /media/banshee:/media/banshee \
	    -v ~/Documents/MineRL/ray_results:/root/ray_results \
	    -v ~/Documents/ray:/home/ray \
	    $(IMAGE)

## make run-dev-render IMAGE= [Default: MineRL:v<VER>]
##	Starts container with specified image, directory mounted, and X11 forwarding to render
# Ignore errors from docker exit to force xhost - to be set again
IMAGE=MineRL:v$(VER)
run-dev-render :
	xhost +
	-sudo docker run -it --rm --gpus all --network="host" \
	    --shm-size="5g" \
	    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$(DISPLAY) \
	    -v ~/Documents/MineRL/:/home/MineRL -v /media/banshee:/media/banshee \
	    -v ~/Documents/MineRL/ray_results:/root/ray_results \
	    $(IMAGE)
	xhost -


## make run IMAGE= [Default: MineRL:v<VER>]
##	Starts a docker container with specified image. Code dir will be from the image
IMAGE=MineRL:v$(VER)
run :
	docker run -it --rm --gpus all --network="host" \
	    --shm-size="5g" \
	    -v /media/banshee:/media/banshee \
	    -v ~/Documents/MineRL/ray_results:/root/ray_results \
	    $(IMAGE)

## make clean
##	Prunes all stopped containers and dangling images. Tagged images are NOT removed
clean :
	docker container prune && docker image prune

# help from https://swcarpentry.github.io/make-novice/08-self-doc/index.html
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<