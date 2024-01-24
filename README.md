# Two View Reconstruction


## This is a mini project to conduct  two-view reconstruction
 - The project has been tested on Ubuntu 20.04.

## Prerequisites
 - This project relies on opencv 3.4 and opencv_contrib 3.4. Please make sure to have them installed on your machine. You can follow the commands below, but plaese make sure to modify the commands as needed:

```
git clone -b 3.4 https://github.com/opencv/opencv.git

git clone -b 3.4 https://github.com/opencv/opencv_contrib.git


sudo apt-get update
sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

cd opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..

make -j$(nproc)

sudo make install

sudo ldconfig

```
## build:

 - clone to the directory:
```
git clone https://github.com/meqdadhn/mini_project.git
```

 - build the executable (chmod command is only needed for the first time of running):

```
chmod +x build.sh
./build.sh
```

 - run the program (chmod command is only needed for the first time of running)
```
chmod +x run.sh
./run.sh
```
