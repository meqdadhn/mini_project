** Two View Reconstruction ** 

> Prerequisites:
>> this project uses opencv 3.4 and opencv_contib 3.4. Please make sure to have them installed in your local machine. You can follow the commands below, but make sure you make necessary changes:
'''
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

'''

> This is a mini project toconduct  two-view reconstruction
> The project has been tested in Ubuntu 20.04, and opencv 3.4

> clone to the directory
'''
git clone https://github.com/meqdadhn/mini_project.git
'''


> build the executable using following command:

'''
$ chmod +x build.sh
$ ./build.sh
'''

> run the executable
'''
chmod +x run.sh
./run.sh
'''