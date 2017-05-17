# Darknet with NNPACK
NNPACK was used to optimize [Darknet](https://github.com/pjreddie/darknet) without using a GPU. It is useful for embedded devices using ARM CPUs.

## Build from Raspberry Pi 3
Log in to Raspberry Pi using SSH.<br/>
Install [PeachPy](https://github.com/Maratyszcza/PeachPy)
```
git clone https://github.com/Maratyszcza/PeachPy.git
cd PeachPy
python setup.py build
sudo python setup.py install
```
Install [Ninja](https://ninja-build.org/)
```
git clone https://github.com/ninja-build/ninja.git
cd ninja
git checkout release
./configure.py --bootstrap
```
Install clang
```
sudo apt-get install clang
```
Install [NNPACK-rpi](https://github.com/thomaspark-pkj/NNPACK-rpi.git)
```
git clone https://github.com/thomaspark-pkj/NNPACK-rpi.git --recursive
cd NNPACK-rpi
git checkout raspberrypi
python ./configure.py --enable-rpi --enable-shared
$NINJA_PATH/ninja
sudo cp -a lib/libnnpack.* /usr/lib/
sudo cp include/nnpack.h /usr/include/
sudo cp third-party/pthreadpool/include/pthreadpool.h /usr/include/
```
Build darknet-nnpack
```
git clone https://github.com/thomaspark-pkj/darknet-nnpack.git
git checkout nnpack
make
```

## Test
The weight files can be downloaded from the [YOLO homepage](https://pjreddie.com/darknet/yolo/).
```
YOLO
./darknet detector test cfg/coco.data cfg/yolo.cfg yolo.weights data/person.jpg
Tiny-YOLO
./darknet detector test cfg/coco.data cfg/tiny-yolo.cfg tiny-yolo.weights data/person.jpg
```
## Result
Model | Build Options | Prediction Time (seconds)
:-:|:-:|:-:
YOLO | NNPACK=1,ARM_NEON=1 | 7.7
YOLO | NNPACK=0,ARM_NEON=0 | 156
Tiny-YOLO | NNPACK=1,ARM_NEON=1 | 1.8
Tiny-YOLO | NNPACK=0,ARM_NEON=0 | 38
