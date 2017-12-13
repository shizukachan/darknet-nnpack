Forked from https://github.com/digitalbrain79/darknet-nnpack/

# Darknet with NNPACK
NNPACK was used to optimize [Darknet](https://github.com/pjreddie/darknet) without using a GPU. It is useful for embedded devices using ARM CPUs.
Idein's [qmkl](https://github.com/Idein/qmkl) is also used to accelerate the SGEMM using the GPU. This is slower than NNPACK on NEON-capable devices, and primarily useful for ARM CPUs without NEON.

## Build from Raspberry Pi 3
Log in to Raspberry Pi using SSH.<br/>
Install [PeachPy](https://github.com/Maratyszcza/PeachPy) and [confu](https://github.com/Maratyszcza/confu)
```
sudo pip install --upgrade git+https://github.com/Maratyszcza/PeachPy
sudo pip install --upgrade git+https://github.com/Maratyszcza/confu
```
Install [Ninja](https://ninja-build.org/)
```
git clone https://github.com/ninja-build/ninja.git
cd ninja
git checkout release
./configure.py --bootstrap
export NINJA_PATH=$PWD
```
Install clang
```
sudo apt-get install clang
```
Install [NNPACK-darknet](https://github.com/thomaspark-pkj/NNPACK-darknet.git)
```
git clone https://github.com/thomaspark-pkj/NNPACK-darknet.git
cd NNPACK-darknet
confu setup
```
If you are compiling for the Pi Zero, run `python ./configure.py --backend scalar`, otherwise run `python ./configure.py --backend auto`
It's also recommended to examine and edit https://github.com/digitalbrain79/NNPACK-darknet/blob/master/src/init.c#L215 to match your CPU architecture:
Model | L1 cache size | L1 cache associativity | L2 cache size | L2 cache associativity | L2 cache inclusiveness | L2 cache shared threads
:-:|:-:|:-:|:-:|:-:|:-:|:-:
BCM2835 | 16*1024 | 4 | 128*1024*1024 | ? | ? | 1
BCM2837 | 32*1024 | 4 | 512*1024*1024 | 16 | yes (l1i) and no (l1d) | 4
Since none of the ARM CPUs have a L3, it's recommended to set L3 = L2 -- and also set L3 threads to 1.
```
$NINJA_PATH/ninja
sudo cp -a lib/* /usr/lib/
sudo cp include/nnpack.h /usr/include/
sudo cp deps/pthreadpool/include/pthreadpool.h /usr/include/
```

Install [qmkl](https://github.com/Idein/qmkl)
```
sudo apt-get install cmake
git clone https://github.com/Idein/qmkl.git
cd qmkl
cmake .
make
sudo make install
```

Install [qasm2](https://github.com/Terminus-IMRC/qpu-assembler2)
```
sudo apt-get install flex
git clone https://github.com/Terminus-IMRC/qpu-assembler2
cd qpu-assembler2
make
sudo make install
```

Install [qbin2hex](https://github.com/Terminus-IMRC/qpu-bin-to-hex)
```
git clone https://github.com/Terminus-IMRC/qpu-bin-to-hex
cd qpu-bin-to-hex
make
sudo make install
```

At this point, you can build darknet-nnpack using `make`. Be sure to enable QPU_GEMM if you want to use the QPU.

## Test
The weight files can be downloaded from the [YOLO homepage](https://pjreddie.com/darknet/yolo/).
```
YOLOv2
./darknet detector test cfg/coco.data cfg/yolo.cfg yolo.weights data/person.jpg
Tiny-YOLO
./darknet detector test cfg/voc.data cfg/tiny-yolo-voc.cfg tiny-yolo-voc.weights data/person.jpg
```
## NNPACK CPU-only Results (Raspberry Pi 3)
Model | Build Options | Prediction Time (seconds)
:-:|:-:|:-:
YOLOv2 | NNPACK=1,ARM_NEON=1 | 8.2
YOLOv2 | NNPACK=0,ARM_NEON=0 | 156
Tiny-YOLO | NNPACK=1,ARM_NEON=1 | 1.3
Tiny-YOLO | NNPACK=0,ARM_NEON=0 | 38

## NNPACK+QPU_GEMM Results
Raspberry Pi | Model | Build Options | Prediction Time (seconds)
:-:|:-:|:-:|:-:
Pi 3 | Tiny-YOLO | NNPACK=1,ARM_NEON=1,QPU_GEMM=1 | 5.3

The QPU is slower than NNPACK-NEON. qmkl is just unable to match the performance NNPACK's extremely well tuned NEON implicit GEMM.
I imagine the best use case for this repo would be to run neural networks on Raspberry Pi's without NEON.

