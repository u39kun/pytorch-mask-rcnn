# How to install the environment and run the demo

## Prerequisites

* A GPU compatible with CUDA 9.1
  * Modify the -arch flag in the `nvcc` commands below based on the GPU; see [README.md](README.md)
* [Anaconda](https://conda.io/docs/user-guide/install/index.html)

```
git clone git@github.com:u39kun/pytorch-mask-rcnn.git
cd pytorch-mask-rcnn/

conda create --name maskrcnn python=3.6
source activate maskrcnn
conda install scikit-image -y

git clone git@github.com:cocodataset/cocoapi.git
pushd cocoapi/PythonAPI
conda install cython
make

popd
ln -s cocoapi/PythonAPI/pycocotools pycocotools

conda install pytorch torchvision cuda91 -c pytorch

pushd nms/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
cd ../../
python build.py
popd

pushd roialign/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
cd ../../
python build.py
popd
```

## Download the pre-trained weight files

Download the Goole Drive CLI.
For 64-bit Linux; for other platforms, see https://github.com/prasmussen/gdrive.
```
wget "https://docs.google.com/uc?id=0B3X9GlR6EmbnQ0FtZmJJUXEyRTA&export=download" -O gdrive; chmod a+x ./gdrive
```

Download the weight files.
```
./gdrive download 1VV6WgX_RNl6a9Yi9-Pe7ZyVKHRJZSKkm
./gdrive download 12veVlnggRRaghRRyDIWTQkuxT8WSzIw6
```

## Running the demo
Run the demo code to make sure everything works; predictions are stored in the `predictions` directory.
```
python demo.py
```

