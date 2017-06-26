ESR
=========================

C++ implementation of Face Alignment by Explicit Shape Regression.

## Dataset
We use several public dataset (LFPW, HELEN, AFW, IBUG) to train and test the model. You can download the data from [iBug Homepage](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) and decompress it under the `data` folder.

## Pre-trained Model
I have trained some models based on some dataset such as Helen, LFPW, you can download them from [BaiDu YunPan](https://pan.baidu.com/s/1gftIcsN). 

## Dependencies
+ OpenCV 2.x
+ CMake 2.8

## How to build

```
cd esr
cmake -D CMAKE_BUILD_TYPE=Release ./
make
```

When the above make command completed, you will get an executable file named as esr.

## How to train model

```
./esr train data/lfpw/trainset jpg pts haarcascade_frontalface_alt.xml lfpw.model
```
- train: esr train submodule
- data/lfpw/trainset: the path that holds the image and shape files
- jpg: image file format
- pts: shape file format
- haarcascade_frontalface_alt.xml: OpenCV haar-like face detect model
- lfpw.model: output model file 

## How to locate facial points from image
```
./esr live lfpw.model haarcascade_frontalface_alt.xml test.jpg
```
- live: esr fitting submodule
- lfpw.model: pre-trained model file 
- haarcascade_frontalface_alt.xml: OpenCV haar-like face detect model
- test.jpg: input image file 

## How to locate facial points from camera
```
./esr camera lfpw.model haarcascade_frontalface_alt.xml 0
```
- camera: esr fitting submodule
- lfpw.model: pre-trained model file 
- haarcascade_frontalface_alt.xml: OpenCV haar-like face detect model
- 0: camera index 


##References

[Cao X, Wei Y, Wen F, et al. Face alignment by explicit shape regression[J]. International Journal of Computer Vision, 2014, 107(2): 177-190.](http://download.springer.com/static/pdf/767/art%253A10.1007%252Fs11263-013-0667-3.pdf?originUrl=http%3A%2F%2Flink.springer.com%2Farticle%2F10.1007%2Fs11263-013-0667-3&token2=exp=1460503837~acl=%2Fstatic%2Fpdf%2F767%2Fart%25253A10.1007%25252Fs11263-013-0667-3.pdf%3ForiginUrl%3Dhttp%253A%252F%252Flink.springer.com%252Farticle%252F10.1007%252Fs11263-013-0667-3*~hmac=6505e6647730a48451f067d7ceb45fe222614be4990a779a370666c57c7d82f7)

[Dollár P, Welinder P, Perona P. Cascaded pose regression[C]//Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on. IEEE, 2010: 1078-1085.](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5540094)

