# ESR

C++ implementation of Explicit Shape Regression(ESR) algorithm.

## Dataset
We use LFPW dataset to train and test the model. Please download the data from [here](https://www.dropbox.com/s/1xl8jlyce1f4tei/lfpw.zip?dl=0) and decompress it under the `data` folder.

## Pre-trained Model
Please download the pre-trained model for the LFPW dataset and put it under the `data` folder [here](https://www.dropbox.com/s/cbl54ja2sejacgj/myModel_LFPW.txt?dl=0). 

## Dependencies
+ OpenCV2

## How to build and Run

```
#open directory
cd PATH_TO_DIRECTORY

#build project
make all

#training ESR model
./ESRTrain

#testing ESR model
./ESRTest
```

##References

[Cao X, Wei Y, Wen F, et al. Face alignment by explicit shape regression[J]. International Journal of Computer Vision, 2014, 107(2): 177-190.](http://download.springer.com/static/pdf/767/art%253A10.1007%252Fs11263-013-0667-3.pdf?originUrl=http%3A%2F%2Flink.springer.com%2Farticle%2F10.1007%2Fs11263-013-0667-3&token2=exp=1460503837~acl=%2Fstatic%2Fpdf%2F767%2Fart%25253A10.1007%25252Fs11263-013-0667-3.pdf%3ForiginUrl%3Dhttp%253A%252F%252Flink.springer.com%252Farticle%252F10.1007%252Fs11263-013-0667-3*~hmac=6505e6647730a48451f067d7ceb45fe222614be4990a779a370666c57c7d82f7)

[Dollár P, Welinder P, Perona P. Cascaded pose regression[C]//Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on. IEEE, 2010: 1078-1085.](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5540094)

