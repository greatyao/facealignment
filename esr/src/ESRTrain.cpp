#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <vector>
#include <fstream>
#include <string>

#include "ESRUtils.hpp"
#include "ESRBbox.hpp"
#include "ESRRegressor.hpp"
#include "ESRCommon.hpp"

using namespace cv;

static std::vector<Mat> trainImg;
static std::vector<Mat> trainImgShape;
static std::vector<ESR::Bbox> trainImgBbox;
static std::vector<std::string> imglists;
static std::vector<std::string> ptslists;
static cv::CascadeClassifier cc;
static int NUM_LAND_MARK = 0;

void loadTrainingData();
void showTrainingData();

int train_main(int argc, char * argv[])
{
	if(argc != 5) 
	{
        	std::cout << "train dataset_path img_ext pts_ext cascade_file model_file" << std::endl;
        	return 0;
    	}

	imglists = ESR::ScanNSortDirectory(argv[0], argv[1]);
    	ptslists = ESR::ScanNSortDirectory(argv[0], argv[2]);
	cc = CascadeClassifier(argv[3]);
	if(cc.empty())
	{
		std::cout << "Cannot open model file " << argv[3] << " for OpenCV face detector!" << std::endl;
		return 0;
	}

	loadTrainingData();
	//showTrainingData();
	ESR::Regressor regressor;
	regressor.train(trainImg, trainImgBbox, trainImgShape);
	regressor.storeModel(argv[4]);

	return 0;
}

void loadTrainingData()
{
	std::cout << "loading training data..." << std::endl;

	for(int i = 0;  i < imglists.size(); i++) {
		string filename = imglists[i];
		cv::Mat image = cv::imread(filename.c_str());
		std::cout << filename << " " ;
		if(image.data == NULL) continue;
		cv::Mat gray_image;
		cv::cvtColor(image, gray_image, CV_BGR2GRAY);
		
		vector<cv::Rect> faces;
		cc.detectMultiScale(gray_image, faces);
		if (faces.size() != 1){
		    continue;
		}

		Mat_<uchar> tempi = imread(filename.c_str(), 0);
		trainImg.push_back(tempi);

		std::string temp;
		std::fstream fp;
		fp.open(ptslists[i].c_str(), std::ios::in);
		std::cout << ptslists[i] << std::endl;
		
		getline(fp, temp);
		getline(fp, temp, ' ');
		getline(fp, temp);
		int NbOfPoints = atoi(temp.c_str());
		getline(fp, temp);

		if(NUM_LAND_MARK == 0) {
			NUM_LAND_MARK = NbOfPoints;
		} else if(NUM_LAND_MARK != NbOfPoints) {
			std::cout << "ERROR: Points data does NOT has the same size!\n";
			std::cout <<  ptslists[i] << " has " << NbOfPoints << " points, ";
			std::cout << "but other has " << NUM_LAND_MARK << " points\n";
			exit(0);
		}
			
		Mat_<double> temps(NUM_LAND_MARK, 2);
		for (int j = 0; j < NUM_LAND_MARK; j++)
		{
		    fp >> temps(j,0) >> temps(j,1);
		}
		getline(fp, temp);
		fp.close ();
		trainImgShape.push_back(temps);

		ESR::Bbox tempbb;
		tempbb.sx = faces[0].x;
		tempbb.sy = faces[0].y;
		tempbb.w  = faces[0].width;
		tempbb.h  = faces[0].height;
		tempbb.cx = tempbb.sx + tempbb.w/2.0;
		tempbb.cy = tempbb.sy + tempbb.h/2.0;
		trainImgBbox.push_back(tempbb);
    }

}

void showTrainingData()
{
	for(int i=0; i<trainImg.size(); i++)
	{
		ESR::dispImgWithDetectionAndLandmarks(trainImg[i], trainImgShape[i], trainImgBbox[i], true, true);
	}
}
