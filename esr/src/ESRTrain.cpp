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
		if (faces.size() == 0) {
			std::cout << "no face, skip!!!" << std::endl;
		    continue;
		}

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
		double center_x = 0, center_y = 0;
		for (int j = 0; j < NUM_LAND_MARK; j++)
		{
		    fp >> temps(j, 0) >> temps(j, 1);
		    center_x += temps(j, 0);
        	center_y += temps(j, 1);
		}
		getline(fp, temp);
		fp.close ();
		center_x /= NUM_LAND_MARK;
    	center_y /= NUM_LAND_MARK;

		double x_min, x_max, y_min, y_max;
    	x_min = x_max = temps(0, 0);
    	y_min = y_max = temps(0, 1);
    	for (int j = 0; j < NUM_LAND_MARK; j++)
		{
		    x_min = min(x_min, temps(j, 0));
	        x_max = max(x_max, temps(j, 0));
	        y_min = min(y_min, temps(j, 1));
	        y_max = max(y_max, temps(j, 1));
	 	}

	 	int k = -1;
	 	for (int j = 0; j < faces.size(); j++) {
	        Rect r = faces[j];
	        if (x_max - x_min > r.width*1.5) continue;
	        if (y_max - y_min > r.height*1.5) continue;
	        if (abs(center_x - (r.x + r.width / 2)) > r.width / 2) continue;
	        if (abs(center_y - (r.y + r.height / 2)) > r.height / 2) continue;
	        k = j;
	        break;
    	}
    	if(k == -1) {
    		std::cout << "points not within face region!!!" << std::endl;
    		continue;
    	}

    	ESR::Bbox tempbb;
		tempbb.sx = faces[k].x;
		tempbb.sy = faces[k].y;
		tempbb.w  = faces[k].width;
		tempbb.h  = faces[k].height;
		tempbb.cx = tempbb.sx + tempbb.w/2.0;
		tempbb.cy = tempbb.sy + tempbb.h/2.0;

		Mat_<uchar> tempi = imread(filename.c_str(), 0);
		trainImg.push_back(tempi);
		trainImgShape.push_back(temps);
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
