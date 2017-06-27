#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;

#include "ESRRegressor.hpp"
#include "ESRUtils.hpp"
#include "ESRFaceDetector.hpp"
#include "ESRBbox.hpp"
#include "ESRCommon.hpp"

int test_main(int argc, char *argv[])
{
	if (argc != 4) 
	{
		std::cout << "test dataset_path img_ext model_file cascade_file" << std::endl;
		return -1;
	}

	std::vector<std::string> imglists = ESR::ScanNSortDirectory(argv[0], argv[1]);

	//load in model
	ESR::Regressor regressor;
	if(!regressor.loadModel(argv[2]))
		return -1;
	
	ESR::FaceDetector faceDetector;
	if(!faceDetector.loadModel(argv[3]))
		return -1;

	for(int j = 0; j < imglists.size(); j++)
	{
		std::cout << "Processing image " << imglists[j] << std::endl;
	
		Mat image_original = imread(imglists[j], CV_LOAD_IMAGE_COLOR);
		if(image_original.data == NULL)
		{
			std::cout << "Cannot open image from " << imglists[j] << std::endl;
			continue;
		}
		Mat image;
		cvtColor(image_original, image, CV_BGR2GRAY);

		cv::TickMeter tk;
		tk.start();
		vector<ESR::Bbox> faces = faceDetector.detectFace(image);
		tk.stop();
		std::cout<<"Face detect time cost: "<< tk.getTimeMilli() << " milli second" << std::endl; 
		vector<Mat> shapes;
		for(int i = 0 ; i < faces.size(); i++)
		{
			cv::TickMeter tk;
			tk.start();
			Mat shape;
			regressor.predict(image, faces[i], shape);
			tk.stop();
			std::cout<<"Face alignment time cost: "<< tk.getTimeMilli() << " milli second" << std::endl; 
			shapes.push_back(shape);
			ESR::dispImgWithDetectionAndLandmarks(image_original, shape, faces[i], false, false);
		}
		
		waitKey(100);
	}

	return 0;
}

int live_main(int argc, char *argv[])
{
	if (argc != 3) 
	{
		std::cout << "live model_file cascade_file image_file" << std::endl;
		return -1;
	}

	//load in model
	ESR::Regressor regressor;
	if(!regressor.loadModel(argv[0]))
		return -1;
	
	ESR::FaceDetector faceDetector;
	if(!faceDetector.loadModel(argv[1]))
		return -1;

	Mat image_original = imread(argv[2], CV_LOAD_IMAGE_COLOR);
	if(image_original.data == NULL)
	{
		std::cout << "Cannot open image from " << argv[2] << std::endl;
		return -1;
	}

	Mat image;
	cvtColor(image_original, image, CV_BGR2GRAY );

	cv::TickMeter tk;
	tk.start();
	vector<ESR::Bbox> faces = faceDetector.detectFace(image);
	tk.stop();
	std::cout<<"Face detect time cost: "<< tk.getTimeMilli() << " milli second" << std::endl; 
	
	vector<Mat> shapes;
	for(int i = 0 ; i < faces.size(); i++)
	{
		cv::TickMeter tk;
		tk.start();
		Mat shape;
		regressor.predict(image, faces[i], shape);
		tk.stop();
		std::cout<<"Face alignment time cost: "<< tk.getTimeMilli() << " milli second" << std::endl; 
		shapes.push_back(shape);
		ESR::dispImgWithDetectionAndLandmarks(image_original, shape, faces[i], false, false);
	}
		
	waitKey(0);
	return 0;
}

int camera_main(int argc, char *argv[]) 
{
	if (argc != 3) 
	{
		std::cout << "camera model_file cascade_file camera_index" << std::endl;
		return -1;
	}

	//load in model
	ESR::Regressor regressor;
	if(!regressor.loadModel(argv[0]))
		return -1;
	
	ESR::FaceDetector faceDetector;
	if(!faceDetector.loadModel(argv[1]))
		return -1;

	int index = atoi(argv[2]);
	VideoCapture stream(index);  
	while(1)
	{
		Mat image, shape, image_gray;
		stream.read(image);
		cvtColor(image, image_gray, CV_BGR2GRAY );
		cv::TickMeter tk;
		tk.start();
		vector<ESR::Bbox> faces = faceDetector.detectFace(image_gray);
		tk.stop();
		std::cout<<"Face detect time cost: "<< tk.getTimeMilli() << " milli second" << std::endl; 
	
		 vector<Mat> shapes;
		 for(int i = 0 ; i < faces.size(); i++)
		 {
			cv::TickMeter tk;
			tk.start();
		 	Mat shape;
			regressor.predict(image_gray, faces[i], shape);
			tk.stop();
			std::cout<<"Face alignment time cost: "<< tk.getTimeMilli() << " milli second" << std::endl;
			shapes.push_back(shape);
			ESR::dispImgWithDetectionAndLandmarks(image, shape, faces[i], false, false);
		 }
		 waitKey(1);
	}

	return 0;
}

