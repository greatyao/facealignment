#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ESRFaceDetector.hpp"

using namespace std;

namespace ESR
{
	/**
	 * load the pretrained model from file
	 */
	bool FaceDetector::loadModel(std::string file)
	{
		if(!face_cascade.load( file ))
		{
			std::cout << "[Error](loadModel): error loading model from '" << file << "'" << std::endl;
			return false;
		}
		return true;
	}

	/**
	 * simply detect and return the face bounding box
	 */
	std::vector<Bbox> FaceDetector::detectFace(const cv::Mat& frame)
	{
		double scale = 1.3;
		std::vector<Bbox> result;
		std::vector<cv::Rect> faces;
		cv::Mat gray, smallImg(cvRound (frame.rows/scale), cvRound(frame.cols/scale), CV_8UC1);

		if(frame.channels() == 3 || frame.channels() == 4)
		{
			cvtColor(frame, gray, CV_BGR2GRAY);
		}
		else
		{
			gray = frame;
		}
		resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
    		equalizeHist( smallImg, smallImg );

		face_cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );
		for( size_t i = 0; i < faces.size(); i++ )
		{
			Bbox bbox(scale*faces[i].x, scale*faces[i].y,
				  scale*(faces[i].x + faces[i].width*0.5),
				  scale*(faces[i].y + faces[i].height*0.5),
				  scale*faces[i].width, scale*faces[i].height);
			result.push_back(bbox);
		}
		return result;
	}
}
