#include "ESRFaceDetector.hpp"

#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>

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
		std::vector<Bbox> result;
		std::vector<cv::Rect> faces;
		cv::Mat frame_gray;
		if(frame.channels() == 3 || frame.channels() == 4)
		{
			cvtColor( frame, frame_gray, CV_BGR2GRAY );
		}
		else
		{
			frame_gray = frame.clone();
		}
		cv::equalizeHist( frame_gray, frame_gray );
		face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );
		for( size_t i = 0; i < faces.size(); i++ )
		{
			Bbox bbox(faces[i].x, faces[i].y, faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5, faces[i].width, faces[i].height);
			result.push_back(bbox);
		}
		return result;
	}
}
