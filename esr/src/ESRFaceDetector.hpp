#ifndef ESRFACEDETECTOR_H
#define ESRFACEDETECTOR_H

#include "ESRBbox.hpp"
#include <string>
#include <vector>
#include <opencv2/objdetect/objdetect.hpp>

namespace ESR
{
	class FaceDetector
	{

public:
	/**
	 * load the pretrained model from file
	 */
	bool loadModel(std::string file);


	/**
	 * simply detect and return the face bounding box
	 */
	std::vector<Bbox> detectFace(const cv::Mat& frame);

private:

	cv::CascadeClassifier face_cascade;

	};
}

#endif
