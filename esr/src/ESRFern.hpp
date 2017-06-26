#ifndef ESRFERN_H
#define ESRFERN_H

#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>

#include "ESRBbox.hpp"
#include "ESRUtils.hpp"

using namespace cv;

namespace ESR
{
	class Fern
	{
	public:
		void loadModel(std::ifstream& fin);
		void storeModel(std::ofstream& fout);
		void predict(const cv::Mat& image, Bbox bbox, const RSTransform& t, const Mat& curShape, cv::Mat& deltaShape);
		vector<Mat> train(const Mat& target, const Mat& densities, const Mat& covariance, const Mat& candidatePositions, const std::vector<int>& candidateLandmarks);

	private:
		//number of fern feature, bin
		int numFeature;
		int numBin;

		//shape indexed features
		int numLandmark;
		Mat featureLandmarks;
		Mat featurePositions;
		Mat featurePixelIdx;

		//threasholds for each feature
		Mat featureThresholds;

		//output table
		std::vector<Mat> output;

		double extractFeature(int featureIdx, const Mat& image, const Bbox& bbox, const Mat& curShape, const RSTransform& t);
	};
}

#endif