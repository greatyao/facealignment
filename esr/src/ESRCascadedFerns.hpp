#ifndef ESRCASCADEDFERNS_H
#define ESRCASCADEDFERNS_H

#include <fstream>
#include <vector>
#include "ESRFern.hpp"
#include "ESRBbox.hpp"
#include <opencv2/core/core.hpp>
#include <vector>


namespace ESR
{
	class CascadedFerns
	{
	public:
		void loadModel(std::ifstream& fin);
		void storeModel(std::ofstream& fout);
		void predict(const cv::Mat& image, Bbox bbox, const Mat& curShape, const Mat& meanShape, cv::Mat& shape);
		
		/**
		 * train the cascaded ferns
		 * @ret predicted regresstion offset
		 */
		std::vector<cv::Mat> train(
				   const std::vector<cv::Mat>& images, 
				   const std::vector<Bbox>& bboxes, 
				   const Mat& meanShape, 
				   const std::vector<cv::Mat>& target_shapes, 
				   const std::vector<cv::Mat>& currentShapes);

	private:
		int numRegressor;
		std::vector<Fern> regressors;

		/**
		 * generate candidate pixles used for feature selection
		 * @param num the number of candidate pixels to be generated
		 * @param meanShape meanShape
		 * @param&ret candidatePositions candidate points' local coordinates
		 * @param&ret candidateLandmarks candidate points' corresponding landmark index
		 */
		void generateCandidatePixels(int num, const Mat& meanShape, cv::Mat& candidatePositions,vector<int>& candidateLandmarks);

		/**
		 * compute densities of the candidate pixels on each shape
		 * @ret densities MxN matrix where M is number of landmark and N is the number of shapes
		 */
		void computeDensities(const vector<Mat>& images, 
									 const vector<Bbox>& bboxes, 
									 const vector<int>& candidateLandmarks, 
									 const Mat& candidatePositions, 
									 const std::vector<cv::Mat>& currentShapes, 
									 const Mat& meanShape, 
									 Mat& densities);

		/**
		 * compute the covariance matrix of features
		 */
		void computeCovariance(const Mat& densities, Mat &covariance);
	};
}


#endif