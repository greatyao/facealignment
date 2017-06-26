#ifndef ESRREGRESSOR_H
#define ESRREGRESSOR_H

#include <string>
#include <opencv2/core/core.hpp>
#include "ESRBbox.hpp"
#include "ESRCascadedFerns.hpp"

#include <vector>
#include <fstream>


#define DEBUG_MODE

using namespace cv;

namespace ESR
{
	class Regressor
	{

	public:
		/**
		 * load model from file
		 */
		bool loadModel(std::string filepath);

		/**
		 * load model from stream
		 */
		void loadModel(std::ifstream& fin);

		/**
		 * store model to file
		 */
		bool storeModel(std::string filepath);

		/**
		 * store model to stream
		 */
		void storeModel(std::ofstream& fout);

		/**
		 * predict landmark positions using the trained model
		 * @ret the predicted landmark position (numLandMark x n matrix)
		 */
		void predict(const Mat& image, Bbox bbox, Mat& shape);

		/**
		 * train the ESR model
		 */
		void train(const std::vector<Mat>& images, const std::vector<Bbox>& bbox, const std::vector<Mat>& shape);

#ifndef DEBUG_MODE
	private:
#endif
		int numRegressor;
		int numLandmark;
        Mat meanShape;
        std::vector<CascadedFerns> regressors;

        //training data
        std::vector<Mat> trainingShapes;
        std::vector<Bbox> trainingBboxes;
        
	};
}


#endif
