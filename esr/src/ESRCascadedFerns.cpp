#include "ESRCascadedFerns.hpp"

#include <iostream>
#include "ESRUtils.hpp"
#include "ESRCommon.hpp"
#include <algorithm>

using namespace cv;

namespace ESR
{
	void CascadedFerns::loadModel(std::ifstream& fin)
	{
		loadVal(fin, numRegressor);
		regressors.resize(numRegressor);
		for(int i=0; i<numRegressor; i++)
		{
			regressors[i].loadModel(fin);
		}
		return;
	}

	void CascadedFerns::predict(const Mat& image, Bbox bbox, const Mat& curShape, const Mat& meanShape, Mat& deltaShape)
	{
		RSTransform t;
		similarityTransform(meanShape, curShape, t);
		deltaShape = Mat::zeros(curShape.rows, curShape.cols, CV_64F);
		for(int i=0; i<regressors.size(); i++)
		{
			Mat deltadeltaShape;
			regressors[i].predict(image, bbox, t, curShape, deltadeltaShape);
			deltaShape += deltadeltaShape;
		}
		return;
	}

	std::vector<cv::Mat> CascadedFerns::train(
		   const std::vector<cv::Mat>& images, 
		   const std::vector<Bbox>& bboxes, 
		   const Mat& meanShape, 
		   const std::vector<cv::Mat>& targetShapes, 
		   const std::vector<cv::Mat>& currentShapes)
	{
		numRegressor = NUM_INTERNAL_REGRESSOR;
		int numTraining = images.size();

		//compute the regression target with respect to the mean shape
		Mat currentTarget;
		int numLandmark = meanShape.rows;
		currentTarget.create(numLandmark*2,numTraining,CV_64F);

		for(int i=0; i< numTraining; i++)
		{
			RSTransform transform;
			Mat regressionTarget, regressionTargetWarped;
			regressionTarget = targetShapes[i] - currentShapes[i];
			similarityTransform(currentShapes[i], meanShape, transform);
			applyTransform(regressionTarget, transform, regressionTargetWarped);
			regressionTargetWarped.col(0).copyTo(currentTarget(Range(0,numLandmark), Range(i,i+1)));
			regressionTargetWarped.col(1).copyTo(currentTarget(Range(numLandmark,numLandmark*2), Range(i,i+1)));
		}

		transpose(currentTarget, currentTarget);

		//randomly generate candidate pixels positions
		Mat candidatePositions;
		Mat densities, covariance;
		vector<int> candidateLandmarks;
		
		generateCandidatePixels(NUM_CANDIDATE_PIXEL, meanShape, candidatePositions, candidateLandmarks);
		computeDensities(images, bboxes, candidateLandmarks, candidatePositions, currentShapes, meanShape, densities);
		computeCovariance(densities, covariance);

		//cascadedly train the ferns
		regressors.resize(numRegressor);
		std::vector<Mat> predicts(numTraining);
		for(int i=0; i<predicts.size(); i++)
		{
			predicts[i] = Mat::zeros(numLandmark,2,CV_64F);
		}

		for(int i=0; i<numRegressor; i++)
		{
			std::cout << "--training internal regressor [" << i << "]" << std::endl;
			std::vector<Mat> fernPredict = regressors[i].train(currentTarget, densities, covariance, candidatePositions, candidateLandmarks);
			for(int j=0; j<numTraining; j++)
			{
				Mat predictT;
				transpose(fernPredict[j],predictT);
				currentTarget(Range(j,j+1),Range(0,numLandmark)) -= predictT(Range(0,1),Range::all());
				currentTarget(Range(j,j+1),Range(numLandmark,2*numLandmark)) -= predictT(Range(1,2),Range::all());
				predicts[j] += fernPredict[j];
			}
		}

		//transform the predicts from local space to normalized bbox space
		for(int i=0; i<predicts.size(); i++)
		{
			Mat current = currentShapes[i];
			RSTransform transform;
			Mat realPredict;
			similarityTransform(meanShape, current, transform);
			applyTransform(predicts[i], transform, realPredict);
			predicts[i] = realPredict;
		}

		return predicts;
	}

	void CascadedFerns::generateCandidatePixels(int num, const Mat& meanShape, cv::Mat& candidatePositions,vector<int>& candidateLandmarks)
	{
		candidatePositions.create(num,2,CV_64F);
		candidateLandmarks.resize(num);

		RNG rng(getTickCount());
		for(int i=0; i<num; i++)
		{
			//randomly generate sample points on normalized bounding box space
			double x = rng.uniform(-1.0,1.0);
			double y = rng.uniform(-1.0,1.0);

			//sampling in an uniform circle
	        // if(x*x + y*y > 1.0){
	        //     i--;
	        //     continue;
	        // }

			//find the nearest landmark in the meanshape
			double min_dist = 1e10;
			int nearest_idx = -1;
			for(int j=0; j<meanShape.rows; j++)
			{
				double dist = std::pow((meanShape.at<double>(j,0) - x),2) + 
							  std::pow((meanShape.at<double>(j,1) - y),2);  
				if(dist < min_dist)
				{
					min_dist = dist;
					nearest_idx = j;
				}
			}

			//put result
			candidatePositions.at<double>(i,0) = x - meanShape.at<double>(nearest_idx,0);
			candidatePositions.at<double>(i,1) = y - meanShape.at<double>(nearest_idx,1);
			candidateLandmarks[i] = nearest_idx;
		}

		return;
	}

	void CascadedFerns::computeDensities(const vector<Mat>& images, 
										 const vector<Bbox>& bboxes, 
										 const vector<int>& candidateLandmarks, 
										 const Mat& candidatePositions, 
										 const std::vector<cv::Mat>& currentShapes, 
										 const Mat& meanShape, 
										 Mat& densities)
	{
		int numShapes    = currentShapes.size();
		int numCandidate = candidateLandmarks.size();
		densities = Mat::zeros(numCandidate,numShapes,CV_64F);
		for(int i=0; i<numShapes; i++)
		{
			Mat curShape = currentShapes[i];

			//evaluate similarity transform from meanshape to current shape
			RSTransform transform;
			similarityTransform(meanShape, curShape, transform);

			//transform the candidate position from landmark's local space to uniform bbox space
			Mat transformedCandidatePositions;
			applyTransform(candidatePositions, transform, transformedCandidatePositions);

			for(int j = 0; j<numCandidate; j++)
			{

				int landmarkIdx = candidateLandmarks[j];

				//compute coordinate of feature pixels in normalized bbox space
				Mat coords = Mat::zeros(1,2,CV_64F);
				coords.at<double>(0) = transformedCandidatePositions.at<double>(j,0) + curShape.at<double>(landmarkIdx,0);
				coords.at<double>(1) = transformedCandidatePositions.at<double>(j,1) + curShape.at<double>(landmarkIdx,1);

				//convert this coordinate to image space
				Mat coordsImagespace;
				transformBBox2Image(coords, bboxes[i], coordsImagespace);

				//and btw make sure it falls inside the image
				int x = (int)std::min(std::max((int)coordsImagespace.at<double>(0,0),0),images[i].cols-1);
				int y = (int)std::min(std::max((int)coordsImagespace.at<double>(0,1),0),images[i].rows-1);

				//lookup and record the density
				double density = (double)images[i].at<uint8_t>(y, x);
				densities.at<double>(j,i) = density;
			}

		}
		return;
	}

	void CascadedFerns::computeCovariance(const Mat& densities, Mat& covariance)
	{
		int numCandidatePixel = densities.rows;
		covariance = Mat::zeros(numCandidatePixel,numCandidatePixel,CV_64F);
		for(int i=0; i<numCandidatePixel; i++)
		{
			covariance.at<double>(i,i) = 1.;
			for(int j=i; j<numCandidatePixel; j++)
			{
				Mat temp;
				multiply(densities.row(i), densities.row(j), temp);
				double covar = mean(temp)(0) - mean(densities.row(i))(0) * mean(densities.row(j))(0);
			    covariance.at<double>(i,j) = covar;
			    if(i!=j)
			    {
			    	covariance.at<double>(j,i) = covar;
			    }
			}
		}
		return;
	}

	void CascadedFerns::storeModel(std::ofstream& fout)
	{
		storeVal(fout, numRegressor);
		for(int i=0; i<numRegressor; i++)
		{
			regressors[i].storeModel(fout);
		}
		return;
	}

}
