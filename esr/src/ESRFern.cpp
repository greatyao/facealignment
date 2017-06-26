#include "ESRFern.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>

#include "ESRUtils.hpp"

#include "ESRCommon.hpp"

using namespace cv;

namespace ESR
{
	void Fern::loadModel(std::ifstream& fin)
	{
		loadVal(fin, numFeature);
		loadVal(fin, numLandmark);
		featureLandmarks.create(numFeature, 2, CV_32S);
		featurePositions.create(numFeature, 4, CV_64F);
		featureThresholds.create(numFeature, 1, CV_64F);
		
		//load position indexed feature definition
		for(int i=0; i<numFeature; i++)
		{
			loadVal(fin, featurePositions.at<double>(i,0));
			loadVal(fin, featurePositions.at<double>(i,1));
			loadVal(fin, featurePositions.at<double>(i,2));
			loadVal(fin, featurePositions.at<double>(i,3));

			loadVal(fin, featureLandmarks.at<int>(i,0));
			loadVal(fin, featureLandmarks.at<int>(i,1));
			loadVal(fin, featureThresholds.at<double>(i));
		}
		
		//load bins' output table
		numBin = std::pow(2,numFeature);
		output.resize(numBin);
		for(int i = 0; i < numBin; i++)
		{
			Mat tmp(numLandmark, 2, CV_64F);
			for(int j =0; j< numLandmark; j++)
			{
				loadVal(fin, tmp.at<double>(j,0));
				loadVal(fin, tmp.at<double>(j,1)); 
			}
			output[i] = tmp;
		}

		return;
	}

	void Fern::predict(const cv::Mat& image, Bbox bbox, const RSTransform& t, const Mat& curShape, cv::Mat& deltadeltaShape)
	{
		//std::cout << "Fern::predict" <<std::endl;
		int index = 0;
		for(int i=0; i<numFeature; i++)
		{
			double pixelDiff = extractFeature(i,image,bbox,curShape,t);

			if(pixelDiff >= featureThresholds.at<double>(i))
			{
				index += 1<<i;
			}
		}

		//look up output table
		applyTransform(output[index], t, deltadeltaShape);
		return;
	}

	vector<Mat> Fern::train(const Mat& target, const Mat& densities, const Mat& covariance, const Mat& candidatePositions, const std::vector<int>& candidateLandmarks)
	{
		numFeature = NUM_FERN_FEATURE;
		numLandmark = target.cols/2;
		numBin = std::pow(2, numFeature);
		int numTraining = target.rows;
		int numCandidatePixels = densities.rows;
		output.resize(numBin);

		featurePositions.create(numFeature,4,CV_64F);
		featureLandmarks.create(numFeature,2,CV_32S);
		featurePixelIdx.create(numFeature,2,CV_32S);
		featureThresholds.create(numFeature,1,CV_64F);

		for(int i=0; i<numBin; i++)
		{
			output[i] = Mat::zeros(numLandmark,2, CV_64F);
		}

		//Fast Correlation Computation
		for(int i=0; i<numFeature; i++)
		{
			//randomly choose projection vector v
			RNG rng(getTickCount());
			Mat randomDir(numLandmark*2, 1, CV_64F);
			rng.fill(randomDir, RNG::UNIFORM, -1.0, 1.0);
			normalize(randomDir,randomDir);

			//project Y using v
			Mat projectedY;
			transpose(target * randomDir, projectedY);
			double projectedYvar = ESR::computeCovariance(projectedY,projectedY);

			//compute target-pixel covariance
			Mat tpcovar(numCandidatePixels, 1, CV_64F);
			for(int j=0; j<numCandidatePixels; j++)
			{
				tpcovar.at<double>(j,0) = ESR::computeCovariance(projectedY, densities.row(j));
			}

			double maxCorrelation = -100;
			int idx1 = -1;
			int idx2 = -1;
			bool first = true;
			int invalidCount = 0;
			for(int j = 0; j < numCandidatePixels; j++)
			{
				for(int k = 0; k < numCandidatePixels; k++)
				{
					if(k==j) continue;
					//here is the magic
					double numerator = tpcovar.at<double>(j,0) - tpcovar.at<double>(k,0);
					double denominator = sqrt( projectedYvar * (covariance.at<double>(j,j) + covariance.at<double>(k,k) - 2 * covariance.at<double>(j,k)));
					if(numerator == 0 && denominator == 0)
					{
						invalidCount ++;
						continue;
					}

					double correlation = std::abs(numerator/denominator);

					if(correlation > maxCorrelation || first)
					{
						first =  false;
						//igonore the pixel pairs already selected
						bool alreadySelected = false;
						for(int ii = 0; ii<i; ii++)
						{
							if((featurePixelIdx.at<int>(ii,0) == j &&
							   featurePixelIdx.at<int>(ii,1) == k) || 
							   (featurePixelIdx.at<int>(ii,0) == k &&
							   	featurePixelIdx.at<int>(ii,1) == j))
							{
								//std::cout << "already selected" << std::endl;
								alreadySelected = true;
								break;
							}
						}
						if(alreadySelected) continue;

						//update the max correlation
						maxCorrelation = correlation;
						idx1 = j;
						idx2 = k;
					}
				}
			}

			featurePixelIdx.at<int>(i,0) = idx1;
			featurePixelIdx.at<int>(i,1) = idx2;

			featureLandmarks.at<int>(i,0) = candidateLandmarks[idx1];
			featureLandmarks.at<int>(i,1) = candidateLandmarks[idx2];

			featurePositions.at<double>(i,0) = candidatePositions.at<double>(idx1,0);
			featurePositions.at<double>(i,1) = candidatePositions.at<double>(idx1,1);
			featurePositions.at<double>(i,2) = candidatePositions.at<double>(idx2,0);
			featurePositions.at<double>(i,3) = candidatePositions.at<double>(idx2,1);

			double max_abs = -1;
			for(int i=0; i<numTraining; i++)
			{
				double diff = std::abs(densities.at<double>(idx1,i) - densities.at<double>(idx2,i));
				max_abs = max_abs < diff? diff: max_abs;
			}

			featureThresholds.at<double>(i) = rng.uniform(-0.2*max_abs,0.2*max_abs);		
		}

		//computing the output table
		std::vector<int> binSize(numBin);
		std::vector<int> shapeBin(numTraining);
		for(int i=0; i<numTraining; i++)
		{
			//classification
			int index = 0;
			for(int j=0; j<numFeature; j++)
			{
				int idx1 = featurePixelIdx.at<int>(j,0);
				int idx2 = featurePixelIdx.at<int>(j,1);
				double diff = densities.at<double>(idx1,i) - densities.at<double>(idx2,i);
				if(diff >= featureThresholds.at<double>(j))
				{
					index += 1<<j;
				}
			}

			Mat outputX,outputY;
			transpose(target(Range(i,i+1),Range(0,numLandmark)), outputX);
			transpose(target(Range(i,i+1),Range(numLandmark,2*numLandmark)), outputY);

			output[index].col(0) += outputX;
			output[index].col(1) += outputY;

			binSize[index] ++;
			shapeBin[i] = index;
		}

		for(int i=0; i<binSize.size(); i++)
		{
			if(binSize[i] == 0)
			{
				continue;
			}
			output[i] = (1.0/((1.0+1000.0/binSize[i]) * binSize[i])) * output[i];
		}

		//now return the result
		vector<Mat> predicts(numTraining);
		for(int i=0; i<shapeBin.size(); i++)
		{
			output[shapeBin[i]].copyTo(predicts[i]);	
		}
		return predicts;
	}

	/**
	 * extract pose index feature
	 */
	double Fern::extractFeature(int featureIdx, const Mat& image, const Bbox& bbox, const Mat& curShape, const RSTransform& t)
	{
			int landmark1 = featureLandmarks.at<int>(featureIdx,0);
			int landmark2 = featureLandmarks.at<int>(featureIdx,1);

			double pos1x  = featurePositions.at<double>(featureIdx,0);
			double pos1y  = featurePositions.at<double>(featureIdx,1);
			double pos2x  = featurePositions.at<double>(featureIdx,2);
			double pos2y  = featurePositions.at<double>(featureIdx,3);

			double x1,y1,x2,y2;
			applyTransform(pos1x, pos1y, t, x1, y1);
			applyTransform(pos2x, pos2y, t, x2, y2);

			int realx1 = (x1  + curShape.at<double>(landmark1,0)) * (bbox.w/2.0) + bbox.cx;
			int realy1 = (y1  + curShape.at<double>(landmark1,1)) * (bbox.h/2.0) + bbox.cy;
			int realx2 = (x2  + curShape.at<double>(landmark2,0)) * (bbox.w/2.0) + bbox.cx;
			int realy2 = (y2  + curShape.at<double>(landmark2,1)) * (bbox.h/2.0) + bbox.cy;

			realx1 = max(min(realx1,image.cols-1),0);
			realy1 = max(min(realy1,image.rows-1),0);
			realx2 = max(min(realx2,image.cols-1),0);
			realy2 = max(min(realy2,image.rows-1),0);

			double pixelDiff = (int)image.at<uint8_t>(realy1, realx1) - 
				   (int)image.at<uint8_t>(realy2, realx2);

			return 	pixelDiff;	
	}

	void Fern::storeModel(std::ofstream& fout)
	{
		//std::cout << "Fern::storeModel" << std::endl;
		storeVal(fout, numFeature);
		storeVal(fout, numLandmark);

		for(int i=0; i<numFeature; i++)
		{
			storeVal(fout, featurePositions.at<double>(i,0));
			storeVal(fout, featurePositions.at<double>(i,1));
			storeVal(fout, featurePositions.at<double>(i,2));
			storeVal(fout, featurePositions.at<double>(i,3));

			storeVal(fout, featureLandmarks.at<int>(i,0));
			storeVal(fout, featureLandmarks.at<int>(i,1));

			storeVal(fout, featureThresholds.at<double>(i));
		}

		for(int i = 0; i<numBin; i++)
		{
			for(int j=0; j< numLandmark; j++)
			{
				storeVal(fout, output[i].at<double>(j,0));
				storeVal(fout, output[i].at<double>(j,1));
			}
		}

		return;		
	}

}
