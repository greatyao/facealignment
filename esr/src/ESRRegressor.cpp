#include "ESRRegressor.hpp"

#include <fstream>
#include <iostream>

#include "ESRUtils.hpp"
#include "ESRCommon.hpp"


namespace ESR
{
	void Regressor::predict(const Mat& image, Bbox bbox, Mat& shape)
	{
		//use mean shape as initial shape
		Mat cur = meanShape.clone();
		for(int i=0; i<numRegressor; i++)
		{
			//compute delta shape
			Mat deltaShape;
			regressors[i].predict(image, bbox, cur, meanShape, deltaShape);

			//update current shape
			cur += deltaShape;
		}
		transformBBox2Image(cur, bbox, shape);
		return;
	}

	void Regressor::train(const std::vector<Mat>& images, 
						  const std::vector<Bbox>& bboxes, 
						  const std::vector<Mat>& target_shapes)
	{
		std::cout << "training ESR regressor" << std::endl;
		numRegressor = NUM_EXTERNAL_REGRESSOR;
		numLandmark = target_shapes[0].rows;
		trainingBboxes = bboxes;
		trainingShapes = target_shapes;

		//compute mean shape
		computeMeanShape(target_shapes, bboxes, meanShape);

		//transform target shape to normalized bbox space
		std::vector<Mat> targetShapes(target_shapes.size());
		for(int i=0; i<targetShapes.size(); i++)
		{
			Mat temp;
			transformImage2BBox(target_shapes[i],bboxes[i],temp);
			targetShapes[i] = temp;
		}

		//training data augmentation
		int augment = 10;
		int augmentNum = augment*images.size();
		std::vector<Mat> augmentedImages(augmentNum);
		std::vector<Mat> augmentedTargetShape(augmentNum);
		std::vector<Bbox> augmentedBboxes(augmentNum);
		std::vector<Mat> currentShapes(augmentNum);

		RNG rng(getTickCount());
		for(int i=0; i<images.size(); i++)
		{
			for(int j=0; j<augment; j++)
			{
				//randomly select initial shape
				int index = 0;
				do{
					index = rng.uniform(0, images.size());
				}while(index == i);

				//augment data
				augmentedImages[i * augment + j] = images[i];
				augmentedTargetShape[i * augment + j] = targetShapes[i];
				augmentedBboxes[i*augment + j] = bboxes[i];
				targetShapes[index].copyTo(currentShapes[i*augment + j]);
			}
		}

		//cascaded training
		regressors.resize(NUM_EXTERNAL_REGRESSOR);
		for(int i=0; i<numRegressor; i++)
		{
			std::cout << "-training external regressor [" << i << "]" << std::endl;
			vector<Mat> predicts = regressors[i].train(augmentedImages, augmentedBboxes, meanShape, augmentedTargetShape, currentShapes);
			//update currentShapes

			for(int j=0; j<predicts.size(); j++)
			{
				currentShapes[j] += predicts[j];
			}

			// Mat temp;
			// transformBBox2Image(currentShapes[0], bboxes[0], temp);
			// dispImgWithDetectionAndLandmarks(images[0], temp, bboxes[0], true);

		}
		return;
	}

	bool Regressor::loadModel(std::string filepath)
	{
		std::ifstream fin;
		fin.open(filepath.c_str(), std::ios::binary);
		if(!fin.good())
		{
			std::cout << "[Error](loadModel): unable to open '" << filepath << "'" << std::endl;
			return false;
		}
		loadModel(fin);
		fin.close();
		return true;
	}

	void Regressor::loadModel(std::ifstream& fin)
	{
		//number of external regressor
		loadVal(fin, numRegressor);

		//number of land mark
		loadVal(fin, numLandmark);

		//mean shape
		meanShape = Mat::zeros(numLandmark,2,CV_64F);
		for(int i = 0; i< numLandmark; i++)
		{
			loadVal(fin, meanShape.at<double>(i,0));
			loadVal(fin, meanShape.at<double>(i,1));
		}

		//training data (for multiple initialization)
		int numTraining;
		loadVal(fin, numTraining);
		trainingShapes.resize(numTraining);
		trainingBboxes.resize(numTraining);
		for (int i=0; i<numTraining; i++)
		{
			//training bbox
			double sx,sy,w,h,cx,cy;
			loadVal(fin, sx);
			loadVal(fin, sy);
			loadVal(fin, w);
			loadVal(fin, h);
			loadVal(fin, cx);
			loadVal(fin, cy);
			
			trainingBboxes[i] = Bbox(sx,sy,cx,cy,w,h);

			//training shape
			Mat shape(numLandmark, 2, CV_64F);
			for(int j=0; j<numLandmark; j++)
			{
				loadVal(fin, shape.at<double>(j,0));
				loadVal(fin, shape.at<double>(j,1));
			}
			trainingShapes[i] = shape;
		}

		//iteratively load model data into first level regressors
		regressors.resize(numRegressor);
		for(int i=0; i<regressors.size(); i++)
		{
			regressors[i].loadModel(fin);
		}

		std::cout << "[Info](loadModel): model loading completed." << std::endl;
	}

	bool Regressor::storeModel(std::string filepath)
	{
		std::ofstream fout;
		fout.open(filepath.c_str(), std::ios::binary);
		if(!fout.good())
		{
			std::cout << "[Error](storeModel): unable to open '" << filepath << "'" << std::endl;
			return false;			
		}
		storeModel(fout);
		fout.close();
		return true;
	}

	void Regressor::storeModel(std::ofstream& fout)
	{
		std::cout << "regressors::storeModel" << std::endl;
		storeVal(fout, numRegressor);
		storeVal(fout, numLandmark);
		for(int i=0; i<numLandmark; i++)
		{
			storeVal(fout, meanShape.at<double>(i,0));
			storeVal(fout, meanShape.at<double>(i,1));
		}
		int numTraining = trainingShapes.size();
		storeVal(fout, numTraining);
		for(int i=0; i<numTraining; i++)
		{
			const Bbox& bbox = trainingBboxes[i];
			storeVal(fout, bbox.sx);
			storeVal(fout, bbox.sy);
			storeVal(fout, bbox.w);
			storeVal(fout, bbox.h);
			storeVal(fout, bbox.cx);
			storeVal(fout, bbox.cy);

			for(int j=0; j<numLandmark; j++)
			{
				storeVal(fout, trainingShapes[i].at<double>(j,0));
				storeVal(fout, trainingShapes[i].at<double>(j,1));
			}
		}

		for(int i = 0; i< regressors.size(); i++)
		{
			regressors[i].storeModel(fout);
		}

		std::cout << "[Info](storeModel): model storing completed." << std::endl;

	}
}
