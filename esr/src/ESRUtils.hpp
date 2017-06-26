/**
 * Utility functions used in ESR
 */

#ifndef ESRUTILS_H
#define ESRUTILS_H
#include <opencv2/core/core.hpp>
#include <string>
#include <fstream>

#include "ESRBbox.hpp"

namespace ESR
{

////Image IO and visualization helper functions

	/**
	 * display the image by always opening a new window and wait for key to close.
	 */
	void dispImg(cv::Mat& mat, bool closeByKey = false, bool alwaysNewWindow = true);
	void dispImg(const std::string& filename, bool closeByKey = false, bool alwaysNewWindow = true);
	void dispImgWithDetection(cv::Mat& mat, const Bbox& bbox, bool closeByKey = false, bool alwaysNewWindow = true);
	void dispImgWithDetection(const std::string& filename, const Bbox& bbox, bool closeByKey = false, bool alwaysNewWindow = true);
	void dispImgWithLandmarks(cv::Mat& mat, const cv::Mat& landmarks, bool closeByKey = false, bool alwaysNewWindow = true);
	void dispImgWithLandmarks(const std::string& filename, const cv::Mat& landmarks, bool closeByKey = false, bool alwaysNewWindow = true);
	void dispImgWithDetectionAndLandmarks(cv::Mat& mat, const cv::Mat& landmarks, const Bbox& bbox, bool closeByKey = false, bool alwaysNewWindow = true);
	void dispImgWithDetectionAndLandmarks(const std::string& filename, const cv::Mat& landmarks, const Bbox& bbox, bool closeByKey = false, bool alwaysNewWindow = true);

	/**
	 * safe wrapper of image write function
	 */
	void saveImg(const std::string& filename, cv::Mat& mat);

	/**
	 * read file as gray scale image
	 */
	void readImgGray(const std::string& filename, cv::Mat& result);

	/**
	 * compute Pearson Correlation between the two vectors
	 */
	double pearsonCorrelation(const cv::Mat& vec1, const cv::Mat& vec2);

	/**
	 * compute covariance between two vectors
	 */
	double computeCovariance(const cv::Mat& vec1, const cv::Mat& vec2);

	/**
	 * transform the coordates from bbox's normalized space to image space
	 */
	void transformBBox2Image(const cv::Mat& shape, const Bbox& bbox, cv::Mat& result);

	/**
	 * transform the coordates from image space to tbbox's normalized space
	 */
	void transformImage2BBox(const cv::Mat& shape, const Bbox& bbox, cv::Mat& result);

	/**
	 * project the shape from image1's bbox into image2's bbox
	 *
	 * @param shape1 shape1's landmarks' coordinates in image space
	 * @param bbox1  shape1's corresponding bounding box
	 * @param bbox2  the bounding box where shape1 is going to be projected to
	 * @return projected shape1's landmarks' coordinate in image space
	 */
	void projectBbox2Bbox(const cv::Mat& shape1, Bbox bbox1, Bbox bbox2, cv::Mat& result);


	struct RSTransform
	{
		cv::Mat  rotation;
		double   scale;
	};

	/**
	 * compute the similarity transform to project shape1 to shape2
	 * 
	 * Note: we only compute the rotation and scaling component, because translation
	 * it not needed in our algorithm.
	 *
	 * @param shape1 src shape
	 * @param shape2 dest shape
	 * @param rotation 2x2 rotation matrix
	 * @param scale scale
	 */
	void similarityTransform(const cv::Mat& shape1, const cv::Mat& shape2, RSTransform& transform);

	/**
	 * Apply ratation and scaling to shape
	 */
	void applyTransform(const cv::Mat& shape, const RSTransform& transform, cv::Mat& result);

	/**
	 * Apply ratation and scaling to shape
	 */
	void applyTransform(double x, double y, const RSTransform& transform, double& resultx, double& resulty);

	/**
	 * store value to file
	 */
	template <class T>
	inline void storeVal(std::ofstream& fout, T val)
	{
		fout.write((char*)&val, sizeof(T));
	}

	/**
	 * store value to file
	 */
	template <class T>
	inline void loadVal(std::ifstream& fin, T& val)
	{
		fin.read((char*)&val, sizeof(val));
	}

	/**
	 * compute mean shape from a set of shapes
	 * @param shapes: shapes defined in image space
	 */
	void computeMeanShape(const std::vector<cv::Mat>& shapes, const std::vector<Bbox>& bboxes, cv::Mat& meanShape);

	/**
	 * Scan all files with the "extension" under "path" directory and sort them
	 * @param path: the directory 
	 * @param string: the file extension
	 */
	std::vector<std::string> ScanNSortDirectory(const std::string &path, const std::string &extension);

}



#endif
