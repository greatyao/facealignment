#include "ESRUtils.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;


namespace ESR
{
	void dispImg(Mat& mat, bool closeByKey, bool alwaysNewWindow)
	{
		const static std::string titlePrefix = "dispImg";
		static int idx = 0;

		if(mat.data == NULL)
		{
			std::cout << "[Error](dispImg): the matrix contains no data" << std::endl;
			return;
		}
		idx ++;
		std::string windowName = alwaysNewWindow?titlePrefix + std::to_string(idx):titlePrefix;
		if(alwaysNewWindow) namedWindow(windowName, WINDOW_AUTOSIZE);
		imshow(windowName, mat);
		if(closeByKey)
		{
			waitKey(0);
			destroyWindow(windowName);
		}
		return;
	}

	void dispImg(const std::string& filename, bool closeByKey, bool alwaysNewWindow)
	{
		Mat mat = imread(filename, CV_LOAD_IMAGE_UNCHANGED);

		if(mat.data == NULL)
		{
			std::cout << "[Error](dispImg): cannot read image from '" << filename << "'" << std::endl;
			return;
		}		

		dispImg(mat, closeByKey, alwaysNewWindow);
		return;
	}

	void dispImgWithDetection(cv::Mat& mat, const Bbox& bbox, bool closeByKey, bool alwaysNewWindow)
	{
		rectangle(mat, Point(bbox.sx, bbox.sy), Point(bbox.sx + bbox.w, bbox.sy + bbox.h), Scalar(0.0,0.0,255.0,1.0));

		dispImg(mat, closeByKey, alwaysNewWindow);

		return;
	}

	void dispImgWithDetection(const std::string& filename, const Bbox& bbox, bool closeByKey, bool alwaysNewWindow)
	{
		Mat mat = imread(filename, CV_LOAD_IMAGE_UNCHANGED);

		if(mat.data == NULL)
		{
			std::cout << "[Error](dispImgWithDetection): cannot read image from '" << filename << "'" << std::endl;
			return;
		}
		dispImgWithDetection(mat,bbox,closeByKey, alwaysNewWindow);

		return;
	}

	void dispImgWithLandmarks(cv::Mat& mat, const cv::Mat& landmarks, bool closeByKey, bool alwaysNewWindow)
	{
		for(int i=0; i<landmarks.rows; i++)
		{
			circle(mat, Point(landmarks.at<double>(i,0), landmarks.at<double>(i,1)), 3, Scalar(0.0,255.0,0.0,255.0),CV_FILLED);;
		}
		dispImg(mat, closeByKey, alwaysNewWindow);

		return;
	}

	void dispImgWithLandmarks(const std::string& filename, const Mat& landmarks, bool closeByKey, bool alwaysNewWindow)
	{
		Mat mat = imread(filename, CV_LOAD_IMAGE_UNCHANGED);

		if(mat.data == NULL)
		{
			std::cout << "[Error](dispImgWithLandmarks): cannot read image from '" << filename << "'" << std::endl;
			return;
		}

		dispImgWithLandmarks(mat, landmarks, closeByKey, alwaysNewWindow);

		return;		
	}

	void dispImgWithDetectionAndLandmarks(cv::Mat& mat, const cv::Mat& landmarks, const Bbox& bbox, bool closeByKey, bool alwaysNewWindow)
	{
		rectangle(mat, Point(bbox.sx, bbox.sy), Point(bbox.sx + bbox.w, bbox.sy + bbox.h), Scalar(0.0,0.0,255.0,1.0));
		for(int i=0; i<landmarks.rows; i++)
		{
			circle(mat, Point(landmarks.at<double>(i,0), landmarks.at<double>(i,1)), 3, Scalar(0.0,255.0,0.0,255.0),CV_FILLED);;
		}
		dispImg(mat, closeByKey, alwaysNewWindow);
		return;		
	}

	void dispImgWithDetectionAndLandmarks(const std::string& filename, const cv::Mat& landmarks, const Bbox& bbox, bool closeByKey, bool alwaysNewWindow)
	{
		Mat mat = imread(filename, CV_LOAD_IMAGE_UNCHANGED);
		if(mat.data == NULL)
		{
			std::cout << "[Error](dispImgWithDetectionAndLandmarks): cannot read image from '" << filename << "'" << std::endl;
			return;
		}
		dispImgWithDetectionAndLandmarks(mat,landmarks,bbox,closeByKey, alwaysNewWindow);
		return;		
	}

	void saveImg(const std::string& filename, cv::Mat& mat)
	{
		bool result = imwrite(filename, mat);
		if(!result)
		{
			std::cout << "[Error](saveImg): cannot save image to '" << filename << "'" << std::endl;
		}
		return;
	}

	void readImgGray(const std::string& filename, cv::Mat& result)
	{
		result = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		if(result.data == NULL)
		{
			std::cout << "[Error](readImgGray): cannot read image from '" << filename << "'" << std::endl;
		}
		return;
	}

	void transformBBox2Image(const cv::Mat& shape, const Bbox& bbox, cv::Mat& result)
	{
		shape.copyTo(result);

		for (int i=0; i<result.rows; i++)
		{
			result.row(i).at<double>(0) = result.row(i).at<double>(0) * (bbox.w / 2.0) + bbox.cx;
			result.row(i).at<double>(1) = result.row(i).at<double>(1) * (bbox.h / 2.0) + bbox.cy;
		}
		return;
	}

	void transformImage2BBox(const cv::Mat& shape, const Bbox& bbox, cv::Mat& result)
	{

		shape.copyTo(result);

		for (int i=0; i<result.rows; i++)
		{
			result.row(i).at<double>(0) = (result.row(i).at<double>(0) - bbox.cx)/(bbox.w/2.0);
			result.row(i).at<double>(1) = (result.row(i).at<double>(1) - bbox.cy)/(bbox.h/2.0);
		}
		return;
	}

	void projectBbox2Bbox(const cv::Mat& shape1, Bbox bbox1, Bbox bbox2, cv::Mat& result)
	{
		transformImage2BBox(shape1,bbox1, result);
		transformBBox2Image(result, bbox2, result);
		return;
	}

	double pearsonCorrelation(const cv::Mat& vec1, const cv::Mat& vec2)
	{
		//compute mean and standard deviation
		Scalar m1,m2;
		Scalar std1,std2;
		meanStdDev(vec1,m1,std1);
		meanStdDev(vec2,m2,std2);

		//compute covariance
		Mat vec3;
		multiply(vec1 - m1, vec2 - m1, vec3);
		double covariance = mean(vec3)[0];
		
		//compute pearson correlation
		//which is defined as covar(v1,v2)/(std(v1) * std(v2))
		double correlation = covariance / (std1[0] * std2[0]);

		return correlation;
	}

	double computeCovariance(const cv::Mat& vec1, const cv::Mat& vec2)
	{
		Mat vec1xvec2;
		multiply(vec1, vec2, vec1xvec2);
		double covar = mean(vec1xvec2)[0] - mean(vec1)[0] * mean(vec2)[0];
		return covar;
	}

	//compute the similarity transformation from shape1 to shape2
	void similarityTransform(const Mat& shape1, 
							 const Mat& shape2, 
							 RSTransform& transform)
	{
	    transform.rotation = Mat::zeros(2,2,CV_64FC1);
	    transform.scale = 0;
	    
	    // center the data
	    double centerx1 = mean(shape1.col(0))[0]; 	double centery1 = mean(shape1.col(1))[0];
	    double centerx2 = mean(shape2.col(0))[0];	double centery2 = mean(shape2.col(1))[0];
	    Mat x1,y1,x2,y2;
	    shape1.col(0).copyTo(x1);	shape1.col(1).copyTo(y1);
	    shape2.col(0).copyTo(x2);	shape2.col(1).copyTo(y2);
	    x1 -= centerx1;	y1 -= centery1;
	    x2 -= centerx2;	y2 -= centery2;

	    //compute a and b
	    double tmp = (x1.dot(x1) + y1.dot(y1));
	    double a   = (x1.dot(x2) + y1.dot(y2))/tmp;
	    double b   = (x1.dot(y2) - y1.dot(x2))/tmp;

	    //compute rotation and scaling
	    transform.scale = std::sqrt(a*a+b*b);
	    transform.rotation.create(2,2,CV_64F);
	    transform.rotation.at<double>(0,0) = a;	transform.rotation.at<double>(0,1) = -b;
	    transform.rotation.at<double>(1,0) = b;	transform.rotation.at<double>(1,1) = a;

	    return;
	}

	void applyTransform(const cv::Mat& shape, const RSTransform& transform, cv::Mat& result)
	{
		Mat rotationT;
		transpose(transform.rotation, rotationT);
		result = transform.scale * shape * rotationT;
		return;
	}

	void applyTransform(double x, double y, const RSTransform& transform, double& resultx, double& resulty)
	{
		resultx = (transform.rotation.at<double>(0,0) * x + transform.rotation.at<double>(0,1) * y) * transform.scale;
		resulty = (transform.rotation.at<double>(1,0) * x + transform.rotation.at<double>(1,1) * y) * transform.scale;
		return;
	}

	void computeMeanShape(const std::vector<Mat>& shapes, const std::vector<Bbox>& bboxes, Mat& meanShape)
	{
		if(shapes.size() == 0) return;
		meanShape = Mat::zeros(shapes[0].rows, shapes[0].cols, CV_64F);
		for(int i=0; i<shapes.size(); i++)
		{
			Mat normalizedShape;
			transformImage2BBox(shapes[i], bboxes[i], normalizedShape);
			meanShape += normalizedShape;
		}
		meanShape /= shapes.size();
		return;
	}

	static int str_compare(const void *arg1, const void *arg2)
	{
	    return strcmp((*(std::string*)arg1).c_str(), (*(std::string*)arg2).c_str());
	}

	#ifdef WIN32
	#include <direct.h>
	#include <io.h>
	std::vector<std::string> ScanNSortDirectory(const std::string &path, const std::string &extension)
	{
	    WIN32_FIND_DATA wfd;
	    HANDLE hHandle;
	    std::string searchPath, searchFile;
	    std::vector<std::string> vFilenames;
		int nbFiles = 0;
	    
		searchPath = path + "/*" + extension;
		hHandle = FindFirstFile(searchPath.c_str(), &wfd);
		if (INVALID_HANDLE_VALUE == hHandle)
	    {
			fprintf(stderr, "ERROR(%s, %d): Cannot find (*.%s)files in directory %s\n",
				__FILE__, __LINE__, extension.c_str(), path.c_str());
			exit(0);
	    }
	    do
	    {
		//. or ..
		if (wfd.cFileName[0] == '.')
		{
		    continue;
		}
		// if exists sub-directory
		if (wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
		    continue;
		    }
		else//if file
		{
				searchFile = path + "/" + wfd.cFileName;
				vFilenames.push_back(searchFile);
				nbFiles++;
			}
	    }while (FindNextFile(hHandle, &wfd));

	    FindClose(hHandle);

	    // sort the filenames
	    qsort((void *)&(vFilenames[0]), (size_t)nbFiles, sizeof(string), str_compare);

	    return vFilenames;
	}

	#else

	#include <dirent.h>
	#include <sys/types.h>
	#include <sys/stat.h>
	#include <unistd.h>

	#define MAX_PATH 1024

	static int match(const char* s1, const char* s2)
	{
		int diff = strlen(s1) - strlen(s2);
		if(diff >= 0 && !strcmp(s1+diff, s2))
			return 1;
		return 0;
	}

	std::vector<std::string> ScanNSortDirectory(const std::string &path, const std::string &extension)
	{
		struct dirent *d;
		DIR* dir;
		struct stat s;
		char fullpath[MAX_PATH];
		std::vector<std::string> allfiles;
		int num = 0;

		dir = opendir(path.c_str());
		if(dir == NULL)	
		{
			 fprintf(stderr, "Can not open directory %s\n", path.c_str());
			 exit(0);
		}

		while(d = readdir(dir))
		{
			sprintf(fullpath, "%s/%s", path.c_str(), d->d_name);
			if(stat(fullpath, &s) != -1)
			{
				if(S_ISDIR(s.st_mode))
					continue;
				if(match(d->d_name, extension.c_str()))
				{
					allfiles.push_back(std::string(fullpath));
					num++;
				}
			}

		}
		closedir(dir);
		qsort((void*)&(allfiles[0]), size_t(num), sizeof(std::string), str_compare);
		return allfiles;
	}

	#endif
}
