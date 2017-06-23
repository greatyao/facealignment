#include <cstdio>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>

#include "lbf.hpp"

using namespace cv;
using namespace std;
using namespace lbf;

// dirty but works
void parseTxt(string &txt, vector<Mat> &imgs, vector<Mat> &gt_shapes, vector<BBox> &bboxes);

int test(void) {
    Config &config = Config::GetInstance();

    LbfCascador lbf_cascador;
    FILE *fd = fopen(config.saved_file_name.c_str(), "rb");
    lbf_cascador.Read(fd);
    fclose(fd);

    LOG("Load test data from %s", config.dataset.c_str());
    string txt = config.dataset + "/test.txt";
    vector<Mat> imgs, gt_shapes;
    vector<BBox> bboxes;
    parseTxt(txt, imgs, gt_shapes, bboxes);

    int N = imgs.size();
    lbf_cascador.Test(imgs, gt_shapes, bboxes);

    return 0;
}

int run(void) {
    Config &config = Config::GetInstance();
    FILE *fd = fopen((config.dataset + "/test.txt").c_str(), "r");
    assert(fd);
    int N;
    int landmark_n = config.landmark_n;
    fscanf(fd, "%d", &N);
    char img_path[256];
    double bbox[4];
    vector<double> x(landmark_n), y(landmark_n);

    LbfCascador lbf_cascador;
    FILE *model = fopen(config.saved_file_name.c_str(), "rb");
    lbf_cascador.Read(model);
    fclose(model);

    for (int i = 0; i < N; i++) {
        fscanf(fd, "%s", img_path);
        for (int j = 0; j < 4; j++) {
            fscanf(fd, "%lf", &bbox[j]);
        }
        for (int j = 0; j < landmark_n; j++) {
            fscanf(fd, "%lf%lf", &x[j], &y[j]);
        }
        Mat img = imread(img_path);
        // crop img
        double x_min, y_min, x_max, y_max;
        x_min = *min_element(x.begin(), x.end());
        x_max = *max_element(x.begin(), x.end());
        y_min = *min_element(y.begin(), y.end());
        y_max = *max_element(y.begin(), y.end());
	printf("1. %g %g %g %g\n", bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]);
        printf("2. %g %g %g %g\n", x_min, y_min, x_max, y_max);
        x_min = max(0., x_min - bbox[2] / 2);
        x_max = min(img.cols - 1., x_max + bbox[2] / 2);
        y_min = max(0., y_min - bbox[3] / 2);
        y_max = min(img.rows - 1., y_max + bbox[3] / 2);
	printf("3. %g %g %g %g\n", x_min, y_min, x_max, y_max);
        double x_, y_, w_, h_;
        x_ = x_min; y_ = y_min;
        w_ = x_max - x_min; h_ = y_max - y_min;
        BBox bbox_(bbox[0] - x_, bbox[1] - y_, bbox[2], bbox[3]);
        printf("4. %g %g %g %g\n", x_, y_, w_, h_);
        printf("5. %g %g %g %g\n", bbox[0] - x_, bbox[1] - y_, bbox[2], bbox[3]);
        Rect roi(x_, y_, w_, h_);
        img = img(roi).clone();

        Mat gray, shape;
        cvtColor(img, gray, CV_BGR2GRAY);
        LOG("Run %s", img_path);
	TIMER_BEGIN
        shape = lbf_cascador.Predict(gray, bbox_);
	LOG("alignment time costs %.4lf s", TIMER_NOW);
	TIMER_END
        img = drawShapeInImage(img, shape, bbox_);
        imshow("landmark", img);
        waitKey(0);
    }
    fclose(fd);
    return 0;
}

int live(int argc, const char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "model_name cascade_name image_name\n");
        return -1;
    }

    const char *model_name = argv[0];
    const char *cascade_name = argv[1];
    const char *image_name = argv[2];

    Config &config = Config::GetInstance();
    int N;
    int landmark_n = config.landmark_n;
    double bbox[4];
    double scale = 1.3;
    vector<double> x(landmark_n), y(landmark_n);
    vector<Rect> faces;
    vector<Mat> shapes;
    vector<BBox> bboxes;

    CascadeClassifier cascade;
    if(!cascade.load(cascade_name)){
        fprintf(stderr, "Could not load classifier cascade %s\n", cascade_name);
        return -1;
    }

    LbfCascador lbf_cascador;
    FILE *model = fopen(model_name, "rb");
    if(!model) {
        fprintf(stderr, "Could not load lbf model %s\n", model_name);
        return -1;
    }
    lbf_cascador.Read(model);
    fclose(model);

    Mat img = imread(image_name);
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    double t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0|CV_HAAR_SCALE_IMAGE,
        Size(30, 30));

    t = (double)cvGetTickCount() - t;
    printf("detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );

    cvtColor(img, gray, CV_BGR2GRAY);    
    for(vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++){
        Mat shape;
        bbox[0] = r->x*scale;
        bbox[1] = r->y*scale;
        bbox[2] = r->width*scale;
        bbox[3] = r->height*scale;
        BBox bbox_(bbox[0], bbox[1], bbox[2], bbox[3]);

        t =(double)cvGetTickCount();
        shape = lbf_cascador.Predict(gray, bbox_);
        t = (double)cvGetTickCount() - t;
        printf( "alignment time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
        shapes.push_back(shape);
	bboxes.push_back(bbox_);
    }

    for(int i = 0; i < shapes.size(); i++) {
        img = drawShapeInImage(img, shapes[i], bboxes[i]);
    }

    imshow("landmark", img);
    imwrite("result.jpg", img);
    waitKey(0);
    return 0;
}

int camera(int argc, const char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "model_name cascade_name camera_index\n");
        return -1;
    }

    const char *model_name = argv[0];
    const char *cascade_name = argv[1];
    int camera_index = atoi(argv[2]);

    Config &config = Config::GetInstance();
    int landmark_n = config.landmark_n;
    double bbox[4];
    double scale = 1.3;
    vector<double> x(landmark_n), y(landmark_n);
    cv::VideoCapture vc(camera_index);
    Mat frame, frameCopy, image, img;
    vc >> frame;

    CascadeClassifier cascade;
    if(!cascade.load(cascade_name)){
        fprintf(stderr, "Could not load classifier cascade %s\n", cascade_name);
        return -1;
    }

    LbfCascador lbf_cascador;
    FILE *model = fopen(model_name, "rb");
    if(!model) {
        fprintf(stderr, "Could not load lbf model %s\n", model_name);
        return -1;
    }
    lbf_cascador.Read(model);
    fclose(model);

    for(;;){
    	vector<BBox> bboxes;
    	vector<Mat> shapes;
    	vector<Rect> faces;
    	vc >> frame;
        Mat gray, smallImg;
        double t;
   	cvtColor(frame, gray, CV_BGR2GRAY);
 
    	smallImg = Mat(cvRound (frame.rows/scale), cvRound(frame.cols/scale), CV_8UC1 );
    	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
    	equalizeHist(smallImg, smallImg);
    	t = (double)cvGetTickCount();
    	cascade.detectMultiScale( smallImg, faces,
    		1.1, 2, 0|CV_HAAR_SCALE_IMAGE,
    		Size(30, 30) );
    	t = (double)cvGetTickCount() - t;
    	printf("detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
        if(faces.size() == 0) 
            continue;

    	for(vector<Rect>::iterator r = faces.begin(); r != faces.end(); r++){
    	    Mat shape;
    	    bbox[0] = r->x*scale;
    	    bbox[1] = r->y*scale;
            bbox[2] = r->width*scale;
    	    bbox[3] = r->height*scale;
    	    BBox bbox_(bbox[0], bbox[1], bbox[2], bbox[3]);
    	    t =(double)cvGetTickCount();
    	    shape = lbf_cascador.Predict(gray, bbox_);
    	    t = (double)cvGetTickCount() - t;
    	    printf("alignment time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    	    shapes.push_back(shape);
    	    bboxes.push_back(bbox_);
    	}

    	for(int i = 0; i < shapes.size(); i++) {
    	    frame = drawShapeInImage(frame, shapes[i], bboxes[i]);
    	}
    	imshow("camera", frame);

        if(waitKey(1) == 'q')
            break;
    }

    return 0;
}
