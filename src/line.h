#ifndef LINE_H
#define LINE_H

#include <vector>
#include "Eigen/Core"
#include <Eigen/Dense>
#include<opencv2/core/core.hpp>
#include<ctime>
#include<cmath>
#include<opencv2/calib3d/calib3d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgcodecs.hpp>
#include<iostream>
#include<fstream>
#include<stdlib.h>
#include<random>

using namespace std;
using namespace cv;


struct normalWithVariance
{
  double variance;
  Eigen::Vector2d normal;
};


struct ModelCoefficients
{
	std::vector<float> values;
};

struct ImageContour
{
    Mat src;
    vector<vector<Point> > contours;
	  vector<Vec4i> hierarchy;
};
ImageContour imagecontour(Mat& input);
class Linelidar
{
private:
public:
    void zhang(Mat& input, Mat& output);
    void point(Mat& inputimg, vector<Point2d>& pt);
    double ijpixel(double& x, double& y, Mat& m);
    void CalcNormVec(Point2d ptA, Point2d ptB, Point2d ptC, double pfCosSita, double pfSinSita);
    void filter_img1(Mat& img1, Mat& img2,Mat& img3);
    void StegerLine(Mat& img ,vector<Point>& points);
    void grayline(Mat& img,vector<Point>& pts);
    void GetSample(int& indexe_size,vector<int>& index);
    void fitcicle(vector<Point2f> points,Point2f& circleCenter,float& radius);
    void RanSanCirfit(vector<Point2f> points,Point2f& circleCenter,float& radius);
    void choseline(Mat& src);
    void leastsquaresfit(vector<float> &x,vector<float> &y);
    void Interpolation(vector<float> &x,vector<float> &y,float xx); 
};
#endif