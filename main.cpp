#include "line.h"

int main()
{
	Linelidar filter;
	// 轮廓匹配去除背景影响
	Mat img1,img2,output_img;
	img1 = imread("/home/zhangwei/图片/mid.png");
	img2 = imread("/home/zhangwei/图片/left.png");
	filter.filter_img1(img1,img2,output_img);
    vector<Point> pt; 
	filter.grayline(output_img,pt);
	waitKey(0);
	return 0;
}

