#include "line.h"

int main()
{
	Linelidar filter;
	// Mat input_img,img,img_gray,output_img;
	// 轮廓匹配去除背景影响
	Mat img1,img2,output_img;
	img1 = imread("/home/zhangwei/图片/mid.png");
	img2 = imread("/home/zhangwei/图片/left.png");
	filter.filter_img1(img1,img2,output_img);

    vector<Point> pt; 
	filter.grayline(output_img,pt);




	// img = imread("/home/zhangwei/图像文件/751.png");
	// cvtColor(img, img_gray, COLOR_RGB2GRAY);
	// GaussianBlur(img_gray,img_gray, Size(3, 3), 0);
	// threshold(img_gray, input_img,150, 255, 3);
	// vector<int>  sample_indexes;
	// vector<int>  index;
	// // imshow("img",img);
	// Mat out;
	// Canny(input_img,out,200,300,3);
    
    // for(int i = 1;i<pt.size() -1 ;i++)
	// {
	// 	int x1 = pt[i].x;
	// 	int x2 = pt[i + 1].x;
	// 	int y1 = pt[i].y;
	// 	int y2 = pt[i + 1].y;
	// 	double distance =  sqrt(pow(x2 - x1,2) + pow(y2 -y1,2));
	// 	if(abs(x2-x1)>2 && abs(y2-y1)>2)
	// 	{
	// 		// cout<<"x1=  "<<x1<<"  "<<"y1=  "<<y1<<endl;
	// 		// cout<<"x2=  "<<x2<<"  "<<"y2=  "<<y2<<endl;
	// 	}
	// }
    
/////////////////////////////////////////////////////////
    // Mat imgcontour = Mat::zeros(img.size(),CV_8UC1);
	// cvtColor(imgcontour, imgcontour, CV_GRAY2BGR);
    // vector<Point> ptt;
	// vector<Point> pttt;  
	// ptt.reserve(pt.size());
	// for(int i = 1;i<pt.size() -1 ;i++)
	// {
	// 	float k = -3.2878e-006;
	// 	double distance =  sqrt(pow(ptt[i].x - pt[i].x,2) + pow(ptt[i].y -pt[i].y,2));
	// 	int rd = sqrt(pow(pt[i].x - 320,2) + pow(pt[i].y -240,2));
	// 	int ru = sqrt(pow(ptt[i].x - 320,2) + pow(ptt[i].y -240,2));
    //     ptt[i].x = (pt[i].x - 320)*(1 + k * rd * rd);
	// 	ptt[i].y = (pt[i].y - 320)*(1 + k * rd * rd);
        // cout<<"x"<< ptt[i].x<<"  "<<"y"<< ptt[i].y<<endl;
		// circle(imgcontour, Point(ptt[i].x, ptt[i].y), 0.5, Scalar(0, 0, 255));
	//}
//////////////////////////////////////////////////////////
     


    // vector<float> x = {1.0,2.5,3.0};
    // vector<float> y = {6.0,9.0,10.0};
	// vector<float> x = {35.1,29.7,30.8,58.8,61.4,71.3,74.4,76.6,70.7,57.5,46.4,28.9,28.1};//
    // vector<float> y = {10.98,11.13,12.51,8.40,9.27,8.73,6.63,8.50,7.82,9.14,8.24,12.19,11.88};//


	// vector<float> x = {1.236,5.362,3.365,9.654,45.512,95.264,78.651,16.153,25.251,56.154};//
    // vector<float> y = {-1.91209,-8.87528,-6.57621,-6.59888,-1.69338,-2.28951,-6.13382,-7.41196,-5.21807,-7.22801};//

    //插值
    // vector<float> x = {82.0,130.0,156.0,176.0,190.0,197.0,222.0};
    // vector<float> y = {30.0,60.0,90.0,120.0,150.0,180.0,210.0};
	// filter.leastsquaresfit(x,y);
	// float t = 100.0;
	// filter.Interpolation(x,y,t);
    
	// 随机数据
    // random_device rd; //随机数发生器
    // mt19937 mt(rd()); //随机数引擎
	// uniform_int_distribution<int> dist(0,360);
	// normal_distribution<double> distr(0,30);
	// Point2f center(250,250);
	// vector<Point2f> pts;
	// for(int i = 1;i<150;i++)
	// {
	// 	float r = 150;
	// 	if(i % 2 == 0) { r += distr(mt); }
	// 	double theta = dist(mt);
	// 	float x = center.x + r * cos(theta / 180.0 * 3.1415926);
	// 	float y = center.y + r * sin(theta / 180.0 * 3.1415926);
    //     pts.emplace_back(x,y);
	// }
	//

    // 测试圆参数计算
	// vector<Point2f> points;
	// points = {{2,0},{2,4},{4,2}};
	// Point2f circleCenter;
	// float radius = 0;
    // filter.fitcicle(points,circleCenter,radius);
    // cout<<"radius"<<radius<<endl;
	// cout<<"circleCenter.x"<<" "<< circleCenter.x<<" "<<"circleCenter.y"<<" "<< circleCenter.y<<endl;
	// // 





	//  Mat img4; 
	//  img4 = imread("/home/zhangwei/矫正图片/757.png");  
	//  vector<Point> pt; 
	//  filter.StegerLine(img4,pt);
	// //  filter.choseline(img4);
    //  imshow("StegerLine", img4);
    //  vector<Point2f> ptf(pt.begin(),pt.end());






    //  cout<<"ptf"<<ptf<<endl;
	// // 
	//  Point2f zww;
	//  float lss = 0;
	//  filter.RanSanCirfit(ptf,zww,lss);
	//  cout<<"circleCenter_x = "<<" "<< zww.x<<" "<<"circleCenter_y = "<<" "<< zww.y<<endl;
	//  cout<<" radius = "<<" "<< lss <<endl;
	 waitKey(0);
	return 0;
}

