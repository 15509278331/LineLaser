#include "line.h"

ModelCoefficients LeastSquaresFit(vector<float> &x,vector<float> &y)
{
    ModelCoefficients coeff;
    int N = 3;  //设置多项式阶数
    int num = x.size();
    Mat X = Mat(num,N + 1,CV_64FC1);
    Mat Y = Mat(num,1,CV_64FC1);
    Mat A;
	// 构造X矩阵
    for(int i = 0;i<num;i++)
    {
	   for(int n = 0,dex = 0;n < N + 1 ;++n,++dex)
	   {
		   X.at<double>(i,dex) = pow(x[i],n);
	   }
	   X.at<double>(i,0) = 1;
    }
	// 构造Y矩阵
    for(int j = 0;j < y.size() ;j++ )
    {
	   Y.at<double>((0,j)) = y[j];
    }
	// 解算A矩阵
    A = ((X.t() * X).inv()) * (X.t() * Y);
	for(int i = 0;i < N + 1;i++)
	{
       coeff.values.push_back(A.at<double>(i,0));
	}
    return coeff;
}

void Linelidar::zhang(Mat& input, Mat& output)
{
	Mat copymat;
	input.copyTo(copymat);
	int k = 0;
	//防止溢出
	while (1)
	{
		k++;
		bool stop = false;
		//step1
		for (int i = 1; i < input.cols - 1; i++)
			for (int j = 1; j < input.rows - 1; j++)
			{
				if (input.at<uchar>(j, i) > 0)
				{
					int p1 = int(input.at<uchar>(j, i)) > 0 ? 1 : 0;
					int p2 = int(input.at<uchar>(j - 1, i)) > 0 ? 1 : 0;
					int p3 = int(input.at<uchar>(j - 1, i + 1)) > 0 ? 1 : 0;
					int p4 = int(input.at<uchar>(j, i + 1)) > 0 ? 1 : 0;
					int p5 = int(input.at<uchar>(j + 1, i + 1)) > 0 ? 1 : 0;
					int p6 = int(input.at<uchar>(j + 1, i)) > 0 ? 1 : 0;
					int p7 = int(input.at<uchar>(j + 1, i - 1)) > 0 ? 1 : 0;
					int p8 = int(input.at<uchar>(j, i - 1)) > 0 ? 1 : 0;
					int p9 = int(input.at<uchar>(j - 1, i - 1)) > 0 ? 1 : 0;
					int np1 = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
					int sp2 = (p2 == 0 && p3 == 1) ? 1 : 0;
					int sp3 = (p3 == 0 && p4 == 1) ? 1 : 0;
					int sp4 = (p4 == 0 && p5 == 1) ? 1 : 0;
					int sp5 = (p5 == 0 && p6 == 1) ? 1 : 0;
					int sp6 = (p6 == 0 && p7 == 1) ? 1 : 0;
					int sp7 = (p7 == 0 && p8 == 1) ? 1 : 0;
					int sp8 = (p8 == 0 && p9 == 1) ? 1 : 0;
					int sp9 = (p9 == 0 && p2 == 1) ? 1 : 0;
					int sp1 = sp2 + sp3 + sp4 + sp5 + sp6 + sp7 + sp8 + sp9;

					if (np1 >= 2 && np1 <= 6 && sp1 == 1 && ((p2 * p4 * p6) == 0) && ((p4 * p6 * p8) == 0))
					{
						stop = true;
						copymat.at<uchar>(j, i) = 0;
					}
				}
			}
		//step2
		for (int i = 1; i < input.cols - 1; i++)
		{
			for (int j = 1; j < input.rows - 1; j++)
			{
				if (input.at<uchar>(j, i) > 0)
				{
					int p2 = int(input.at<uchar>(j - 1, i)) > 0 ? 1 : 0;
					int p3 = int(input.at<uchar>(j - 1, i + 1)) > 0 ? 1 : 0;
					int p4 = int(input.at<uchar>(j, i + 1)) > 0 ? 1 : 0;
					int p5 = int(input.at<uchar>(j + 1, i + 1)) > 0 ? 1 : 0;
					int p6 = int(input.at<uchar>(j + 1, i)) > 0 ? 1 : 0;
					int p7 = int(input.at<uchar>(j + 1, i - 1)) > 0 ? 1 : 0;
					int p8 = int(input.at<uchar>(j, i - 1)) > 0 ? 1 : 0;
					int p9 = int(input.at<uchar>(j - 1, i - 1)) > 0 ? 1 : 0;
					int np1 = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
					int sp2 = (p2 == 0 && p3 == 1) ? 1 : 0;
					int sp3 = (p3 == 0 && p4 == 1) ? 1 : 0;
					int sp4 = (p4 == 0 && p5 == 1) ? 1 : 0;
					int sp5 = (p5 == 0 && p6 == 1) ? 1 : 0;
					int sp6 = (p6 == 0 && p7 == 1) ? 1 : 0;
					int sp7 = (p7 == 0 && p8 == 1) ? 1 : 0;
					int sp8 = (p8 == 0 && p9 == 1) ? 1 : 0;
					int sp9 = (p9 == 0 && p2 == 1) ? 1 : 0;
					int sp1 = sp2 + sp3 + sp4 + sp5 + sp6 + sp7 + sp8 + sp9;
					if (np1 >= 2 && np1 <= 6 && sp1 == 1 && (p2 * p4 * p8) == 0 && (p2 * p6 * p8) == 0)
					{
						stop = true;
						copymat.at<uchar>(j, i) = 0;
					}
				}
			}
		}
		//将新得到的图片赋给新的图片
		copymat.copyTo(input);
		if (!stop)
		{
			break;
		}
	}
	copymat.copyTo(output);
}

//第i，j个点像素值
double Linelidar::ijpixel(double& x, double& y, Mat& m)
{
	int x_0 = int(x);
	int x_1 = int(x + 1);
	int y_0 = int(y);
	int y_1 = int(y + 1);
	int px_0y_0 = int(m.at<uchar>(y_0, x_0));
	int px_0y_1 = int(m.at<uchar>(y_1, x_0));
	int px_1y_0 = int(m.at<uchar>(y_0, x_1));
	int px_1y_1 = int(m.at<uchar>(y_1, x_1));
	double x_y0 = px_0y_0 + (x - double(x_0)) * (px_1y_0 - px_0y_0);
	double x_y1 = px_0y_1 + (x - double(x_0)) * (px_1y_1 - px_0y_1);
	double x_y = x_y0 + (y - double(y_0)) * (x_y1 - x_y0);
	return x_y;
}

//normal vector 法向量
void Linelidar::CalcNormVec(Point2d ptA, Point2d ptB, Point2d ptC, double pfCosSita, double pfSinSita)
{
	double fVec1_x, fVec1_y, fVec2_x, fVec2_y;

	if (ptA.x == 0 && ptA.y == 0)
	{
		ptA.x = ptC.x;
		ptA.y = ptC.y;
		//先用B点坐标减A点坐标。
		fVec1_x = -(ptB.x - ptA.x);
		fVec1_y = -(ptB.y - ptA.y);
	}
	else
	{
		//先用B点坐标减A点坐标。
		fVec1_x = ptB.x - ptA.x;
		fVec1_y = ptB.y - ptA.y;
	}

	if (ptC.x == 0 && ptC.y == 0)
	{
		ptC.x = ptA.x;
		ptC.y = ptA.y;
		//再用C点坐标减B点坐标。
		fVec2_x = (ptB.x - ptC.x);
		fVec2_y = (ptB.y - ptC.y);
	}
	else
	{
		//再用C点坐标减B点坐标。
		fVec2_x = ptC.x - ptB.x;
		fVec2_y = ptC.y - ptB.y;
	}

	//单位化。
	double fMod = sqrt(fVec1_x * fVec1_x + fVec1_y * fVec1_y);
	fVec1_x /= fMod;
	fVec1_y /= fMod;
	//计算垂线。
	double fPerpendicularVec1_x = -fVec1_y;
	double fPerpendicularVec1_y = fVec1_x;


	//单位化。
	fMod = sqrt(fVec2_x * fVec2_x + fVec2_y * fVec2_y);
	fVec2_x /= fMod;
	fVec2_y /= fMod;
	//计算垂线。
	double fPerpendicularVec2_x = -fVec2_y;
	double fPerpendicularVec2_y = fVec2_x;
	//求和。
	double fSumX = fPerpendicularVec1_x + fPerpendicularVec2_x;
	double fSumY = fPerpendicularVec1_y + fPerpendicularVec2_y;
	//单位化。
	fMod = sqrt(fSumX * fSumX + fSumY * fSumY);
	double fCosSita = fSumX / fMod;
	double fSinSita = fSumY / fMod;
	pfCosSita = fCosSita;
	pfSinSita = fSinSita;
}

void Linelidar::point(Mat& inputimg, vector<Point2d>& pt)
{
	pt.push_back(Point2d(0, 0));
	for (int i = 0; i < inputimg.cols; i++)
		for (int j = 0; j < inputimg.rows; j++)
		{
			if (inputimg.at<uchar>(j, i) >= 200)
			{
				Point2d curr = Point2d(i, j);
				pt.push_back(curr);
			}
		}
	pt.push_back(Point2d(0, 0));
}

void Linelidar::filter_img1(Mat& img1, Mat& img2,Mat& img3)
{
	Mat left_img,mid_img;
    ImageContour mid = imagecontour(img1);
    ImageContour left = imagecontour(img2);
	left_img = left.src;
	mid_img = mid.src;
	imshow("left",left_img);
    imshow("mid",mid_img);
    if(mid.contours.size() == 0)
	{
		img3 = left_img;
		return ;
	}
    Mat sum_img = Mat::zeros(left.src.size(),CV_8UC1);
    Mat sqsum_img = Mat::zeros(left.src.size(),CV_8UC1);
    integral(mid_img,sum_img,sqsum_img);
	normalize(sum_img,left_img,0,255,NORM_MINMAX,CV_8UC1,Mat());
	Mat imgcontour = Mat::zeros(left.src.size(),CV_8UC1);
	for(int j = 0;j < left.contours.size();j++)
	{   
		double min_pro = 1;
		double pro = matchShapes(mid.contours[0],left.contours[j],CV_CONTOURS_MATCH_I3,1.0);
		if(pro > min_pro)
		{  
            drawContours(imgcontour,left.contours,j,Scalar(255,255,255),CV_FILLED,
	 			                                 8,left.hierarchy,0,Point(0,0));
		}                        
	}
	imshow("img3",imgcontour);
	img3 = imgcontour;
}

ImageContour imagecontour(Mat& input)
{   
	ImageContour A;
    int size = 100;
	Mat img;
	input.copyTo(img);
    cvtColor(img, img, COLOR_RGB2GRAY);
	medianBlur(img,img,5);
	GaussianBlur(img,img, Size(3, 3), 0);
	threshold(img, img,100, 255, 3);//THRESH_BINARY
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(img,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point());
	Mat imgcontour = Mat::zeros(img.size(),CV_8UC1);
    for(int i = 0;i < contours.size();i++)
	{  
		int areas = 0;
		areas = fabs(contourArea(contours[i]));
		if(areas > size)
		{
			drawContours(imgcontour,contours,i,Scalar(255,255,255),CV_FILLED,
	 			                                 8,hierarchy,0,Point(0,0));
		}
	}
    vector<vector<Point> > contours1;
	vector<Vec4i> hierarchy1;
	findContours(imgcontour,contours1,hierarchy1,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point());
	A.src = imgcontour;
	A.hierarchy = hierarchy1;
	A.contours = contours1;
	return A;
}

void Linelidar::StegerLine(Mat& img0 ,vector<Point>& points)
{ 
	//ctime_r startime,endtime;
    Mat img;
    cvtColor(img0, img0, CV_BGR2GRAY);
    img = img0.clone();
    //高斯滤波
    img.convertTo(img, CV_32FC1);
    GaussianBlur(img, img, Size(0, 0), 6, 6);
    threshold(img, img,30, 255, 3);
    //一阶偏导数
	//startime = clock();
    Mat m1, m2;
    m1 = (Mat_<float>(1, 2) << 1, -1);  //x偏导
    m2 = (Mat_<float>(2, 1) << 1, -1);  //y偏导

    Mat dx, dy;
    filter2D(img, dx, CV_32FC1, m1);
    filter2D(img, dy, CV_32FC1, m2);

    //二阶偏导数
    Mat m3, m4, m5;
    m3 = (Mat_<float>(1, 3) << 1, -2, 1);   //二阶x偏导
    m4 = (Mat_<float>(3, 1) << 1, -2, 1);   //二阶y偏导
    m5 = (Mat_<float>(2, 2) << 1, -1, -1, 1);   //二阶xy偏导

    Mat dxx, dyy, dxy;
    filter2D(img, dxx, CV_32FC1, m3);
    filter2D(img, dyy, CV_32FC1, m4);
    filter2D(img, dxy, CV_32FC1, m5);
    //hessian矩阵
    double maxD = -1;
    int imgcol = img.cols;// 2D数组中列数
    int imgrow = img.rows; // 2D数组中行数
    vector<double> Pt;

    for (int i=0;i<imgcol;i++)
    {
        for (int j=0;j<imgrow;j++)
        {
            if (img0.at<uchar>(j,i)>200)
            {
                Mat hessian(2, 2, CV_32FC1);//CV_32FC1  float
                hessian.at<float>(0, 0) = dxx.at<float>(j, i);
                hessian.at<float>(0, 1) = dxy.at<float>(j, i);
                hessian.at<float>(1, 0) = dxy.at<float>(j, i);
                hessian.at<float>(1, 1) = dyy.at<float>(j, i);

                Mat eValue; //特征值
                Mat eVectors; //特征向量
                eigen(hessian, eValue, eVectors);

                double nx, ny;
                double fmaxD = 0;
                if (fabs(eValue.at<float>(0,0))>= fabs(eValue.at<float>(1,0)))  //求特征值最大时对应的特征向量
                {
                    nx = eVectors.at<float>(0, 0);
                    ny = eVectors.at<float>(0, 1);
                    fmaxD = eValue.at<float>(0, 0);
                }
                else
                {
                    nx = eVectors.at<float>(1, 0);
                    ny = eVectors.at<float>(1, 1);
                    fmaxD = eValue.at<float>(1, 0);
                }

                double t = -(nx*dx.at<float>(j, i) + ny*dy.at<float>(j, i)) 
				        / (nx*nx*dxx.at<float>(j,i)+2*nx*ny*dxy.at<float>(j,i)+ny*ny*dyy.at<float>(j,i));

                if (fabs(t*nx)<=0.5 && fabs(t*ny)<=0.5)
                {
                    Pt.push_back(i);
                    Pt.push_back(j);
                }
            }
        }
    }
	//灰度图转色彩图
    cvtColor(img0, img0, CV_GRAY2BGR);
	cvtColor(img, img, CV_GRAY2BGR);
	//显示结果

	Mat imgcontour = Mat::zeros(img0.size(),CV_8UC1);
	cvtColor(imgcontour, imgcontour, CV_GRAY2BGR);
	for (int k = 0;k<Pt.size() / 2;k++)
    {
        Point2d p;
        p.x = Pt[2 * k + 0];
        p.y = Pt[2 * k + 1];
		points.push_back(p);
        circle(img0, p, 0.5, Scalar(0, 0, 255));
    }

    //cout<<"point"<<points<<endl;
    // imshow("result", img0);
    // waitKey(0);
}

void Linelidar::grayline(Mat& img,vector<Point>& pts)
{   
	Mat img1 ,img2;
	img.copyTo(img1);
	zhang(img1, img2);
	vector<Point2d> points;
	point(img1, points);
	Point2d pt;
	double sum = 0, sum_sumx = 0, sum_sumy = 0;
	Mat imgcontour = Mat::zeros(img.size(),CV_8UC1);
	cvtColor(imgcontour, imgcontour, CV_GRAY2BGR);
    for (int i = 1; i < points.size() - 1; i++)
	{
		//normal
		double pfCosSita = 0, pfSinSita = 0;
		CalcNormVec(Point2d(points[i - 1].x, points[i - 1].y), 
		                 Point2d(points[i].x, points[i].y), 
						 Point2d(points[i + 1].x, points[i + 1].y), pfCosSita, pfSinSita);
		//-------------------灰度中心法gdd---------------------//
		for (int j = 0; j < 2; j++)
		{
			if (j == 0)
			{
				double cj = points[i].x;
				double ci = points[i].y;
				sum = ijpixel(cj, ci, img1);
				sum_sumx = ijpixel(cj, ci, img2) * cj;
				sum_sumy = ijpixel(cj, ci, img2) * ci;
			}
			else
			{
				double x_cor = points[i].x + j * pfCosSita;
				double y_cor = points[i].y + j * pfSinSita;
				double x_cor1 = points[i].x - j * pfCosSita;
				double y_cor1 = points[i].y - j * pfSinSita;
				sum = sum + ijpixel(x_cor, y_cor, img1) + ijpixel(x_cor1, y_cor1, img1);
				sum_sumx = sum_sumx + ijpixel(x_cor, y_cor, img1) * x_cor + ijpixel(x_cor1, y_cor1, img1) * x_cor1;
				sum_sumy = sum_sumy + ijpixel(x_cor, y_cor, img1) * y_cor + ijpixel(x_cor1, y_cor1, img1) * y_cor1;
			}
		}
		//图像中心线画出来
        pt.x = sum_sumx / sum;
		pt.y = sum_sumy / sum;
		pts.push_back(pt);
		circle(imgcontour, pt, 0.5, Scalar(0, 0, 255));
	}
    // cout<<"pose"<<pts<<endl;
	imshow("imgcontour",imgcontour);
}

void Linelidar::leastsquaresfit(vector<float> &x,vector<float> &y)
{
	ModelCoefficients  test = LeastSquaresFit(x,y);
	int n = test.values.size();
    vector<double> yyy;
	double yy ;
	for(int i = 0;i<x.size();i++)
	{
		for(int j = 0;j < n+1;j++)
	    {
		   double Y = test.values[j] * pow(x[i],j);
		   yy += Y;
	    }
		yyy.push_back(yy);
		cout<<"测试x = "<<x[i]<<"  "<<"  预测值为: "<<yyy[i]<<"   观测值为:"<<y[i]<<endl;
        yy = 0;
	}
	for(int i = 0;i < test.values.size();i++)
	{
       cout<<"values: "<<test.values[i]<<endl;
	}

    ////////////////////////////////////////////////
	double xx = 100.0;                             //
	double sum;                                   //
	for(int j = 0;j < n+1;j++)                    //
	{                                             //
		double Y = test.values[j] * pow(xx,j);    //
		sum += Y;                                 //
	}                                             //
    cout<<"拟合结果"<<sum<<endl;                   //3.0
    ////////////////////////////////////////////////
}

 void Linelidar::fitcicle(vector<Point2f> points,Point2f& circleCenter,float& radius)
 {
	 double Xi1Sum = 0;
	 double Xi2Sum = 0;
	 double Xi3Sum = 0;
	 double Yi1Sum = 0;
	 double Yi2Sum = 0;
	 double Yi3Sum = 0;
	 double XiYi1Sum = 0;
	 double Xi2YiSum = 0;
	 double XiYi2Sum = 0;
	 double WiSum = 0;

	 for(size_t i = 0;i<points.size();i++)
	 {
		Xi1Sum += points.at(i).x;
	    Xi2Sum += points.at(i).x * points.at(i).x;
	    Xi3Sum += points.at(i).x * points.at(i).x * points.at(i).x;
	    Yi1Sum += points.at(i).y;
	    Yi2Sum += points.at(i).y * points.at(i).y;
	    Yi3Sum += points.at(i).y * points.at(i).y * points.at(i).y;
	    XiYi1Sum += points.at(i).x * points.at(i).y;
	    Xi2YiSum += points.at(i).x * points.at(i).x * points.at(i).y;
	    XiYi2Sum += points.at(i).x * points.at(i).y * points.at(i).y;
	    WiSum += 1;
	 }

	 int N = 3;
	 Mat A = Mat::zeros(N,N,CV_64FC1);
	 Mat B = Mat::zeros(N,1,CV_64FC1);

	 A.at<double>(0,0) = Xi2Sum;
	 A.at<double>(0,1) = XiYi1Sum;
	 A.at<double>(0,2) = Xi1Sum;

	 A.at<double>(1,0) = XiYi1Sum;
	 A.at<double>(1,1) = Yi2Sum;
	 A.at<double>(1,2) = Yi1Sum;

	 A.at<double>(2,0) = Xi1Sum;
	 A.at<double>(2,1) = Yi1Sum;
	 A.at<double>(2,2) = WiSum;

	 B.at<double>(0,0) = -(Xi3Sum + XiYi2Sum);
	 B.at<double>(1,0) = -(Xi2YiSum + Yi3Sum);
	 B.at<double>(2,0) = -(Xi2Sum + Yi2Sum);

	 Mat X = Mat::zeros(N,1,CV_64FC1);
	 solve(A,B,X,DECOMP_LU);
	 double D = X.at<double>(0,0);
	 double E = X.at<double>(1,0);
	 double F = X.at<double>(2,0);
   
     circleCenter.x = -D / 2;
	 circleCenter.y = -E / 2;
	 radius = sqrt(D * D + E * E - 4 * F) / 2;
 }
 
void Linelidar::GetSample(int& indexe_size,vector<int>& index)
 {

     random_device rd; //随机数发生器
     mt19937 mt(rd()); //随机数引擎
	 uniform_int_distribution<int> dist(0,indexe_size - 1); 
	 // 随机采样三个不相同的点
	 index = {dist(mt),dist(mt),dist(mt)};
	 if(index[0] != index[1] && index[0] != index[2] && index[1] != index[2])
	 {
		sort(index.begin(),index.end());
	 }
   
 }

void Linelidar::RanSanCirfit(vector<Point2f> points,Point2f& circleCenter,float& radius)
{
     if(points.size() <= 3) {return;}
	 int iterate_num = 500;
	 float sample_point_min_distance = 5.0;
	 
	 //输入点数
	 vector<int> sampled_indexes;
	 vector<int> index;
	 vector<int> index2;
	 sampled_indexes.reserve(iterate_num);
	 //存储线上点
	 vector<int> inlier(points.size(),0);
	 vector<int> inlier_temp(points.size(),0);
	 int max_inliner_num = 0;
	 int sample_count = 0;
     int point_size = points.size();
	 while (sample_count < iterate_num)
	 {   
		 GetSample(point_size,index);
		 int p1 = index[0];
		 int p2 = index[1];
		 int p3 = index[2];
		 
		 if(abs(points[p1].x - points[p2].x < sample_point_min_distance)
		    && abs(points[p1].y - points[p2].y < sample_point_min_distance)
			&& abs(points[p2].x - points[p3].x < sample_point_min_distance)
			&& abs(points[p2].y - points[p3].y < sample_point_min_distance))
		 {
			continue;
		 }
		 else
		 {
			 sampled_indexes.push_back(p1);
			 sampled_indexes.push_back(p2);
			 sampled_indexes.push_back(p3);
		 }

        float r;
	    Point2f ct;
	    fitcicle({points[p1],points[p2],points[p3]},ct,r);
	    int inlier_num = 0;
		vector<Point2f> inliers;
	    for(int i = 0;i < points.size();i++)
	    {
           Point2f p = points[i];
           inlier_temp[i] = 0;
		   double p_2_center = sqrt(pow(p.x - ct.x,2) + pow(p.y - ct.y,2));
		   if(p_2_center - r < 5)
		   {
			  inlier_temp[i] = 1;
			  inlier_num++;
			  inliers.push_back(p);
		   }
	    } 
        
		// cout<<"inlier_num"<<inlier_num<<endl;
		// cout<<"max_inliner_num"<<max_inliner_num<<endl;
	    if(inlier_num > max_inliner_num)
	    {
		    max_inliner_num = inlier_num;
		    inlier = inlier_temp;
	    }
   
	    else
	    {  
		   double epsilon = 1.0 - double(inlier_num) / (double)points.size();
		   double p = 0.99;
		   double s = 3.0;
		   iterate_num = int(log(1.0 - p) / log(1.0 - pow((1 - epsilon),s)));
	    }
	    sample_count++;
    } 

    vector<Point2f> best_inliers;
	best_inliers.reserve(max_inliner_num);
	for(int i = 0;i < max_inliner_num;i++)
	{
		if(inlier_temp[i] == 0)
		{
			best_inliers.push_back(points[i]);
		}
	}

	fitcicle(best_inliers,circleCenter,radius);
	cout<<"circleCenter_x = "<<" "<< circleCenter.x<<" "<<"circleCenter_y = "<<" "<< circleCenter.y<<endl;
	cout<<" radius = "<<" "<<radius <<endl;
}

void Linelidar::choseline(Mat& src)
{
	 Mat img;
	 src.copyTo(img);
	 vector<Point2f> pp;
	 vector<vector<Point2f> > contours;
	 vector<Vec4i> hierarchy;
	 findContours(img,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point());
	 int contours_num = contours.size();
	
	 const unsigned int sz = contours[0].size();
	//
	 for (int i = 0;i < contours_num;i++)
	 { 
       if(contours[i].size() > contours[i + 1].size())
	   {
		    pp = contours[i];
	   } 
	   else
	   {
		    pp = contours[i + 1];
	   }
	 }
     Point2f circleCenter;
	 float radius;
	 RanSanCirfit(pp,circleCenter,radius);
	 cout<<"circleCenter_x:  "<<circleCenter.x<<" "<<"circleCenter_y:  "<<circleCenter.y<<endl;
	 cout<<"radius:  "<<radius<<endl;

}

void Linelidar::Interpolation(vector<float> &x,vector<float> &y,float xx)
{
     vector<float> px,lx;
	 float m = 1;
	 float n = 1;
	 float yy = 0;
	 for(int i = 0;i < x.size();i++)
	 {
        for(int j = 0;j < x.size();j++)
		{
            if(j==i)
			{
			    continue;
			}
			else
			{
                float xx1 = xx - x[j];
		        m = m * xx1;
				float xx2 = x[i] - x[j];
				n = n * xx2;
			}
		}
        px.push_back(m/n);
		n = 1;
		m = 1;
	 }
	 for(int i = 0;i < x.size();i++)
	 {
        float n = y[i]*px[i];
		yy += n;
		lx.push_back(yy);
	 }
	//  float a = 1.25*pow(35.26,2) + 8.76*35.26+ 1.42;
	//  cout<<"a= "<<a<<endl;
	 cout<<"插值结果"<<yy<<endl;
}