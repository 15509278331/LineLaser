#include "line.h"

int main()
{
	Linelidar linelidar;
	// 读取图像数据
	vector<String> mid_img,left_img,right_img;
	vector<Mat> left_result,right_result; 
    String mid_img_path =  "/home/zhangwei/mid_img/";
    String left_img_path =  "/home/zhangwei/left_img/";
    String right_imgpath =  "/home/zhangwei/right_img/";
	glob(mid_img_path,mid_img,false);
	glob(left_img_path,left_img,false);
	glob(right_imgpath,right_img,false);
	// 判断图像是否为空
    if(mid_img.size() == 0 || left_img.size() == 0 || right_img.size() == 0)
	{   
		cout<<"Image loading error "<<endl;
		return 0;
	}
    
	if(mid_img.size() != left_img.size() || mid_img.size() != right_img.size() || left_img.size() != right_img.size())
	{
		cout<<"Image loading error "<<endl;
		return 0;
	}
    // 存储处理后的图像数据
	for(int i = 0;i < mid_img.size();i++)
	{ 
	   Mat mid_image,left_image,right_image,A,B;
	   mid_image = imread(mid_img[i]);
	   left_image = imread(left_img[i]);
	   right_image = imread(right_img[i]);
	   linelidar.filter_img1(mid_image,left_image,A);
	   linelidar.filter_img1(mid_image,right_image,B);
       left_result.push_back(A);
	   right_result.push_back(B);
	} 
    vector<vector<Point> > left_ptt;
	vector<vector<Point> > right_ptt;
	left_ptt.resize(left_result.size());
	right_ptt.resize(right_result.size());
	// 中心线提取
	Mat l_result,r_result,m_result;
	VideoWriter r_video,l_video,result_video;
	Size size = left_result[0].size();
	r_video.open("r_video.avi",CV_FOURCC('M','J','P','G'),25,size,true);
	l_video.open("l_video.avi",CV_FOURCC('M','J','P','G'),25,size,true);
	result_video.open("result_video.avi",CV_FOURCC('M','J','P','G'),25,size,true);
	
	for(int i = 0;i < left_result.size();i++)
	{ 
	   linelidar.grayline(left_result[i],l_result,left_ptt[i]);
	   linelidar.grayline(right_result[i],r_result,right_ptt[i]);
	//    r_video.write(r_result);
	//    l_video.write(l_result);
	//    waitKey(100);
	} 

	for(int i = 0;i < right_ptt.size();i++)
	{ 
	   linelidar.result_line(left_ptt[i],right_ptt[i],m_result,size);
	   result_video.write(m_result);
	   waitKey(500);
	}   
	return 0;
}

