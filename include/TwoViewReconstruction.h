/*///////////////////////////////////////////////////////////////////////////////////////
//
//  This is the header file for TwoViewReconstruction class
//  author: Meqdad Hashemin, s.meqdad.hn@gmail.com
//  TwoViewReconstruction.h
//  Jan 2024
//
/*///////////////////////////////////////////////////////////////////////////////////////


#ifndef __TWOVIEW_RECONSTRUCT__
#define __TWOVIEW_RECONSTRUCT__

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <fstream>

class TwoViewReconstruction {


public: 

    // constructor
    TwoViewReconstruction(Eigen::Matrix3f& K);
    
    // destructor
    ~TwoViewReconstruction();

    // reset params
    void Reset();

    // processimage
    bool ProcessImage(cv::Mat& img1, cv::Mat& img2);

    // reconstruct two views
    bool Reconstruct(std::vector<cv::Point3f>& vP3D,  std::vector<bool>& vbTriangulated,
                    Eigen::Matrix3f& R, Eigen::Vector3f& t);

private:

    // decompose Essential matrix to the four possible solutions
    void DecomposeE(const Eigen::Matrix3f& E,Eigen::Matrix3f& R1,Eigen::Matrix3f& R2,Eigen::Vector3f& t);


    // check the reconvered R,t between two images
    int CheckRT(const Eigen::Matrix3f &R, const Eigen::Vector3f &t,
                std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, float &parallax);

    // triangulate two points 
    bool Triangulate(Eigen::Vector3f &x_c1, Eigen::Vector3f &x_c2,Eigen::Matrix<float,3,4> &Tc1w ,
                                            Eigen::Matrix<float,3,4> &Tc2w , Eigen::Vector3f &x3D);
    std::vector<cv::KeyPoint> keypoints1_, keypoints2_;
    cv::Mat mask_;
    std::vector<std::pair<int,int>> matches12_;
    std::vector<bool> match_inliers_;
    
    // Calibration
    Eigen::Matrix3f K_, F21_;

};

#endif