/*///////////////////////////////////////////////////////////////////////////////////////
//
//  This is the header file for TwoViewReconstruction class
//  author: Meqdad Hashemin, s.meqdad.hn@gmail.com
//  TwoViewReconstruction.h
//  Jan 2024
//
/*///////////////////////////////////////////////////////////////////////////////////////

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

class TwoViewReconstruction {


public: 

    // constructor
    TwoViewReconstruction();
    
    // destructor
    ~TwoViewReconstruction();


    // processimage
    void processImage(cv::Mat img1, cv::Mat img2);

};