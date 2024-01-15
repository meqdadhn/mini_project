/*///////////////////////////////////////////////////////////////////////////////////////
//
//  This is the main function for TwoViewReconstruction
//  author: Meqdad Hashemin, s.meqdad.hn@gmail.com
//  main.cpp
//  Jan 2024
//
/*///////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TwoViewReconstruction.h"

int main() {


    // Read two images
    cv::Mat img1 = cv::imread("sample/omd/000000.png");
    cv::Mat img2 = cv::imread("sample/omd/000001.png");
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    } 

    // allocate memoory for the stereo images
    TwoViewReconstruction *stereo = new TwoViewReconstruction;
    stereo->processImage(img1, img2);

    // free memory
    delete stereo;

    return 0;
}
