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


    // // Read two images
    cv::Mat img1 = cv::imread("sample/omd/000000.png");
    cv::Mat img2 = cv::imread("sample/omd/000001.png");

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    } 

    // calibration matrix 
    float fx = 618.358;
    float fy = 618.592;
    float cx = 328.986;
    float cy = 237.750;
    Eigen::Matrix3f K;
    K << fx, 0.f, cx, 
         0.f, fy, cy, 
         0.f, 0.f, 1.f;

    // allocate memoory for the stereo images
    TwoViewReconstruction *stereo = new TwoViewReconstruction(K);
    bool status = stereo->ProcessImage(img1, img2);

    std::vector<cv::Point3f> vP3D;
    std::vector<bool> vbTriangulated;
    Eigen::Matrix3f R; 
    Eigen::Vector3f t;

    if (status) {
        stereo->Reconstruct(vP3D, vbTriangulated, R, t);
    }

    // output camera poses
    std::ofstream f1,f2;
    std::string name1, name2;
    name1 = "out/poses.txt";
    name2 = "out/point.txt";

    f1.open(name1);
    f1 << "ImgID\tTx\tTy\tTz" << std::endl;
    f1 << "0\t0\t0\t0" << std::endl;
    f1 << "1\t" << t.x() << "\t" << t.y() << "\t" << t.z() << std::endl;
    f1.close();


    f2.open(name2);
    f2 << "PointID\tX\tY\tZ" << std::endl;
    for (size_t i = 0; i < vP3D.size(); i++) {
        if (vbTriangulated[i]) {
            f2 << i << "\t" << vP3D[i].x << "\t" << vP3D[i].y << "\t" << vP3D[i].z << std::endl;
        }
    }
    f2.close();

    // free memory
    delete stereo;

    std::cout << "Please find the results under out/" << std::endl;

    return 0;
}
