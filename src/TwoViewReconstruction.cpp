/*///////////////////////////////////////////////////////////////////////////////////////
//
//  This is the implementation for TwoViewReconstruction class
//  author: Meqdad Hashemin, s.meqdad.hn@gmail.com
//  TwoViewReconstruction.cpp
//  Jan 2024
//
/*///////////////////////////////////////////////////////////////////////////////////////


#include "TwoViewReconstruction.h"

// constructor
TwoViewReconstruction::TwoViewReconstruction() {

}

// destructor
TwoViewReconstruction::~TwoViewReconstruction() {

}


// processimage
void TwoViewReconstruction::processImage(cv::Mat img1, cv::Mat img2) {
    // Convert images to grayscale
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

    // Detect ORB keypoints and descriptors
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    orb->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);

    // Match features using BFMatcher
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Find fundamental matrix and filter matches
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    cv::Mat mask;
    cv::Mat fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3.0, 0.99, mask);

    // Filter inliers and outliers
    std::vector<cv::DMatch> inliers, outliers;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (mask.at<char>(i,0)) {
            inliers.push_back(matches[i]);
        } else {
            outliers.push_back(matches[i]);
        }
    }

    // Draw all matches
    cv::Mat img_matches_all;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches_all);
    cv::putText(img_matches_all, "All Matches: " + std::to_string(matches.size()), cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::imshow("All Matches", img_matches_all);

    // Draw inliers
    cv::Mat img_matches_inliers;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, inliers, img_matches_inliers);
    cv::putText(img_matches_inliers, "inliers: " + std::to_string(inliers.size()), cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);    
    cv::imshow("Inliers", img_matches_inliers);

    // Draw outliers
    cv::Mat img_matches_outliers;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, outliers, img_matches_outliers);
    cv::putText(img_matches_outliers, "outliers: " + std::to_string(outliers.size()), cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);       
    cv::imshow("Outliers", img_matches_outliers);
    
    cv::waitKey(0);    
}