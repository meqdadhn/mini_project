#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

int main() {
    cv::Mat img1 = cv::imread("sample/000000.png");
    cv::Mat img2 = cv::imread("sample/000009.png");
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

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
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Draw matches
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);

    cv::imshow("Matches", img_matches);
    cv::waitKey(0);

    std::cout << "Finished Successfully!! " << std::endl;
    return 0;
}
