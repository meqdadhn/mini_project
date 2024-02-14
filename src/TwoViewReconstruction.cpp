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
TwoViewReconstruction::TwoViewReconstruction(Eigen::Matrix3f& K) {

    K_ = K;

    Reset();

}

// destructor
TwoViewReconstruction::~TwoViewReconstruction() {

}


void TwoViewReconstruction::Reset() {

    // clear memeber variables
    keypoints1_.clear();
    keypoints2_.clear();
    matches12_.clear();
    match_inliers_.clear();

    mask_.release();
}


// processimage
bool TwoViewReconstruction::ProcessImage(cv::Mat& img1, cv::Mat& img2) {
    // Convert images to grayscale
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

    // Detect ORB keypoints and descriptors
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create();
    cv::Mat descriptors1, descriptors2;

    orb->detectAndCompute(gray1, cv::noArray(), keypoints1_, descriptors1);
    orb->detectAndCompute(gray2, cv::noArray(), keypoints2_, descriptors2);


    // Match features using BFMatcher
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);


    // Find fundamental matrix and filter matches
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(keypoints1_[match.queryIdx].pt);
        points2.push_back(keypoints2_[match.trainIdx].pt);
        
        matches12_.push_back(std::make_pair(match.queryIdx, match.trainIdx));
    }

    if (points1.size() < 20) {
        std::cerr << "Error: too few initial matches." << std::endl;
        return false;
    }


    cv::Mat fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3.0, 0.99, mask_);
    cv::cv2eigen(fundamental_matrix, F21_);
    // Filter inliers and outliers
    std::vector<cv::DMatch> inliers, outliers;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (mask_.at<char>(i,0)) {
            inliers.push_back(matches[i]);
            match_inliers_.push_back(true);
        } else {
            outliers.push_back(matches[i]);
            match_inliers_.push_back(false);
        }
    }

    // Draw all matches
    cv::Mat img_matches_all;
    cv::drawMatches(img1, keypoints1_, img2, keypoints2_, matches, img_matches_all);
    cv::putText(img_matches_all, "All Matches: " + std::to_string(matches.size()), cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::imwrite("out/All.png", img_matches_all);

    // Draw inliers
    cv::Mat img_matches_inliers;
    cv::drawMatches(img1, keypoints1_, img2, keypoints2_, inliers, img_matches_inliers);
    cv::putText(img_matches_inliers, "inliers: " + std::to_string(inliers.size()), cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);    
    cv::imwrite("out/Inliers.png", img_matches_inliers);

    // Draw outliers
    cv::Mat img_matches_outliers;
    cv::drawMatches(img1, keypoints1_, img2, keypoints2_, outliers, img_matches_outliers);
    cv::putText(img_matches_outliers, "outliers: " + std::to_string(outliers.size()), cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);       
    cv::imwrite("out/Outliers.png", img_matches_outliers);
    

    if (inliers.size() < 20){
        std::cerr << "Error: too few inliers matches." << std::endl;
        return false;
    } 
    else
        return true;
  
    
}

bool TwoViewReconstruction::Reconstruct(std::vector<cv::Point3f>& vP3D,  std::vector<bool>& vbTriangulated,
                                        Eigen::Matrix3f& R, Eigen::Vector3f& t) {

    // Compute Essential Matrix from Fundamental Matrix
    Eigen::Matrix3f E21 = K_.transpose() * F21_ * K_;

    Eigen::Matrix3f R1, R2;
    Eigen::Vector3f t0;

    // Recover the 4 motion hypotheses
    DecomposeE(E21,R1,R2,t0);

    Eigen::Vector3f t1 = t0;
    Eigen::Vector3f t2 = -t0;

    // Reconstruct with the 4 hyphoteses and check
    std::vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    std::vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    int nGood1 = CheckRT(R1, t1, vP3D1, 100.0, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2, t1, vP3D2, 100.0, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1, t2, vP3D3, 100.0, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2, t2, vP3D4, 100.0, vbTriangulated4, parallax4);

    int maxGood = std::max(nGood1,std::max(nGood2,std::max(nGood3,nGood4)));

    // If there is not enough triangulated points reject initialization
    if(maxGood<20) {
        std::cerr << "Error: too few inliers to recover pose." << std::endl;
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1)
    {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R = R1.transpose();
            t = - R1.transpose() * t1;
            return true;
    }else if(maxGood==nGood2)
    {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R = R2.transpose();
            t = -R2.transpose()* t1;
            return true;
    }else if(maxGood==nGood3)
    {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R = R1.transpose();
            t = -R1.transpose() * t2;
            return true;
    }else if(maxGood==nGood4)
    {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R = R2.transpose();
            t = -R2.transpose() * t2;
            return true;
    }

    return false;


}

void TwoViewReconstruction::DecomposeE(const Eigen::Matrix3f& E,Eigen::Matrix3f& R1,
                                        Eigen::Matrix3f& R2,Eigen::Vector3f& t) {
    
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f Vt = svd.matrixV().transpose();

    t = U.col(2);
    t = t / t.norm();

    Eigen::Matrix3f W;
    W.setZero();
    W(0,1) = -1;
    W(1,0) = 1;
    W(2,2) = 1;

    R1 = U * W * Vt;
    if(R1.determinant() < 0)
        R1 = -R1;

    R2 = U * W.transpose() * Vt;
    if(R2.determinant() < 0)
        R2 = -R2;
}

int TwoViewReconstruction::CheckRT(const Eigen::Matrix3f &R, const Eigen::Vector3f &t,
                                    std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, 
                                    float &parallax) {


    // Calibration parameters
    const float fx = K_(0,0);
    const float fy = K_(1,1);
    const float cx = K_(0,2);
    const float cy = K_(1,2);

    vbGood = std::vector<bool>(keypoints1_.size(),false);
    vP3D.resize(keypoints1_.size());

    std::vector<float> vCosParallax;
    vCosParallax.reserve(keypoints1_.size());

    // Camera 1 Projection Matrix K[I|0]
    Eigen::Matrix<float,3,4> P1;
    P1.setZero();
    P1.block<3,3>(0,0) = K_;

    Eigen::Vector3f O1;
    O1.setZero();

    // Camera 2 Projection Matrix K[R|t]
    Eigen::Matrix<float,3,4> P2;
    P2.block<3,3>(0,0) = R;
    P2.block<3,1>(0,3) = t;
    P2 = K_ * P2;

    Eigen::Vector3f O2 = -R.transpose() * t;

    int nGood=0;

    for(size_t i=0, iend=matches12_.size();i<iend;i++)
    {
        if(!match_inliers_[i])
            continue;

        const cv::KeyPoint &kp1 = keypoints1_[matches12_[i].first];
        const cv::KeyPoint &kp2 = keypoints1_[matches12_[i].second];

        Eigen::Vector3f p3dC1;
        Eigen::Vector3f x_p1(kp1.pt.x, kp1.pt.y, 1);
        Eigen::Vector3f x_p2(kp2.pt.x, kp2.pt.y, 1);

        Triangulate(x_p1, x_p2, P1, P2, p3dC1);


        if(!isfinite(p3dC1(0)) || !isfinite(p3dC1(1)) || !isfinite(p3dC1(2)))
        {
            vbGood[matches12_[i].first]=false;
            continue;
        }

        // Check parallax
        Eigen::Vector3f normal1 = p3dC1 - O1;
        float dist1 = normal1.norm();

        Eigen::Vector3f normal2 = p3dC1 - O2;
        float dist2 = normal2.norm();

        float cosParallax = normal1.dot(normal2) / (dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(p3dC1(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        Eigen::Vector3f p3dC2 = R * p3dC1 + t;

        if(p3dC2(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1(2);
        im1x = fx*p3dC1(0)*invZ1+cx;
        im1y = fy*p3dC1(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);
        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2(2);
        im2x = fx*p3dC2(0)*invZ2+cx;
        im2y = fy*p3dC2(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);
        if(squareError2>th2)
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[matches12_[i].first] = cv::Point3f(p3dC1(0), p3dC1(1), p3dC1(2));
        nGood++;

        if(cosParallax<0.99998)
            vbGood[matches12_[i].first]=true;
    }

    if(nGood>0)
    {
        std::sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = std::min(50,int(vCosParallax.size()-1));
        parallax = std::acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;

}


bool TwoViewReconstruction::Triangulate(Eigen::Vector3f &x_c1, Eigen::Vector3f &x_c2,Eigen::Matrix<float,3,4> &Tc1w ,
                                        Eigen::Matrix<float,3,4> &Tc2w , Eigen::Vector3f &x3D)
{
    Eigen::Matrix4f A;
    A.block<1,4>(0,0) = x_c1(0) * Tc1w.block<1,4>(2,0) - Tc1w.block<1,4>(0,0);
    A.block<1,4>(1,0) = x_c1(1) * Tc1w.block<1,4>(2,0) - Tc1w.block<1,4>(1,0);
    A.block<1,4>(2,0) = x_c2(0) * Tc2w.block<1,4>(2,0) - Tc2w.block<1,4>(0,0);
    A.block<1,4>(3,0) = x_c2(1) * Tc2w.block<1,4>(2,0) - Tc2w.block<1,4>(1,0);

    Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);

    Eigen::Vector4f x3Dh = svd.matrixV().col(3);

    if(x3Dh(3)==0)
        return false;

    // Euclidean coordinates
    x3D = x3Dh.head(3)/x3Dh(3);

    return true;
}
