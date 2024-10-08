
#include <../eigen3/Eigen/Core>
#include <../eigen3/Eigen/Dense>
#include <cmath>
#include <iostream>
#include <fstream>

Eigen::Matrix3d rpy2R(const Eigen::Vector3d& rpy)
{
  double r = rpy.x();
  double p = rpy.y();
  double y = rpy.z();

  Eigen::Matrix3d R_z;
  R_z  << std::cos(y), -std::sin(y), 0,
          std::sin(y),  std::cos(y), 0,
          0          ,  0          , 1;

  Eigen::Matrix3d R_y;
  R_y  <<  std::cos(p), 0, std::sin(p),
           0          , 1, 0          ,
          -std::sin(p), 0, std::cos(p);

  Eigen::Matrix3d R_x;
  R_x  << 1, 0          ,  0          ,
          0, std::cos(r), -std::sin(r),
          0, std::sin(r),  std::cos(r);
  
  return R_z * R_y * R_x;
};

Eigen::Vector3d R2rpy(const Eigen::Matrix3d& R)
{
  double r = std::atan2(R(2,1), R(2,2));
  double p = std::asin(-R(2,0));
  double y = std::atan2(R(1,0), R(0,0));

  return Eigen::Vector3d(r,p,y);
}

int PCA(const std::vector<Eigen::Vector3d>& point_list, Eigen::Vector3d& plane_dir)
{
  int status = 0;
  
  // from vecotr to matrix
  Eigen::MatrixXd A(point_list.size(), 3);

  for (size_t i = 0; i < point_list.size(); i++)
  {
    A(i, 0) = point_list[i].x();
    A(i, 1) = point_list[i].y();
    A(i, 2) = point_list[i].z();
  }
  
  // Get the centroid point
  Eigen::Vector3d centroid = A.colwise().mean();

  // shift the points to the origin
  Eigen::MatrixXd A_shifted = A.rowwise() - centroid.transpose();

  // covariance matrix 3 by 3
  Eigen::MatrixXd cov = (A_shifted.transpose() * A_shifted) / 
                    static_cast<float>(point_list.size()- 1);

  // solve for eigenvectors and eigenvalues
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);

  std::cout << "centroid: " << centroid.transpose() << std::endl;
  std::cout << "------------------------------" << std::endl;
  std::cout << "Covariance mat: \n" << cov << std::endl; 
  std::cout << "------------------------------" << std::endl;
  std::cout << "eigen vecs: \n" << solver.eigenvectors().col(0) << std::endl; 
  std::cout << "------------------------------" << std::endl;
  std::cout << "eigen vals: \n" << solver.eigenvalues().transpose() << std::endl;
  std::cout << "------------------------------" << std::endl;

  // Determine the type of feature based on the eigenvalues
  double sum_eigenvalues = solver.eigenvalues().sum();
  double ratio1 = solver.eigenvalues()(2) / sum_eigenvalues;  // largest eigenvalue ratio
  double ratio2 = solver.eigenvalues()(1) / sum_eigenvalues;  // second largest eigenvalue ratio

  std::cout << "Feature classification: ";
  if (ratio1 > 0.9) 
  {
    status = 1;
    std::cout << "Linear feature (points are aligned along a line)" << std::endl;
  } 
  else if (ratio1 + ratio2 > 0.9)
  {
    status = 2;
    plane_dir = solver.eigenvectors().col(0);
    std::cout << "Planar feature (points lie on a plane)" << std::endl;
  } 
  else 
  {
    std::cout << "Rough 3D feature (points are scattered in 3D space)" << std::endl;
  }
  return status;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> FitLine(std::vector<Eigen::Vector3d> point_list)
{
  Eigen::Vector3d p0;
  Eigen::Vector3d p1;
 
  return std::make_pair(p0, p1);
}

Eigen::Vector4d FitPlane(std::vector<Eigen::Vector3d> point_list)
{
  Eigen::Vector4d param;

  Eigen::MatrixXd A(point_list.size(), 4);

  for (size_t i = 0; i < point_list.size(); i++)
  {
    A(i, 0) = point_list[i].x();  
    A(i, 1) = point_list[i].y();
    A(i, 2) = point_list[i].z();
    A(i, 3) = 1;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);

  std::cout << "singular values\n" << svd.singularValues() << std::endl;
  std::cout << "--------------------------------------" << std::endl;
  std::cout << "eigen vectors:\n " << svd.matrixV() << std::endl;
  
  param = svd.matrixV().col(3);
  param /= param.head<3>().norm();
  
  double a = param(0);
  double b = param(1);
  double c = param(2);
  double d = param(3);
  double mean_err = 0.0;
  for (size_t i = 0; i < point_list.size(); i++)
  {
    double res = a * point_list[i].x() + b * point_list[i].y() + c * point_list[i].z() + d; 
    mean_err += res*res;
  }
  double rmse = std::sqrt(mean_err/static_cast<float>(point_list.size()));

  std::cout << "Plane Fitting RMSE: " << rmse << std::endl;
  return  param;
}

Eigen::Vector4d FitPlaneLSA(std::vector<Eigen::Vector3d> point_list,
                            const Eigen::Vector3d& plane_dir)
{
  Eigen::Vector4d param;

  Eigen::MatrixXd A(point_list.size(), 3);
  Eigen::VectorXd Y(point_list.size());
  
  double main_dir;
  main_dir = std::max(std::max(std::fabs(plane_dir.x()), std::fabs(plane_dir.y())),
                      std::fabs(plane_dir.z()));
  
  for (size_t i = 0; i < point_list.size(); i++)
  {
    if (main_dir == std::fabs(plane_dir.z()))
    {
      A(i, 0) = point_list[i].x();  
      A(i, 1) = point_list[i].y();
      A(i, 2) = 1;
      Y(i) = -point_list[i].z();
    }
    if (main_dir == std::fabs(plane_dir.y()))
    {
      A(i, 0) = point_list[i].x();  
      A(i, 1) = point_list[i].z();
      A(i, 2) = 1;
      Y(i) = -point_list[i].y();
    }
    if (main_dir == std::fabs(plane_dir.x()))
    {
      A(i, 0) = point_list[i].y();  
      A(i, 1) = point_list[i].z();
      A(i, 2) = 1;
      Y(i) = -point_list[i].x();
    }
  }

  Eigen::Vector3d x = (A.transpose() * A).ldlt().solve(A.transpose() * Y);

  if (main_dir == std::fabs(plane_dir.z()))
  {
    param = Eigen::Vector4d(x(0), x(1), 1, x(2));
  }
  if (main_dir == std::fabs(plane_dir.y()))
  {
    param = Eigen::Vector4d(x(0), 1, x(1), x(2));
  }
  if (main_dir == std::fabs(plane_dir.x()))
  {
    param = Eigen::Vector4d(1, x(0), x(1), x(2));
  }
  param /= param.head<3>().norm();
  double a = param(0);
  double b = param(1);
  double c = param(2);
  double d = param(3);
  double mean_err = 0.0;
  for (size_t i = 0; i < point_list.size(); i++)
  {
    double res = a * point_list[i].x() + b * point_list[i].y() + c * point_list[i].z() + d; 
    mean_err += res*res;
  }
  double rmse = std::sqrt(mean_err/static_cast<float>(point_list.size()));

  std::cout << "LSA Plane Fitting RMSE: " << rmse << std::endl;
  return  param;
}

void EstimateTransformation(const std::vector<Eigen::Vector3d>& point_list1, 
                            const std::vector<Eigen::Vector3d>& point_list2,
                            Eigen::Matrix3d& R, Eigen::Vector3d& t)
{
   // from vecotr to matrix
  Eigen::MatrixXd Q(point_list1.size(), 3); // desitnation 
  Eigen::MatrixXd P(point_list1.size(), 3); // source

  // 
  for (size_t i = 0; i < point_list1.size(); i++)
  {
    Q(i, 0) = point_list1[i].x();
    Q(i, 1) = point_list1[i].y();
    Q(i, 2) = point_list1[i].z();

    P(i, 0) = point_list2[i].x();
    P(i, 1) = point_list2[i].y();
    P(i, 2) = point_list2[i].z();
  }
  
  
  // Get the centroid point
  Eigen::Vector3d centroid_Q = Q.colwise().mean();
  Eigen::Vector3d centroid_P = P.colwise().mean();

  // shift the points to the origin
  Eigen::MatrixXd Q_shifted = Q.rowwise() - centroid_Q.transpose(); 
  Eigen::MatrixXd P_shifted = P.rowwise() - centroid_P.transpose();

  std::string f_name = "src_shift.txt";
  std::ofstream fout(f_name);
  if (fout.is_open())
  {
    fout << -1 << "\t" << centroid_P.x() << "\t" << centroid_P.y() << "\t" << centroid_P.z() << std::endl; 
    for (int i = 0; i < P_shifted.rows(); i++)
    {
      fout << i << "\t" << P_shifted(i,0) << "\t" << P_shifted(i,1) << "\t" << P_shifted(i,2) << std::endl;
    }
    fout.close();
  }
  std::string f_name_des = "dest_shift.txt";
  std::ofstream fout_des(f_name_des);
  if (fout_des.is_open())
  {
    fout_des << -1 << "\t" << centroid_Q.x() << "\t" << centroid_Q.y() << "\t" << centroid_Q.z() << std::endl;
    for (int i = 0; i < Q_shifted.rows(); i++)
    {
      fout_des << i << "\t" << Q_shifted(i,0) << "\t" << Q_shifted(i,1) << "\t" << Q_shifted(i,2) << std::endl;
    }
    fout_des.close();
  }
  // H
  Eigen::Matrix3d H = P_shifted.transpose() * Q;

  // SVD
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  R = V * U.transpose();
  std::cout << "R det: " << R.determinant() << std::endl;
  if (R.determinant() < -1)
  {
    V.col(2) *= -1;
    R = V * U.transpose();
  }

  t = centroid_Q - R * centroid_P;
}

// Function to compute the residuals and fill the Jacobian matrix
void ComputeResidualsAndJacobian(const std::vector<Eigen::Vector3d>& points_source,
                                 const std::vector<Eigen::Vector3d>& points_target,
                                 const Eigen::Vector3d& t, const Eigen::Vector3d& rpy,
                                 Eigen::VectorXd& residuals, Eigen::MatrixXd& jacobian)
{
    size_t n_points = points_source.size();
    
    // Initialize residuals and Jacobian matrices
    residuals.resize(3 * n_points);
    jacobian.resize(3 * n_points, 6); // 3 rows for each point, 6 columns for t_x, t_y, t_z, roll, pitch, yaw

    // Rotation matrix from roll, pitch, yaw
    double roll = rpy(0), pitch = rpy(1), yaw = rpy(2);
        // #PROC Compute sines and cosines
    double sinr = std::sin(roll);
    double cosr = std::cos(roll);
    double sinp = std::sin(pitch);
    double cosp = std::cos(pitch);
    double siny = std::sin(yaw);
    double cosy = std::cos(yaw);
    // #END

    Eigen::Matrix3d R = rpy2R(rpy);

    for (size_t i = 0; i < n_points; ++i)
    {
      // Transform the source point
      Eigen::Vector3d transformed_point = t + R * points_source[i];
      
      // Residuals: difference between the transformed source point and the target point
      Eigen::Vector3d res = points_target[i] - transformed_point;
      residuals.segment<3>(3 * i) = res;
      
      // Derivatives w.r.t. roll, pitch, yaw
      double x2 = points_source[i].x();
      double y2 = points_source[i].y();
      double z2 = points_source[i].z() ;
         // Derivative of R * P w.r.t
      double Nx = x2*(cosp*cosy) + y2*(-cosr*siny + sinr*sinp*cosy) + z2*(sinr*siny + cosr*sinp*cosy);
      double Ny = x2*(cosp*siny) + y2*( cosr*cosy + sinr*sinp*siny) + z2*(-sinr*cosy + cosr*sinp*siny);
      double D = -x2*sinp + y2*sinr*cosp + z2 * cosr*cosp;

      // #PROC Get the derivations of [Nx Ny D]  w.r.t roll
      double dNx_droll = y2*( sinr*siny + cosr*sinp*cosy) + z2* (cosr*siny - sinr*sinp*cosy);
      double dNy_droll = y2*(-sinr*cosy + cosr*sinp*siny) + z2* (-cosr*cosy - sinr*sinp*siny);
      double dD_droll  = y2*(cosr*cosp) -z2*sinr*cosp; 
      // #END
      
      // #PROC Get the derivations of [Nx Ny D] w.r.t pitch
      double dNx_dpitch = x2*(-sinp*cosy) + y2*(sinr*cosp*cosy) + z2*(cosr*cosp*cosy);
      double dNy_dpitch = x2*(-sinp*siny) + y2*(sinr*cosp*siny) + z2*(cosr*cosp*siny);
      double dD_dpitch  = -x2*cosp -y2*sinr*sinp -z2* cosr*sinp; 
      // #END

      // #PROC Get the derivations of [Nx Ny D] w.r.t yaw
      double dNx_dyaw = x2*(-cosp*siny) + y2*(-cosr*cosy - sinr*sinp*siny) + z2* (sinr*cosy - cosr*sinp*siny);
      double dNy_dyaw = x2*( cosp*cosy) + y2*(-cosr*siny + sinr*sinp*cosy) + z2* (sinr*siny + cosr*sinp*cosy);
      double dD_dyaw = 0;
      // #PROC

      // Compute partial derivatives (Jacobian)
      jacobian.block<3, 3>(3 * i, 0) = -Eigen::Matrix3d::Identity();  // Derivatives w.r.t. tx, ty, tz
      
      jacobian(3 * i, 3) = -dNx_droll; 
      jacobian(3 * i, 4) = -dNx_dpitch; 
      jacobian(3 * i, 5) = -dNx_dyaw; 

      jacobian(3 * i+1, 3) = -dNy_droll; 
      jacobian(3 * i+1, 4) = -dNy_dpitch; 
      jacobian(3 * i+1, 5) = -dNy_dyaw; 

      jacobian(3 * i+2, 3) = -dD_droll; 
      jacobian(3 * i+2, 4) = -dD_dpitch; 
      jacobian(3 * i+2, 5) = -dD_dyaw; 
    }
}

// Function to perform the refinement using non-linear least squares
void NonlinearRefinement(const std::vector<Eigen::Vector3d>& points_source,
                         const std::vector<Eigen::Vector3d>& points_target,
                         Eigen::Vector3d& t, Eigen::Vector3d& rpy)
{
  const int max_iters = 10;
  const double tolerance = 1e-6;

  for (int iter = 0; iter < max_iters; ++iter)
  {
      Eigen::VectorXd residuals;
      Eigen::MatrixXd jacobian;

      // Compute residuals and Jacobian
      ComputeResidualsAndJacobian(points_source, points_target, t, rpy, residuals, jacobian);
      std::cout << "Iter: " << iter << ", res norm: " << residuals.norm() << std::endl;
      // Solve the linear system using least squares: (J^T * J) delta = J^T * residuals
      Eigen::VectorXd update = jacobian.transpose() * residuals;
      Eigen::MatrixXd H = jacobian.transpose() * jacobian;
      Eigen::VectorXd delta = H.ldlt().solve(update);

      // Update translation and rotation parameters
      t += delta.head<3>();
      rpy += delta.tail<3>();

      // Check for convergence
      if (delta.norm() < tolerance)
      {
          std::cout << "Converged in " << iter + 1 << " iterations." << std::endl;
          break;
      }
  }
}

