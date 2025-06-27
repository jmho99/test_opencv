#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

class cv_cam_lidar_cali : public rclcpp::Node
{
public:
    cv_cam_lidar_cali()
    : Node("cv_cam_lidar_cali")
    {
        board_size_ = cv::Size(10, 7);
        square_size_ = 0.02; // m

        camera_matrix_ = (cv::Mat_<double>(3,3) <<
             1.0412562786381386e+03, 0., 6.8565540026982239e+02, 0.,
       1.0505976084535532e+03, 5.9778012298164469e+02, 0., 0., 1.);

        distortion_coeffs_ = (cv::Mat_<double>(1,5) <<
            -1.3724382468171908e-01, 4.9079709117302012e-01,
       8.2971299771431115e-03, -4.5215579888173568e-03,
       -7.7949268098546165e-01);

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/flir_camera/image_raw", 10,
            std::bind(&cv_cam_lidar_cali::image_callback, this, std::placeholders::_1));

        pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ouster/points", rclcpp::SensorDataQoS(),
            std::bind(&cv_cam_lidar_cali::pc_callback, this, std::placeholders::_1));
    }

private:
    // ROS subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;

    // Latest data
    cv::Mat last_image_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr last_cloud_{new pcl::PointCloud<pcl::PointXYZ>};

    // Chessboard parameters
    cv::Size board_size_;
    double square_size_;

    // Camera intrinsics
    cv::Mat camera_matrix_;
    cv::Mat distortion_coeffs_;

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        last_image_ = cv_bridge::toCvCopy(msg, "bgr8")->image;

        if (!last_image_.empty()) {
            cv::imshow("Camera Image", last_image_);
            input_keyboard(last_image_);
        }
    }

    void pc_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::fromROSMsg(*msg, *last_cloud_);
    }

    void input_keyboard(const cv::Mat &frame)
    {
        int key = cv::waitKey(1);
        if (key == 's') {
            save_current_frame(frame.clone());
        } else if (key == 'c') {
            run_calibration_from_folder();
        } else if (key == 'e') {
            calibration_error();
        }
    }

    void save_current_frame(const cv::Mat &image)
    {
        // Save image
        fs::create_directories("capture");
        std::string img_path = "capture/image.png";
        cv::imwrite(img_path, image);

        // Save point cloud
        if (last_cloud_->empty()) {
            RCLCPP_WARN(this->get_logger(), "No point cloud captured yet!");
            return;
        }
        std::string pc_path = "capture/pointcloud.pcd";
        pcl::io::savePCDFileBinary(pc_path, *last_cloud_);
        RCLCPP_INFO(this->get_logger(), "Saved image and pointcloud.");
    }

    void run_calibration_from_folder()
    {
    // Check files exist
        std::string img_path = "capture/image.png";
        std::string pc_path = "capture/pointcloud.pcd";
        if (!fs::exists(img_path) || !fs::exists(pc_path)) {
            RCLCPP_WARN(this->get_logger(), "No saved data. Capture first!");
            return;
        }

        // Load color image
        cv::Mat img_color = cv::imread(img_path, cv::IMREAD_COLOR);
        if (img_color.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load saved image!");
            return;
        }
            
        // Convert to grayscale
        cv::Mat img_gray;
        cv::cvtColor(img_color, img_gray, cv::COLOR_BGR2GRAY);
        
        // Find corners
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(img_gray, board_size_, corners);
        if (!found) {
            RCLCPP_ERROR(this->get_logger(), "Chessboard not found in image!");
            return;
        }
            

        cv::cornerSubPix(img_gray, corners, cv::Size(11,11),
                         cv::Size(-1,-1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

        // 3D object points in chessboard coordinate system
        std::vector<cv::Point3f> object_points;
        for (int i = 0; i < board_size_.height; i++) {
            for (int j = 0; j < board_size_.width; j++) {
                object_points.emplace_back(j * square_size_, i * square_size_, 0.0);
            }
        }

        // Solve PnP
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(object_points, corners,
                                    camera_matrix_, distortion_coeffs_,
                                    rvec, tvec);

        if (!success) {
            RCLCPP_ERROR(this->get_logger(), "solvePnP failed!");
            return;
        }

        // Load pointcloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile(pc_path, *cloud);

        // 추출 예시: 평면 검출
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0) {
            RCLCPP_ERROR(this->get_logger(), "Plane not found in point cloud.");
            return;
        }

        // 라이다 평면 상 점들 추출
        pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane_cloud);

        // plane_cloud 중심 좌표를 chessboard 중심과 맞추는 식으로 R, t 추정 가능
        // 여기서는 예시로 그냥 출력
        RCLCPP_INFO(this->get_logger(), "PnP rvec: [%f %f %f]", rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
        RCLCPP_INFO(this->get_logger(), "PnP tvec: [%f %f %f]", tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

        // save R, t
        save_rt(rvec, tvec);

        // 여기서 따로 만든 함수 호출
    compute_lidar_camera_extrinsic(pc_path, rvec, tvec, board_size_, square_size_);
    }

    void save_rt(const cv::Mat& rvec, const cv::Mat& tvec)
    {
        std::ofstream file("capture/extrinsic.txt");
        file << "rvec: "
             << rvec.at<double>(0) << " "
             << rvec.at<double>(1) << " "
             << rvec.at<double>(2) << "\n";
        file << "tvec: "
             << tvec.at<double>(0) << " "
             << tvec.at<double>(1) << " "
             << tvec.at<double>(2) << "\n";
        file.close();

        RCLCPP_INFO(this->get_logger(), "Saved extrinsic parameters.");
    }

    void compute_lidar_camera_extrinsic(
    const std::string &pc_path,
    const cv::Mat &rvec,
    const cv::Mat &tvec,
    const cv::Size &board_size,
    double square_size)
{
    // Load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(pc_path, *cloud) < 0) {
        std::cerr << "Failed to load point cloud: " << pc_path << std::endl;
        return;
    }

    // 1. 평면 추출
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    if (!extractPlaneFromPointCloud(cloud, plane_cloud, coefficients)) {
        std::cerr << "Plane extraction failed!" << std::endl;
        return;
    }

    // 2. LiDAR 평면 점들을 체커보드 좌표계로 변환
    std::vector<cv::Point3f> lidar_points_cb =
        transformLidarPointsToChessboard(plane_cloud, rvec, tvec);

    // 3. 체커보드 기준 object points 생성
    std::vector<cv::Point3f> object_points;
    for (int i = 0; i < board_size.height; i++) {
        for (int j = 0; j < board_size.width; j++) {
            object_points.emplace_back(j * square_size, i * square_size, 0.0);
        }
    }

    // 크기 맞춤
    size_t n_points = std::min(lidar_points_cb.size(), object_points.size());
    lidar_points_cb.resize(n_points);
    object_points.resize(n_points);

    // 4. SVD 기반 R, t 계산
    cv::Mat R_lidar_to_cam, t_lidar_to_cam;
    computeRigidTransformSVD(lidar_points_cb, object_points, R_lidar_to_cam, t_lidar_to_cam);

    std::cout << "[Lidar → Camera] R:\n" << R_lidar_to_cam << std::endl;
    std::cout << "[Lidar → Camera] t:\n" << t_lidar_to_cam << std::endl;

    // 저장
    std::ofstream ofs("capture/extrinsic_lidar_to_camera.txt");
    ofs << "R:\n" << R_lidar_to_cam << "\nt:\n" << t_lidar_to_cam << std::endl;
    ofs.close();
}

       void calibration_error()
    {
        // TODO: 재투영 오차 계산
        RCLCPP_INFO(this->get_logger(), "Calibration error check not implemented yet.");
    }

    bool extractPlaneFromPointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &plane_cloud,
    pcl::ModelCoefficients::Ptr &coefficients)
{
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        return false;
    }

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*plane_cloud);

    return true;
}

std::vector<cv::Point3f> transformLidarPointsToChessboard(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &plane_cloud,
    const cv::Mat &rvec, const cv::Mat &tvec)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);  // 회전벡터 → 회전행렬 변환

    std::vector<cv::Point3f> transformed_points;
    for (const auto& pt : plane_cloud->points) {
        cv::Mat pt_mat = (cv::Mat_<double>(3,1) << pt.x, pt.y, pt.z);
        cv::Mat pt_cb = R.t() * (pt_mat - tvec); // 카메라 좌표계 → 체커보드 좌표계 (역변환)
        transformed_points.emplace_back(pt_cb.at<double>(0), pt_cb.at<double>(1), pt_cb.at<double>(2));
    }
    return transformed_points;
}

void computeRigidTransformSVD(
    const std::vector<cv::Point3f> &src_points,
    const std::vector<cv::Point3f> &dst_points,
    cv::Mat &R, cv::Mat &t)
{
    assert(src_points.size() == dst_points.size());
    int N = (int)src_points.size();

    // Eigen 행렬 변환
    Eigen::MatrixXd src(3, N), dst(3, N);
    for (int i = 0; i < N; i++) {
        src(0, i) = src_points[i].x;
        src(1, i) = src_points[i].y;
        src(2, i) = src_points[i].z;

        dst(0, i) = dst_points[i].x;
        dst(1, i) = dst_points[i].y;
        dst(2, i) = dst_points[i].z;
    }

    // 평균 계산
    Eigen::Vector3d src_mean = src.rowwise().mean();
    Eigen::Vector3d dst_mean = dst.rowwise().mean();

    // 중심화
    src.colwise() -= src_mean;
    dst.colwise() -= dst_mean;

    // 공분산 행렬 계산
    Eigen::Matrix3d H = src * dst.transpose();

    // SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Matrix3d R_eigen = V * U.transpose();

    // 반사 행렬 체크
    if (R_eigen.determinant() < 0) {
        V.col(2) *= -1;
        R_eigen = V * U.transpose();
    }

    Eigen::Vector3d t_eigen = dst_mean - R_eigen * src_mean;

    // Eigen → OpenCV 변환
    cv::eigen2cv(R_eigen, R);
    t = (cv::Mat_<double>(3,1) << t_eigen(0), t_eigen(1), t_eigen(2));
}
};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<cv_cam_lidar_cali>());
    rclcpp::shutdown();
    return 0;
}
