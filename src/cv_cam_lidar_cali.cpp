#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

class OneShotCalibrationNode : public rclcpp::Node {
public:
    OneShotCalibrationNode() : Node("one_shot_calibration_node") {
        board_size_ = cv::Size(10, 7);
        square_size_ = 0.02;

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/flir_camera/image_raw", 10,
            std::bind(&OneShotCalibrationNode::image_callback, this, std::placeholders::_1));

        pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ouster/points", rclcpp::SensorDataQoS(),
            std::bind(&OneShotCalibrationNode::pc_callback, this, std::placeholders::_1));

        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/calibrated_image", 10);

        got_image_ = false;
        got_cloud_ = false;
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

    cv_bridge::CvImagePtr latest_image_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr latest_cloud_;

    cv::Size board_size_;
    float square_size_;
    bool got_image_;
    bool got_cloud_;

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (got_image_) return;
        try {
            latest_image_ = cv_bridge::toCvCopy(msg, "bgr8");
            got_image_ = true;
            try_calibration();
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
        }
    }

    void pc_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (got_cloud_) return;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);
        latest_cloud_ = cloud;
        got_cloud_ = true;
        try_calibration();
    }

    void try_calibration() {
        if (!got_image_ || !got_cloud_) return;

        std::vector<cv::Point2f> image_points;
        bool found = cv::findChessboardCorners(latest_image_->image, board_size_, image_points);
        if (!found) {
            RCLCPP_WARN(this->get_logger(), "Checkerboard not found.");
            return;
        }

        cv::Mat gray;
        if (latest_image_->image.channels() == 3)
            cv::cvtColor(latest_image_->image, gray, cv::COLOR_BGR2GRAY);
        else
            gray = latest_image_->image;
        cv::cornerSubPix(gray, image_points, cv::Size(5, 5), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01));

        std::vector<cv::Point3f> object_points;
        for (int i = 0; i < board_size_.height; ++i)
            for (int j = 0; j < board_size_.width; ++j)
                object_points.emplace_back(j * square_size_, i * square_size_, 0.0);

        // LiDAR 평면 추출
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(latest_cloud_);

        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
        seg.segment(*inliers, *coeffs);

        if (inliers->indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "No plane found in LiDAR");
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(latest_cloud_);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane_cloud);

        // LiDAR 포인트 일부 대응
        std::vector<cv::Point3f> lidar_points;
        int n = std::min((int)object_points.size(), (int)plane_cloud->points.size());
        for (int i = 0; i < n; ++i) {
            const auto &pt = plane_cloud->points[i];
            lidar_points.emplace_back(pt.x, pt.y, pt.z);
        }

        if (image_points.size() != lidar_points.size()) {
            RCLCPP_WARN(this->get_logger(), "Size mismatch between image and lidar points");
            return;
        }

        // 실제 카메라 내부 파라미터 사용
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1046.33, 0, 705.29, 0, 1052.94, 568.63, 0, 0, 1);
        cv::Mat dist = (cv::Mat_<double>(1, 5) << -0.146, 0.473, 0.001, -0.0009, -0.543);

        cv::Mat rvec, tvec;
        bool ok = cv::solvePnP(lidar_points, image_points, K, dist, rvec, tvec);
        if (!ok) {
            RCLCPP_ERROR(this->get_logger(), "solvePnP failed");
            return;
        }

        // 전체 LiDAR 포인트를 이미지에 투영
        std::vector<cv::Point2f> projected_points;
        std::vector<cv::Point3f> lidar_all;
        for (const auto &pt : latest_cloud_->points)
            lidar_all.emplace_back(pt.x, pt.y, pt.z);

        cv::projectPoints(lidar_all, rvec, tvec, K, dist, projected_points);

        // 점 시각화
        cv::Mat img = latest_image_->image.clone();
        for (const auto &pt : projected_points) {
            if (pt.x >= 0 && pt.x < img.cols && pt.y >= 0 && pt.y < img.rows)
                cv::circle(img, pt, 1, cv::Scalar(0, 255, 255), -1);
        }

        auto out_msg = cv_bridge::CvImage(latest_image_->header, "bgr8", img).toImageMsg();
        image_pub_->publish(*out_msg);
        RCLCPP_INFO(this->get_logger(), "Published projected LiDAR onto image.");
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OneShotCalibrationNode>());
    rclcpp::shutdown();
    return 0;
}
