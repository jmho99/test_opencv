#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"
#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp> 
#include <opencv2/opencv.hpp>

#include <filesystem>


namespace fs = std::filesystem;

class CalibrationNode : public rclcpp::Node {
public:
  CalibrationNode() : Node("cv_calibration_node"), frame_counter_(0) {
    using std::placeholders::_1;

    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/flir_camera/image_raw", rclcpp::SensorDataQoS(),
      std::bind(&CalibrationNode::image_callback, this, _1));

    save_path_ = "./calib_images/";
    fs::create_directories(save_path_);

    RCLCPP_INFO(this->get_logger(), "노드 초기화 완료, 이미지 수신 대기 중...");
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      current_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
      cv::imshow("FLIR View", current_frame_);

      int key = cv::waitKey(1);
      if (key == 'a') {
        save_current_frame(current_frame_);
      } else if (key == 'c') {
        run_calibration_from_folder();
      }
    } catch (cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge 에러: %s", e.what());
    }
  }

  void save_current_frame(const cv::Mat &frame) {
    std::string filename = save_path_ + "img_" + std::to_string(frame_counter_) + ".png";
    cv::imwrite(filename, frame);
    RCLCPP_INFO(this->get_logger(), "이미지 저장됨: %s", filename.c_str());
    frame_counter_++;
  }

  void run_calibration_from_folder() {
    RCLCPP_INFO(this->get_logger(), "저장된 이미지로 캘리브레이션 시작...");

    std::vector<cv::String> image_files;
    cv::glob(save_path_ + "*.png", image_files);

    if (image_files.size() < 5) {
      RCLCPP_WARN(this->get_logger(), "이미지가 부족합니다 (%lu개)", image_files.size());
      return;
    }

    std::vector<std::vector<cv::Point2f>> img_points;
    std::vector<std::vector<cv::Point3f>> obj_points;

    cv::Size pattern_size(8, 5);
    float square_size = 0.015f;
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < pattern_size.height; ++i)
      for (int j = 0; j < pattern_size.width; ++j)
        objp.emplace_back(j * square_size, i * square_size, 0.0f);

    for (size_t idx = 0; idx < image_files.size(); ++idx) {
  const auto &file = image_files[idx];
  cv::Mat img = cv::imread(file);
      if (img.empty()) continue;

      std::vector<cv::Point2f> corners;
      bool found = cv::findChessboardCorners(img, pattern_size, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

      if (found) {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
        img_points.push_back(corners);
        obj_points.push_back(objp);

    cv::Mat vis = img.clone();
    cv::drawChessboardCorners(vis, pattern_size, corners, found);
    for (size_t i = 0; i < corners.size(); ++i) {
      cv::putText(vis, std::to_string(i), corners[i],
                  cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
    }

    std::string save_name = save_path_ + "img_" + std::to_string(idx) + "_calib_result.png";
    cv::imwrite(save_name, vis);
    RCLCPP_INFO(this->get_logger(), "결과 저장: %s", save_name.c_str());
  }

    if (img_points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "체커보드 인식 실패. 캘리브레이션 중단.");
      return;
    }

    cv::Mat camera_matrix, dist_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(obj_points, img_points, cv::Size(640, 480),
                                     camera_matrix, dist_coeffs, rvecs, tvecs);

    RCLCPP_INFO(this->get_logger(), "RMS 오류: %.4f", rms);
    cv::FileStorage fs("calibration_result.yaml", cv::FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "distortion_coefficients" << dist_coeffs;
    fs.release();
    RCLCPP_INFO(this->get_logger(), "결과 저장 완료: calibration_result.yaml");
    
  }
}

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  std::string save_path_;
  cv::Mat current_frame_;
  int frame_counter_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CalibrationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}