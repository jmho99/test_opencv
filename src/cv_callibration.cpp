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
    declare_parameter("select_connect", "usb");
    declare_parameter("device_path", "/dev/video1");
    declare_parameter("checkerboard_cols", 10);
    declare_parameter("checkerboard_rows", 7);
    declare_parameter("square_size", 0.020);
    declare_parameter("frame_width", 640);
    declare_parameter("frame_height", 480);

    get_parameter("select_connect", select_connect_);
    get_parameter("device_path", device_path_);
    get_parameter("checkerboard_cols", cols_);
    get_parameter("checkerboard_rows", rows_);
    get_parameter("square_size", square_size_);
    get_parameter("frame_width", frame_width_);
    get_parameter("frame_height", frame_height_);

    RCLCPP_INFO(this->get_logger(), "Open camera using %s", select_connect_.c_str());
    RCLCPP_INFO(this->get_logger(), "checkerboard %d x %d", cols_, rows_);
    RCLCPP_INFO(this->get_logger(), "checkerboard_size %f%s", square_size_, " m");
    RCLCPP_INFO(this->get_logger(), "frame_size %d x %d", frame_width_, frame_height_);

    if (select_connect_ == "usb") {
      connect_usb_camera();
    }
    else if (select_connect_ == "ethernet") {
      subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/flir_camera/image_raw", rclcpp::SensorDataQoS(),
      std::bind(&CalibrationNode::image_callback, this, std::placeholders::_1));
      RCLCPP_INFO(this->get_logger(), "Open camera using ETHERNET");
    }

    save_path_ = "./calib_images/";
    fs::create_directories(save_path_);

  }

private:
  void connect_usb_camera() {
    cv::VideoCapture cap;
    cap.open(device_path_,cv::CAP_V4L2);
    if (!cap.isOpened()) {
      RCLCPP_INFO(this -> get_logger(), "ERROR_open");
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Open camera using USB");

    cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width_);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height_);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));

    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
      RCLCPP_INFO(this -> get_logger(), "ERROR_frame");
      return;
    }
    while (true) {
      cap >> frame;
      cv::imshow("MJPEG CAM", frame);
      input_keyboard(frame);
    }
  }
  
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
      current_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
      cv::imshow("FLIR View", current_frame_);
      input_keyboard(current_frame_);
    } catch (cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
    }
  }
  
  void input_keyboard(const cv::Mat &frame) {
    int key = cv::waitKey(1);
    if (key == 's') {
      save_current_frame(frame.clone());
    } else if (key == 'c') {
      run_calibration_from_folder();
    } else if (key == 'e') {
      calibration_error(obj_points_, img_points_, rvecs_, tvecs_, camera_matrix_, dist_coeffs_, image_files_);
    }
  }

  void save_current_frame(const cv::Mat &frame) {
    std::string filename = save_path_ + "img_" + std::to_string(frame_counter_) + ".png";
    cv::imwrite(filename, frame);
    RCLCPP_INFO(this->get_logger(), "Save Image: %s", filename.c_str());
    frame_counter_++;
  }

  void run_calibration_from_folder() {
    RCLCPP_INFO(this->get_logger(), "Start callibration...");
    
    cv::glob(save_path_ + "*.png", image_files_);
    if (image_files_.size() < 5) {
      RCLCPP_WARN(this->get_logger(), "Not enough image (%lu)", image_files_.size());
      return;
    }
    
    cv::Size pattern_size(cols_, rows_);
    float square_size = square_size_;
    std::vector<cv::Point3f> objp;

    for (int i = 0; i < pattern_size.height; ++i)
      for (int j = 0; j < pattern_size.width; ++j)
        objp.emplace_back(j * square_size, i * square_size, 0.0f);

    for (size_t idx = 0; idx < image_files_.size(); ++idx) {
      const auto &file = image_files_[idx];
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
        img_points_.push_back(corners);
        obj_points_.push_back(objp);

        cv::Mat vis = img.clone();
        cv::drawChessboardCorners(vis, pattern_size, corners, found);

        for (size_t i = 0; i < corners.size(); ++i) {
          cv::putText(vis, std::to_string(i), corners[i],
                  cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        }

        std::string save_name = save_path_ + "img_" + std::to_string(idx) + "_calib_result.png";
        cv::imwrite(save_name, vis);
        RCLCPP_INFO(this->get_logger(), "Save calibration image: %s", save_name.c_str());
      }

      if (img_points_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Failed callibration.");
        return;
      }

      
      
      double rms = cv::calibrateCamera(obj_points_, img_points_, cv::Size(frame_width_, frame_height_),
                                     camera_matrix_, dist_coeffs_, rvecs_, tvecs_);

      RCLCPP_INFO(this->get_logger(), "RMS error: %.4f", rms);
      cv::FileStorage fs("calibration_result.yaml", cv::FileStorage::WRITE);
      fs << "camera_matrix_" << camera_matrix_;
      fs << "distortion_coefficients" << dist_coeffs_;
      fs << "rotation" << rvecs_;
      fs << "translation" << tvecs_;
      fs.release();
      RCLCPP_INFO(this->get_logger(), "Successed result saving: calibration_result.yaml");
      
    }
    
  }

  void calibration_error(const std::vector<std::vector<cv::Point3f>>& obj_points_,
                        const std::vector<std::vector<cv::Point2f>>& img_points_,
                        const std::vector<cv::Mat>& rvecs_,
                        const std::vector<cv::Mat>& tvecs_,
                        const cv::Mat& camera_matrix_,
                        const cv::Mat& dist_coeffs_,
                        std::vector<cv::String>& image_files_) {
  

cv::glob(save_path_ + "*.png", image_files_);
  for (size_t i = 0; i < image_files_.size(); ++i) {
    cv::Mat img = cv::imread(image_files_[i]);
    if (img.empty()) {
      RCLCPP_WARN(rclcpp::get_logger("calibration_error"), 
                  "Image load failed: %s", image_files_[i].c_str());
      continue;
    }

    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(obj_points_[i], rvecs_[i], tvecs_[i], 
                      camera_matrix_, dist_coeffs_, projected_points);
    

    cv::Mat ideal_dist = cv::Mat::zeros(1, 5, CV_64F);
    double i_mat[] = {1014.5, 0.0, 720.0,
                        0.0, 1014.5, 540.0,
                        0.0, 0.0, 1.0};
    cv::Mat ideal_camera_matrix(3, 3, CV_64F, i_mat);      

    std::vector<cv::Point2f> projected_ideal;
    cv::projectPoints(obj_points_[i], rvecs_[i], tvecs_[i],
                      ideal_camera_matrix, ideal_dist, projected_ideal);
    cv::Mat vis = img.clone();
    for (size_t j = 0; j < img_points_[i].size(); ++j) {
      cv::Point2f actual = img_points_[i][j];
      cv::Point2f reprojected = projected_points[j];
      cv::Point2f& reproj_ideal = projected_ideal[j];


      cv::circle(vis, actual, 2, cv::Scalar(0, 255, 0), -1);     // 실제 코너: 초록
      cv::circle(vis, reprojected, 2, cv::Scalar(0, 0, 255), -1); // 재투영 코너: 빨강
      cv::circle(vis, reproj_ideal, 4, cv::Scalar(255, 0, 0), -1); // 재투영 코너: 빨강
      cv::line(vis, actual, reprojected, cv::Scalar(255, 0, 0), 1); // 연결선: 파랑

      
    }
    std::string save_name = save_path_ + "img_" + std::to_string(i) + "_error_calib_result.png";
      cv::imwrite(save_name, vis);

      cv::Point3f p = obj_points_[i][3];
      cv::Point2f c = img_points_[i][3];
      
      std::cout << "세 번째 코너의 월드 좌표: " << p << std::endl;
      std::cout << "세 번째 코너의 이미지 좌표: " << c << std::endl;

  }

  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  std::string save_path_;
  cv::Mat current_frame_;
  int frame_counter_;

  std::string select_connect_;
  std::string device_path_;
  int cols_;
  int rows_;
  float square_size_;
  int frame_width_;
  int frame_height_;

  std::vector<std::vector<cv::Point2f>> img_points_;
  std::vector<std::vector<cv::Point3f>> obj_points_;
  std::vector<cv::Mat> rvecs_, tvecs_;
  cv::Mat camera_matrix_, dist_coeffs_;
  std::vector<cv::String> image_files_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CalibrationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}