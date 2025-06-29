#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class PcdPublisher : public rclcpp::Node
{
public:
    PcdPublisher()
    : Node("pcd_publisher")
    {
        // 퍼블리셔 생성
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pcd_cloud", 10);
 
        // PCD 파일 로드
        pcl::PointCloud<pcl::PointXYZ> cloud;
        std::string filename = "/home/icrs/fusion_study/src/test_opencv/capture/pointcloud.pcd";

        if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, cloud) == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Couldn't read PCD file: %s", filename.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loaded %zu points from %s", cloud.points.size(), filename.c_str());

        // PCL → ROS2 메시지 변환
        pcl::toROSMsg(cloud, cloud_msg_);
        cloud_msg_.header.frame_id = "map"; // RViz에서 사용하는 좌표계 설정

        // 타이머로 주기적 퍼블리시
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&PcdPublisher::timerCallback, this));
    }

private:
    void timerCallback()
    {
        cloud_msg_.header.stamp = this->now();
        publisher_->publish(cloud_msg_);
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    sensor_msgs::msg::PointCloud2 cloud_msg_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PcdPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
