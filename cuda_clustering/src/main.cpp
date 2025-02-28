#include "cuda_clustering/cuda_clustering.hpp"
#include <unistd.h>

void handleSignal(int signal) {
    if (signal == SIGINT) {
        std::cout << "Received SIGINT. Killing clustering_plane_finder_cpu process.\n";
        rclcpp::shutdown();
    }
}

int main(int argc, char** argv) {
  signal(SIGINT, handleSignal);
  rclcpp::init(argc, argv);

  auto node = std::make_shared<CudaClusteringNode>();

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
