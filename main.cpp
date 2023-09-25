#include <iostream>
#include "include/ds_yolo.h"

// commandline to run : ./YoloObjectDetector_dm /home/ek/EkinStash/testSpeedData/testVideos/test003.h264 ../results/out003.mp4

int main(int argc, char** argv) {
    yolo_deepstream(argc, argv);
    std::cout << "Finished!" << std::endl;
    return 0;
}
