#include <iostream>
#include "include/ds_yolo.h"

// /home/ek/EkinStash/testSpeedData/testVideos/test003.h264 ../results/test003.mp4
// 3840 × 2160

int main(int argc, char** argv) {



    yolo_deepstream(argc, argv);


    std::cout << "Finished!" << std::endl;
    return 0;
}
