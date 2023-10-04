#
import os
import subprocess





if __name__=="__main__":
    # Run myprogram with arguments arg1, arg2, and arg3
    print(os.getcwd())
    subprocess.run(
        ["./build/YoloObjectDetector_dm", 
                    "~/EkinStash/testSpeedData/BritishSchool/test_01.h264", 
                    "results/british_01.mp4"]
                    )