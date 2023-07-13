//
// Created by ek on 29.07.2022.
//

#ifndef TINYYOLOV2_NVDSPARSEBBOX_TINY_YOLO_H
#define TINYYOLOV2_NVDSPARSEBBOX_TINY_YOLO_H




#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>



/**
 * Function expected by DeepStream for decoding the TinyYOLOv2 output.
 *
 * C-linkage [extern "C"] was written to prevent name-mangling. This function must return true after
 * adding all bounding boxes to the objectList vector.
 *
 * @param [outputLayersInfo] std::vector of NvDsInferLayerInfo objects with information about the output layer.
 * @param [networkInfo] NvDsInferNetworkInfo object with information about the TinyYOLOv2 network.
 * @param [detectionParams] NvDsInferParseDetectionParams with information about some config params.
 * @param [objectList] std::vector of NvDsInferParseObjectInfo objects to which bounding box information must
 * be stored.
 *
 * @return true
 */

// This is just the function prototype. The definition is written at the end of the file.
extern "C" bool NvDsInferParseCustomYoloV2Tiny(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
        NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams,
        std::vector<NvDsInferParseObjectInfo>& objectList);


/**
 * Bounds values between the range [minVal, maxVal].
 *
 * Values that are out of bounds are set to their boundary values.
 * For example, consider the following: clamp(bbox.left, 0, netW). This
 * translates to min(netW, max(0, bbox.left)). Hence, if bbox.left was
 * negative, it is set to 0. If bbox.left is greater than netW, it is set
 * to netW.
 *
 * @param [val] Value to be bound.
 * @param [minVal] Lower bound.
 * @param [maxVal] Upper bound.
 *
 * @return A value that is bound in the range [minVal, maxVal].
 */
static unsigned clamp(const uint val, const uint minVal, const uint maxVal);

static float overlap1D(float x1min, float x1max, float x2min, float x2max);

static float computeIoU(const NvDsInferParseObjectInfo& bbox1, const NvDsInferParseObjectInfo& bbox2);

static bool compareBBoxConfidence(const NvDsInferParseObjectInfo& bbox1, const NvDsInferParseObjectInfo& bbox2);


static NvDsInferParseObjectInfo createBBox(const float& bx, const float& by, const float& bw,
                                           const float& bh, const int& stride, const uint& netW,
                                           const uint& netH);


static void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                            const uint stride, const uint& netW, const uint& netH, const int maxIndex,
                            const float maxProb, std::vector<NvDsInferParseObjectInfo>& bboxInfo);


static std::vector<NvDsInferParseObjectInfo>
nonMaximumSuppression(std::vector<NvDsInferParseObjectInfo> inputBBoxInfo, const float nmsThresh);


static std::vector<NvDsInferParseObjectInfo>
nmsAllClasses(std::vector<NvDsInferParseObjectInfo>& bboxInfo, const uint numClasses, const float nmsThresh);


static std::vector<NvDsInferParseObjectInfo>
decodeYoloV2Tensor(
        const float* detections, const uint& netW, const uint& netH,
        const std::vector<float> &anchors, const uint numBBoxes,
        const uint gridSize, const uint stride, const float probThresh,
        const uint numOutputClasses);


extern "C" bool NvDsInferParseCustomYoloV2Tiny(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
        NvDsInferNetworkInfo const& networkInfo,
        NvDsInferParseDetectionParams const& detectionParams,
        std::vector<NvDsInferParseObjectInfo>& objectList);



/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV2Tiny);



#endif //TINYYOLOV2_NVDSPARSEBBOX_TINY_YOLO_H
