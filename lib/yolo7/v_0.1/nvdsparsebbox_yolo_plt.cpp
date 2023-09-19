#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

extern "C" bool NvDsInferParseYolo7NMS(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams, std::vector<NvDsInferParseObjectInfo> &objectList);

static bool NvDsInferParseCustomYolo7(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams, std::vector<NvDsInferParseObjectInfo> &objectList,
    const uint &numClasses);

// ------------------ ROI CHECKING -------------------------
struct Point
{
    int x;
    int y;
};
struct ROI
{
    //3840 × 2160
    // change this part and add polygon
    std::vector<Point> vertices{{0, 0}, {3840/2, 0}, {3840/2, 2160}, {0, 2160}};
};

// ray trace algorithm to check if a point is inside n vertises polygone
bool isPointInsideROI(const Point &point, const ROI &roi)
{
    bool inside = false;
    int n = roi.vertices.size();

    for (int i = 0, j = n - 1; i < n; j = i++)
    {
        if (((roi.vertices[i].y > point.y) != (roi.vertices[j].y > point.y)) &&
            (point.x < (roi.vertices[j].x - roi.vertices[i].x) * (point.y - roi.vertices[i].y) / (roi.vertices[j].y - roi.vertices[i].y) + roi.vertices[i].x))
        {
            inside = !inside;
        }
    }
    return inside;
}
// ------------------ ROI CHECKING -------------------------

// modified for number plate info
static bool NvDsInferParseCustomYolo7(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams, std::vector<NvDsInferParseObjectInfo> &objectList,
    const uint &numClasses)
{
    if (outputLayersInfo.empty())
    {
        std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
        return false;
    }

    int *num_dets = (int *)outputLayersInfo[0].buffer;
    float *det_boxes = (float *)outputLayersInfo[1].buffer;
    float *det_scores = (float *)outputLayersInfo[2].buffer;
    int *det_classes = (int *)outputLayersInfo[3].buffer;

    objectList.reserve(num_dets[0]);
    // define roi
    ROI roi;

    for (int i = 0; i < num_dets[0]; i++)
    {
        NvDsInferParseObjectInfo single_object;

        single_object.left = det_boxes[i * 4 + 0];
        single_object.top = det_boxes[i * 4 + 1];
        single_object.width = det_boxes[i * 4 + 2] - det_boxes[i * 4 + 0];
        single_object.height = det_boxes[i * 4 + 3] - det_boxes[i * 4 + 1];

        // check if center of the object is inside roi
        if ( isPointInsideROI(
            //3840 × 2160
            Point{(single_object.left + 0.5 *single_object.width)/640 * 3840,
            (single_object.top + 0.5 *single_object.height)/640 * 2160},
             roi) ) {
            continue;
        }
        //
        single_object.detectionConfidence = det_scores[i];
        single_object.classId = det_classes[i];

        objectList.emplace_back(single_object); // make filling objects faster as we have number of detections info at hand
    }

    return true;
}

extern "C" bool NvDsInferParseYolo7NMS(std::vector<NvDsInferLayerInfo> const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,
                                       NvDsInferParseDetectionParams const &detectionParams, std::vector<NvDsInferParseObjectInfo> &objectList)
{
    // objectList.clear();
    return NvDsInferParseCustomYolo7(
        outputLayersInfo, networkInfo, detectionParams, objectList, 1);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo7NMS)