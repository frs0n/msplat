#include "loaders.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

void arkitColumnMajorToRowMajorCamToWorld(const std::vector<float> &m, float out[16]) {
    if (m.size() != 16) {
        throw std::runtime_error("memo frame transform must contain 16 floats");
    }

    out[0] = m[0];  out[1] = m[4];  out[2] = m[8];   out[3] = m[12];
    out[4] = m[1];  out[5] = m[5];  out[6] = m[9];   out[7] = m[13];
    out[8] = m[2];  out[9] = m[6];  out[10] = m[10]; out[11] = m[14];
    out[12] = 0;    out[13] = 0;    out[14] = 0;     out[15] = 1;
}

Camera cameraFromMemoFrame(const json &frame, const fs::path &imageDir) {
    if (!frame.value("acceptedKeyframe", false)) {
        throw std::runtime_error("internal error: non-keyframe passed to cameraFromMemoFrame");
    }
    if (!frame.contains("imageName")) {
        throw std::runtime_error("memo accepted keyframe is missing imageName");
    }

    Camera cam;
    auto resolution = frame.at("resolution").get<std::vector<int>>();
    auto intrinsics = frame.at("intrinsics").get<std::vector<float>>();
    auto transform = frame.at("transform").get<std::vector<float>>();

    if (resolution.size() != 2) {
        throw std::runtime_error("memo frame resolution must contain width and height");
    }
    if (intrinsics.size() != 9) {
        throw std::runtime_error("memo frame intrinsics must contain 9 floats");
    }

    cam.width = resolution[0];
    cam.height = resolution[1];
    cam.fx = intrinsics[0];
    cam.fy = intrinsics[4];
    cam.cx = intrinsics[6];
    cam.cy = intrinsics[7];
    cam.filePath = (imageDir / frame.at("imageName").get<std::string>()).string();
    arkitColumnMajorToRowMajorCamToWorld(transform, cam.camToWorld);
    return cam;
}

} // namespace

InputData loaders::loadMemo(const std::string &projectRoot) {
    fs::path root(projectRoot);
    fs::path framesPath = root / "arkit" / "frames.jsonl";
    fs::path imageDir = root / "images";
    fs::path pointPath = root / "depth" / "fused_points.ply";

    if (!fs::exists(framesPath)) {
        throw std::runtime_error("memo dataset missing arkit/frames.jsonl: " + projectRoot);
    }
    if (!fs::exists(imageDir)) {
        throw std::runtime_error("memo dataset missing images/: " + projectRoot);
    }
    if (!fs::exists(pointPath)) {
        throw std::runtime_error("memo dataset missing depth/fused_points.ply: " + projectRoot);
    }

    InputData data;
    std::ifstream frames(framesPath);
    if (!frames.is_open()) {
        throw std::runtime_error("Cannot open memo frames file: " + framesPath.string());
    }

    std::string line;
    while (std::getline(frames, line)) {
        if (line.empty()) continue;
        json frame = json::parse(line);
        if (!frame.value("acceptedKeyframe", false)) continue;
        data.cameras.push_back(cameraFromMemoFrame(frame, imageDir));
    }

    if (data.cameras.empty()) {
        throw std::runtime_error("memo dataset has no accepted keyframes: " + projectRoot);
    }

    data.points = readPly(pointPath.string());
    if (data.points.count <= 0) {
        throw std::runtime_error("memo dataset LiDAR point cloud is empty: " + pointPath.string());
    }

    autoScaleAndCenter(data);
    return data;
}
