#include "loaders.hpp"
#include "msplat.hpp"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>

static const double C0 = 0.28209479177387814;
static constexpr float kMaxSplatScale = 1.0e6f;

static float sigmoid(float x) {
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(x);
    return z / (1.0f + z);
}

static bool validSplatGaussian(const GaussianParams &p, int64_t i,
                               const float *mp, const float *sp, const float *qp,
                               const float *dp, const float *op) {
    for (int j = 0; j < 3; j++) {
        float m = p.keepCrs ? (mp[i*3+j] / p.scale + p.translation[j]) : mp[i*3+j];
        if (!std::isfinite(m)) return false;

        float rawScale = sp[i*3+j];
        if (!std::isfinite(rawScale)) return false;
        float scale = std::exp(rawScale);
        if (p.keepCrs) scale /= p.scale;
        if (!std::isfinite(scale) || scale <= 0.0f || scale > kMaxSplatScale) return false;

        if (!std::isfinite(dp[i*3+j])) return false;
    }

    float alpha = sigmoid(op[i]);
    if (!std::isfinite(op[i]) || !std::isfinite(alpha)) return false;

    float quatNormSq = 0.0f;
    for (int j = 0; j < 4; j++) {
        float q = qp[i*4+j];
        if (!std::isfinite(q)) return false;
        quatNormSq += q * q;
    }
    if (!std::isfinite(quatNormSq) || quatNormSq <= 0.0f) return false;

    return true;
}

void saveGaussianPly(const std::string &path, GaussianParams &p, int step) {
    msplat_gpu_sync();

    std::ofstream o(path, std::ios::binary);
    int64_t N = p.means.size(0);
    int numDc = (int)p.featuresDc.size(1);
    int frBases = (int)p.featuresRest.size(-2);
    int numFr = frBases * 3;

    o << "ply\nformat binary_little_endian 1.0\n";
    o << "comment msplat v" << step << "\n";
    o << "element vertex " << N << "\n";
    o << "property float x\nproperty float y\nproperty float z\n";
    o << "property float nx\nproperty float ny\nproperty float nz\n";
    for (int i = 0; i < numDc; i++) o << "property float f_dc_" << i << "\n";
    for (int i = 0; i < numFr; i++) o << "property float f_rest_" << i << "\n";
    o << "property float opacity\n";
    o << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n";
    o << "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n";
    o << "end_header\n";

    int floatsPerRow = 3 + 3 + numDc + numFr + 1 + 3 + 4;
    std::vector<float> row(floatsPerRow);
    const float *mp = p.means.data<float>(), *sp = p.scales.data<float>(), *qp = p.quats.data<float>();
    const float *dp = p.featuresDc.data<float>(), *op = p.opacities.data<float>();
    const float *frp = p.featuresRest.data<float>();

    for (int64_t i = 0; i < N; i++) {
        int c = 0;
        for (int j = 0; j < 3; j++)
            row[c++] = p.keepCrs ? (mp[i*3+j] / p.scale + p.translation[j]) : mp[i*3+j];
        row[c++] = 0; row[c++] = 0; row[c++] = 0; // normals
        for (int j = 0; j < numDc; j++) row[c++] = dp[i*numDc+j];
        // Transpose [frBases, 3] → [3, frBases] for PLY convention
        for (int ch = 0; ch < 3; ch++)
            for (int b = 0; b < frBases; b++)
                row[c++] = frp[i*frBases*3 + b*3 + ch];
        row[c++] = op[i];
        for (int j = 0; j < 3; j++)
            row[c++] = p.keepCrs ? std::log(std::exp(sp[i*3+j]) / p.scale) : sp[i*3+j];
        for (int j = 0; j < 4; j++) row[c++] = qp[i*4+j];

        o.write(reinterpret_cast<const char*>(row.data()), floatsPerRow * sizeof(float));
    }
}

void saveGaussianSplat(const std::string &path, GaussianParams &p) {
    msplat_gpu_sync();

    std::ofstream o(path, std::ios::binary);
    int64_t N = p.means.size(0);
    const float *mp = p.means.data<float>(), *sp = p.scales.data<float>(), *qp = p.quats.data<float>();
    const float *dp = p.featuresDc.data<float>(), *op = p.opacities.data<float>();

    std::vector<size_t> idx;
    idx.reserve(N);
    for (int64_t i = 0; i < N; i++) {
        if (validSplatGaussian(p, i, mp, sp, qp, dp, op)) idx.push_back((size_t)i);
    }

    // Sort by size/opacity (largest first)
    std::vector<float> order(N);
    for (size_t i : idx) {
        float s = std::exp(sp[i*3]) + std::exp(sp[i*3+1]) + std::exp(sp[i*3+2]);
        if (p.keepCrs) s /= p.scale;
        order[i] = s * sigmoid(op[i]);
    }
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return order[a] > order[b]; });

    if ((int64_t)idx.size() != N) {
        std::cerr << "Filtered " << (N - (int64_t)idx.size())
                  << " abnormal gaussians while exporting " << path << std::endl;
    }

    for (size_t ii = 0; ii < idx.size(); ii++) {
        size_t i = idx[ii];
        float m[3];
        for (int j = 0; j < 3; j++) m[j] = p.keepCrs ? (mp[i*3+j] / p.scale + p.translation[j]) : mp[i*3+j];
        o.write(reinterpret_cast<const char*>(m), 12);

        float sc[3];
        for (int j = 0; j < 3; j++) sc[j] = p.keepCrs ? (std::exp(sp[i*3+j]) / p.scale) : std::exp(sp[i*3+j]);
        o.write(reinterpret_cast<const char*>(sc), 12);

        uint8_t rgb[3];
        for (int j = 0; j < 3; j++) rgb[j] = (uint8_t)std::clamp(((double)dp[i*3+j] * C0 + 0.5) * 255.0, 0.0, 255.0);
        o.write(reinterpret_cast<const char*>(rgb), 3);

        float sig = sigmoid(op[i]);
        uint8_t a = (uint8_t)std::clamp(sig * 255.0f, 0.0f, 255.0f);
        o.write(reinterpret_cast<const char*>(&a), 1);

        uint8_t q[4];
        for (int j = 0; j < 4; j++) q[j] = (uint8_t)std::clamp(qp[i*4+j] * 128.0f + 128.0f, 0.0f, 255.0f);
        o.write(reinterpret_cast<const char*>(q), 4);
    }
}

LoadedGaussians loadGaussianPly(const std::string &path, float scale, const float translation[3], bool keepCrs) {
    msplat_gpu_sync();

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open PLY file: " + path);

    // Parse header
    std::string line;
    int numPoints = 0, step = 0;
    int numDc = 0, numFr = 0;

    std::getline(f, line); // "ply"
    if (line.find("ply") == std::string::npos) throw std::runtime_error("Not a PLY file: " + path);
    std::getline(f, line); // "format binary_little_endian 1.0"

    while (std::getline(f, line)) {
        if (line == "end_header") break;

        const std::string iterPrefix = "comment Generated by msplat at iteration ";
        if (line.rfind(iterPrefix, 0) == 0)
            step = std::stoi(line.substr(iterPrefix.length()));

        const std::string vertexPrefix = "element vertex ";
        if (line.rfind(vertexPrefix, 0) == 0)
            numPoints = std::stoi(line.substr(vertexPrefix.length()));

        if (line.rfind("property float f_dc_", 0) == 0) numDc++;
        if (line.rfind("property float f_rest_", 0) == 0) numFr++;
    }

    if (numPoints == 0) throw std::runtime_error("PLY has no vertices");
    int frBases = numFr / 3;

    // Read binary data: xyz(3) + normals(3) + f_dc(numDc) + f_rest(numFr) + opacity(1) + scale(3) + rot(4)
    std::vector<float> meansRaw(numPoints * 3);
    std::vector<float> dcRaw(numPoints * numDc);
    std::vector<float> frRaw(numPoints * numFr);
    std::vector<float> opRaw(numPoints);
    std::vector<float> scRaw(numPoints * 3);
    std::vector<float> qtRaw(numPoints * 4);
    float normals[3];

    for (int i = 0; i < numPoints; i++) {
        f.read(reinterpret_cast<char*>(&meansRaw[i*3]), 12);
        f.read(reinterpret_cast<char*>(normals), 12);
        f.read(reinterpret_cast<char*>(&dcRaw[i*numDc]), numDc * 4);
        f.read(reinterpret_cast<char*>(&frRaw[i*numFr]), numFr * 4);
        f.read(reinterpret_cast<char*>(&opRaw[i]), 4);
        f.read(reinterpret_cast<char*>(&scRaw[i*3]), 12);
        f.read(reinterpret_cast<char*>(&qtRaw[i*4]), 16);
    }

    // CRS transform
    if (keepCrs) {
        for (int i = 0; i < numPoints; i++)
            for (int j = 0; j < 3; j++)
                meansRaw[i*3+j] = (meansRaw[i*3+j] - translation[j]) * scale;
        for (int i = 0; i < numPoints * 3; i++)
            scRaw[i] = std::log(scale * std::exp(scRaw[i]));
    }

    // Upload to GPU
    LoadedGaussians g;
    g.step = step;
    auto upload = [](std::vector<int64_t> shape, const float *src, size_t bytes) {
        MTensor t = gpu_empty(shape, DType::Float32);
        memcpy(t.data_ptr(), src, bytes);
        return t;
    };
    g.means = upload({(int64_t)numPoints, 3}, meansRaw.data(), meansRaw.size() * 4);
    g.featuresDc = upload({(int64_t)numPoints, (int64_t)numDc}, dcRaw.data(), dcRaw.size() * 4);
    g.opacities = upload({(int64_t)numPoints, 1}, opRaw.data(), opRaw.size() * 4);
    g.scales = upload({(int64_t)numPoints, 3}, scRaw.data(), scRaw.size() * 4);
    g.quats = upload({(int64_t)numPoints, 4}, qtRaw.data(), qtRaw.size() * 4);

    // Transpose featuresRest: PLY [N, 3, frBases] → internal [N, frBases, 3]
    g.featuresRest = gpu_empty({(int64_t)numPoints, (int64_t)frBases, 3}, DType::Float32);
    float *frOut = g.featuresRest.data<float>();
    for (int i = 0; i < numPoints; i++)
        for (int ch = 0; ch < 3; ch++)
            for (int b = 0; b < frBases; b++)
                frOut[i*frBases*3 + b*3 + ch] = frRaw[i*numFr + ch*frBases + b];

    // numPoints and step available via returned struct
    return g;
}
