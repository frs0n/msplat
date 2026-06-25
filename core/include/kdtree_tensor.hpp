#ifndef KDTREE_TENSOR_H
#define KDTREE_TENSOR_H

#include <nanoflann.hpp>
#include <algorithm>
#include <array>
#include <vector>
#include <cmath>

// nanoflann adapter for a flat float array of 3D points.
// Used to compute per-point initial scales via KNN.
struct PointsTensor {
    const float *data;
    int64_t count;

    PointsTensor(const float *data, int64_t count) : data(data), count(count) {}

    // nanoflann interface
    size_t kdtree_get_point_count() const { return (size_t)count; }
    float kdtree_get_pt(size_t idx, size_t dim) const { return data[idx * 3 + dim]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }

    using KdTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointsTensor>,
        PointsTensor, 3, size_t>;

    struct SurfaceInit {
        std::vector<float> tangentScales;
        std::vector<float> normals;
    };

    // Compute mean distance to k nearest neighbors for each point.
    std::vector<float> scales(int k = 4) const {
        KdTree index(3, *this, {10});

        std::vector<float> result(count);
        int actualK = std::max(1, std::min(k, static_cast<int>(count)));
        std::vector<size_t> indices(actualK);
        std::vector<float> dists(actualK);

        for (int64_t i = 0; i < count; i++) {
            index.knnSearch(&data[i * 3], actualK, indices.data(), dists.data());
            float sum = 0;
            for (int j = 1; j < actualK; j++) sum += std::sqrt(dists[j]);
            result[i] = actualK > 1 ? sum / (actualK - 1) : 1.0e-4f;
        }
        return result;
    }

    SurfaceInit surfaceInit(int normalK = 16, int scaleK = 3) const {
        KdTree index(3, *this, {10});

        SurfaceInit result;
        result.tangentScales.resize(count);
        result.normals.resize(count * 3);

        int queryK = std::max(1, std::min(std::max(normalK, scaleK + 1), static_cast<int>(count)));
        std::vector<size_t> indices(queryK);
        std::vector<float> dists(queryK);

        for (int64_t i = 0; i < count; i++) {
            index.knnSearch(&data[i * 3], queryK, indices.data(), dists.data());

            int scaleSamples = std::min(scaleK, queryK - 1);
            float scaleSum = 0.0f;
            for (int j = 1; j <= scaleSamples; j++) scaleSum += std::sqrt(dists[j]);
            result.tangentScales[i] = scaleSamples > 0
                ? std::max(scaleSum / scaleSamples, 1.0e-6f)
                : 1.0e-4f;

            int normalSamples = std::min(normalK, queryK - 1);
            if (normalSamples < 3) {
                result.normals[i * 3 + 0] = 0.0f;
                result.normals[i * 3 + 1] = 0.0f;
                result.normals[i * 3 + 2] = 1.0f;
                continue;
            }

            std::array<float, 3> centroid{0.0f, 0.0f, 0.0f};
            for (int j = 1; j <= normalSamples; j++) {
                size_t idx = indices[j];
                centroid[0] += data[idx * 3 + 0];
                centroid[1] += data[idx * 3 + 1];
                centroid[2] += data[idx * 3 + 2];
            }
            float invN = 1.0f / normalSamples;
            centroid[0] *= invN; centroid[1] *= invN; centroid[2] *= invN;

            float cov[3][3] = {};
            for (int j = 1; j <= normalSamples; j++) {
                size_t idx = indices[j];
                float v[3] = {
                    data[idx * 3 + 0] - centroid[0],
                    data[idx * 3 + 1] - centroid[1],
                    data[idx * 3 + 2] - centroid[2]
                };
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        cov[r][c] += v[r] * v[c] * invN;
            }

            auto n = smallestEigenvectorSymmetric(cov);
            result.normals[i * 3 + 0] = n[0];
            result.normals[i * 3 + 1] = n[1];
            result.normals[i * 3 + 2] = n[2];
        }

        return result;
    }

private:
    static std::array<float, 3> smallestEigenvectorSymmetric(float a[3][3]) {
        float v[3][3] = {
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f}
        };

        for (int iter = 0; iter < 12; iter++) {
            int p = 0, q = 1;
            float maxOff = std::fabs(a[0][1]);
            if (std::fabs(a[0][2]) > maxOff) { p = 0; q = 2; maxOff = std::fabs(a[0][2]); }
            if (std::fabs(a[1][2]) > maxOff) { p = 1; q = 2; maxOff = std::fabs(a[1][2]); }
            if (maxOff < 1.0e-12f) break;

            float tau = (a[q][q] - a[p][p]) / (2.0f * a[p][q]);
            float t = (tau >= 0.0f ? 1.0f : -1.0f) / (std::fabs(tau) + std::sqrt(1.0f + tau * tau));
            float c = 1.0f / std::sqrt(1.0f + t * t);
            float s = t * c;

            float app = a[p][p];
            float aqq = a[q][q];
            float apq = a[p][q];
            a[p][p] = app - t * apq;
            a[q][q] = aqq + t * apq;
            a[p][q] = a[q][p] = 0.0f;

            for (int k = 0; k < 3; k++) {
                if (k == p || k == q) continue;
                float akp = a[k][p];
                float akq = a[k][q];
                a[k][p] = a[p][k] = c * akp - s * akq;
                a[k][q] = a[q][k] = s * akp + c * akq;
            }

            for (int k = 0; k < 3; k++) {
                float vkp = v[k][p];
                float vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }

        int minIdx = 0;
        if (a[1][1] < a[minIdx][minIdx]) minIdx = 1;
        if (a[2][2] < a[minIdx][minIdx]) minIdx = 2;

        std::array<float, 3> n{v[0][minIdx], v[1][minIdx], v[2][minIdx]};
        float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (!std::isfinite(len) || len <= 1.0e-12f) return {0.0f, 0.0f, 1.0f};
        n[0] /= len; n[1] /= len; n[2] /= len;
        return n;
    }
};

#endif
