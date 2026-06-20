// ObjC++ implementation of the Swift-facing C++ API.
// This is the ONLY file that touches internal C++ types (Model, Camera, MTensor).

#include "msplat_api.hpp"

#include "model.hpp"
#include "input_data.hpp"
#include "msplat.hpp"
#include "ssim.hpp"

#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstdlib>
#include <unordered_map>

#include <TargetConditionals.h>
#include <mach/mach.h>

namespace msplat {

namespace {

size_t currentPhysFootprintBytes() {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t result = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
    return result == KERN_SUCCESS ? (size_t)info.phys_footprint : 0;
}

int memoryLogInterval() {
    static int interval = [] {
        if (const char* env = std::getenv("MSPLAT_MEM_LOG_EVERY")) {
            return std::atoi(env);
        }
        return 0;
    }();
    return interval;
}

double mb(size_t bytes) {
    return (double)bytes / (1024.0 * 1024.0);
}

} // namespace

// ── Dataset::Impl ───────────────────────────────────────────────────────────

struct Dataset::Impl {
    InputData data;
    std::vector<size_t> trainIndices;
    std::vector<size_t> testIndices;
    float imageDownscaleFactor = 1.0f;
    size_t maxCachedImageBytes = 0;
    uint64_t imageCacheClock = 0;

    struct CachedCamera {
        uint64_t lastUse = 0;
        size_t bytes = 0;
    };
    std::unordered_map<size_t, CachedCamera> cachedCameras;

    static size_t defaultImageCacheBudgetBytes() {
        if (const char* env = std::getenv("MSPLAT_IMAGE_CACHE_MB")) {
            int mb = std::atoi(env);
            if (mb > 0) return (size_t)mb * 1024 * 1024;
        }
#if TARGET_OS_IPHONE
        return (size_t)512 * 1024 * 1024;
#else
        return (size_t)2048 * 1024 * 1024;
#endif
    }

    Camera& trainCamera(size_t trainIndex) {
        return data.cameras[trainIndices[trainIndex]];
    }

    Camera& testCamera(size_t testIndex) {
        return data.cameras[testIndices[testIndex]];
    }

    MTensor& gpuImageForDataCamera(size_t dataCameraIndex, int downscaleFactor) {
        Camera& cam = data.cameras[dataCameraIndex];
        cam.loadImage(imageDownscaleFactor);
        MTensor& image = cam.getGPUImage(downscaleFactor);

        auto& cached = cachedCameras[dataCameraIndex];
        cached.lastUse = ++imageCacheClock;
        cached.bytes = cam.cachedImageBytes();
        evictImageCache(dataCameraIndex);
        return image;
    }

    MTensor& gpuImageForTrainCamera(size_t trainIndex, int downscaleFactor) {
        return gpuImageForDataCamera(trainIndices[trainIndex], downscaleFactor);
    }

    MTensor& gpuImageForTestCamera(size_t testIndex, int downscaleFactor) {
        return gpuImageForDataCamera(testIndices[testIndex], downscaleFactor);
    }

    void evictImageCache(size_t protectedCameraIndex) {
        while (cachedImageBytes() > maxCachedImageBytes && cachedCameras.size() > 1) {
            auto victim = cachedCameras.end();
            for (auto it = cachedCameras.begin(); it != cachedCameras.end(); ++it) {
                if (it->first == protectedCameraIndex) continue;
                if (victim == cachedCameras.end() || it->second.lastUse < victim->second.lastUse) {
                    victim = it;
                }
            }
            if (victim == cachedCameras.end()) break;

            data.cameras[victim->first].releaseImageMemory();
            cachedCameras.erase(victim);
        }
    }

    size_t cachedImageBytes() const {
        size_t bytes = 0;
        for (const auto& item : cachedCameras) bytes += item.second.bytes;
        return bytes;
    }
};

Dataset::Dataset(const std::string& path, float downscaleFactor,
                 bool evalMode, int testEvery)
    : impl(std::make_unique<Impl>())
{
    impl->data = inputDataFromX(path);
    impl->imageDownscaleFactor = downscaleFactor;
    impl->maxCachedImageBytes = Impl::defaultImageCacheBudgetBytes();

    if (evalMode) {
        for (int i = 0; i < (int)impl->data.cameras.size(); i++) {
            if (i % testEvery == 0)
                impl->testIndices.push_back(i);
            else
                impl->trainIndices.push_back(i);
        }
    } else {
        impl->trainIndices.reserve(impl->data.cameras.size());
        for (size_t i = 0; i < impl->data.cameras.size(); i++)
            impl->trainIndices.push_back(i);
    }
}

Dataset::~Dataset() = default;
Dataset::Dataset(Dataset&&) noexcept = default;
Dataset& Dataset::operator=(Dataset&&) noexcept = default;

int Dataset::numTrain() const { return (int)impl->trainIndices.size(); }
int Dataset::numTest() const { return (int)impl->testIndices.size(); }
void Dataset::cameraPose(int index, float camToWorld[16]) const {
    if (index >= 0 && index < (int)impl->trainIndices.size())
        memcpy(camToWorld, impl->data.cameras[impl->trainIndices[index]].camToWorld, 16 * sizeof(float));
}
void* Dataset::_handle() const { return impl.get(); }

// ── Trainer::Impl ───────────────────────────────────────────────────────────

struct Trainer::Impl {
    std::unique_ptr<Model> model;
    Config config;
    Dataset::Impl* ds = nullptr;
    int currentStep = 0;

    // Camera iteration
    std::vector<size_t> camIndices;
    size_t camIterPos = 0;
    std::mt19937 rng{42};

    void shuffleCameras() {
        std::shuffle(camIndices.begin(), camIndices.end(), rng);
        camIterPos = 0;
    }

    size_t nextCamera() {
        if (camIterPos >= camIndices.size()) shuffleCameras();
        return camIndices[camIterPos++];
    }
};

Trainer::Trainer(Dataset& dataset, const Config& config)
    : impl(std::make_unique<Impl>())
{
    impl->config = config;
    impl->ds = static_cast<Dataset::Impl*>(dataset._handle());

    impl->model = std::make_unique<Model>(
        impl->ds->data,
        (int)impl->ds->trainIndices.size(),
        config.numDownscales, config.resolutionSchedule,
        config.shDegree, config.shDegreeInterval,
        config.refineEvery, config.warmupLength, config.resetAlphaEvery,
        config.densifyGradThresh, config.densifySizeThresh,
        config.stopScreenSizeAt, config.splitScreenSize,
        config.iterations, config.keepCrs,
        config.bgColor
    );

    impl->camIndices.resize(impl->ds->trainIndices.size());
    std::iota(impl->camIndices.begin(), impl->camIndices.end(), 0);
    impl->shuffleCameras();
}

Trainer::~Trainer() = default;

Stats Trainer::step() {
    impl->currentStep++;
    size_t camIdx = impl->nextCamera();
    Camera& cam = impl->ds->trainCamera(camIdx);

    int ds = impl->model->getDownscaleFactor(impl->currentStep);
    MTensor& gt = impl->ds->gpuImageForTrainCamera(camIdx, ds);

    auto t0 = std::chrono::high_resolution_clock::now();

    impl->model->fullIteration(cam, impl->currentStep, gt, impl->config.ssimWeight);
    impl->model->schedulersStep(impl->currentStep);
    impl->model->afterTrain(impl->currentStep);
    msplat_commit();

    int logEvery = memoryLogInterval();
    if (logEvery > 0 && (impl->currentStep == 1 || impl->currentStep % logEvery == 0)) {
        size_t processBytes = currentPhysFootprintBytes();
        size_t imageBytes = impl->ds->cachedImageBytes();
        size_t modelBytes = impl->model->estimatedGpuBytes();
        size_t tempBytes = msplat_cached_tensor_bytes();
        size_t accountedBytes = imageBytes + modelBytes + tempBytes;
        fprintf(stderr,
                "MSPLAT_MEM step=%d splats=%d phys=%.1fMB accounted=%.1fMB "
                "model=%.1fMB temp=%.1fMB images=%.1fMB imageBudget=%.1fMB\n",
                impl->currentStep,
                (int)impl->model->means.size(0),
                mb(processBytes), mb(accountedBytes),
                mb(modelBytes), mb(tempBytes), mb(imageBytes),
                mb(impl->ds->maxCachedImageBytes));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;

    Stats s;
    s.iteration = impl->currentStep;
    s.splatCount = (int)impl->model->means.size(0);
    s.msPerStep = ms;
    return s;
}

void Trainer::train(int callbackEvery) {
    while (impl->currentStep < impl->config.iterations) {
        step();
        // Note: callbacks handled at the Swift level via polling iteration()
        // to keep the C++ API free of function pointer complexity
    }
}

EvalMetrics Trainer::evaluate() {
    auto& testIndices = impl->ds->testIndices;
    if (testIndices.empty())
        return {};

    double sumPsnr = 0, sumSsim = 0, sumL1 = 0;
    int n = (int)testIndices.size();

    for (int i = 0; i < n; i++) {
        Camera& cam = impl->ds->testCamera(i);
        MTensor rgb = impl->model->render(cam, impl->config.iterations);
        msplat_gpu_sync();
        MTensor rgbCpu = rgb.cpu();
        int dsf = impl->model->getDownscaleFactor(impl->config.iterations);
        MTensor gtCpu = impl->ds->gpuImageForTestCamera(i, dsf).cpu();

        sumPsnr += psnr(rgbCpu, gtCpu);
        sumSsim += ssim_eval(rgbCpu, gtCpu);
        sumL1 += l1_loss(rgbCpu, gtCpu);
    }

    EvalMetrics m;
    m.psnr = (float)(sumPsnr / n);
    m.ssim = (float)(sumSsim / n);
    m.l1 = (float)(sumL1 / n);
    m.numTest = n;
    m.numGaussians = (int)impl->model->means.size(0);
    return m;
}

PixelBuffer Trainer::render(int cameraIndex, bool useTest) {
    auto& indices = useTest ? impl->ds->testIndices : impl->ds->trainIndices;
    if (cameraIndex < 0 || cameraIndex >= (int)indices.size())
        return {};

    Camera& cam = impl->ds->data.cameras[indices[cameraIndex]];
    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();
    MTensor rgbCpu = rgb.cpu();

    int h = (int)rgbCpu.size(0);
    int w = (int)rgbCpu.size(1);
    // Use malloc so callers can free() — PixelBuffer destructor handles both
    float* buf = (float*)malloc(h * w * 3 * sizeof(float));
    memcpy(buf, rgbCpu.data_ptr(), h * w * 3 * sizeof(float));

    return PixelBuffer(buf, w, h);
}

PixelBuffer Trainer::renderFromPose(const float camToWorld[16], int refCameraIndex) {
    auto& indices = impl->ds->trainIndices;
    if (refCameraIndex < 0 || refCameraIndex >= (int)indices.size())
        return {};

    Camera& ref = impl->ds->data.cameras[indices[refCameraIndex]];
    Camera cam;
    cam.width = ref.width; cam.height = ref.height;
    cam.fx = ref.fx; cam.fy = ref.fy; cam.cx = ref.cx; cam.cy = ref.cy;
    memcpy(cam.camToWorld, camToWorld, 16 * sizeof(float));

    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();
    MTensor rgbCpu = rgb.cpu();

    int h = (int)rgbCpu.size(0);
    int w = (int)rgbCpu.size(1);
    float* buf = (float*)malloc(h * w * 3 * sizeof(float));
    memcpy(buf, rgbCpu.data_ptr(), h * w * 3 * sizeof(float));
    return PixelBuffer(buf, w, h);
}

void Trainer::renderFromPoseToBuffer(const float camToWorld[16], int refCameraIndex,
                                  uint8_t* outRGBA, int* outWidth, int* outHeight) {
    auto& indices = impl->ds->trainIndices;
    if (refCameraIndex < 0 || refCameraIndex >= (int)indices.size()) {
        *outWidth = 0; *outHeight = 0; return;
    }

    Camera& ref = impl->ds->data.cameras[indices[refCameraIndex]];
    Camera cam;
    cam.width = ref.width; cam.height = ref.height;
    cam.fx = ref.fx; cam.fy = ref.fy; cam.cx = ref.cx; cam.cy = ref.cy;
    memcpy(cam.camToWorld, camToWorld, 16 * sizeof(float));

    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();

    int h = (int)rgb.size(0), w = (int)rgb.size(1);
    *outWidth = w;
    *outHeight = h;
    if (!outRGBA) return;

    // Read directly from GPU tensor (unified memory on Apple Silicon)
    const float* src = (const float*)rgb.data_ptr();
    int n = w * h;
    for (int i = 0; i < n; i++) {
        outRGBA[i * 4]     = (uint8_t)(fminf(fmaxf(src[i*3],   0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 1] = (uint8_t)(fminf(fmaxf(src[i*3+1], 0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 2] = (uint8_t)(fminf(fmaxf(src[i*3+2], 0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 3] = 255;
    }
}

void Trainer::exportPly(const std::string& path) {
    impl->model->savePly(path, impl->currentStep);
}

void Trainer::exportSplat(const std::string& path) {
    impl->model->saveSplat(path);
}

void Trainer::saveCheckpoint(const std::string& path) {
    impl->model->saveCheckpoint(path, impl->currentStep);
}

int Trainer::loadCheckpoint(const std::string& path) {
    impl->currentStep = impl->model->loadCheckpoint(path);
    // Re-shuffle cameras for resumed training
    impl->shuffleCameras();
    return impl->currentStep;
}

int Trainer::splatCount() const {
    return (int)impl->model->means.size(0);
}

int Trainer::iteration() const {
    return impl->currentStep;
}

// ── Lifecycle ───────────────────────────────────────────────────────────────

void sync() { msplat_gpu_sync(); }
void cleanup() { cleanup_msplat_metal(); }

} // namespace msplat

// ── C API (for Swift interop) ───────────────────────────────────────────────

#include "msplat_c_api.h"

static msplat::Config configFromC(MsplatConfig c) {
    msplat::Config cfg;
    cfg.iterations = c.iterations;
    cfg.shDegree = c.shDegree;
    cfg.shDegreeInterval = c.shDegreeInterval;
    cfg.ssimWeight = c.ssimWeight;
    cfg.numDownscales = c.numDownscales;
    cfg.resolutionSchedule = c.resolutionSchedule;
    cfg.refineEvery = c.refineEvery;
    cfg.warmupLength = c.warmupLength;
    cfg.resetAlphaEvery = c.resetAlphaEvery;
    cfg.densifyGradThresh = c.densifyGradThresh;
    cfg.densifySizeThresh = c.densifySizeThresh;
    cfg.stopScreenSizeAt = c.stopScreenSizeAt;
    cfg.splitScreenSize = c.splitScreenSize;
    cfg.keepCrs = c.keepCrs;
    cfg.downscaleFactor = c.downscaleFactor;
    memcpy(cfg.bgColor, c.bgColor, sizeof(cfg.bgColor));
    return cfg;
}

MsplatDataset msplat_dataset_create(const char* path, float downscaleFactor,
                                     bool evalMode, int testEvery) {
    auto* ds = new msplat::Dataset(std::string(path), downscaleFactor, evalMode, testEvery);
    return static_cast<MsplatDataset>(ds);
}

void msplat_dataset_destroy(MsplatDataset ds) {
    delete static_cast<msplat::Dataset*>(ds);
}

int msplat_dataset_num_train(MsplatDataset ds) {
    return static_cast<msplat::Dataset*>(ds)->numTrain();
}

int msplat_dataset_num_test(MsplatDataset ds) {
    return static_cast<msplat::Dataset*>(ds)->numTest();
}

void msplat_dataset_camera_pose(MsplatDataset ds, int cameraIndex, float camToWorld[16]) {
    static_cast<msplat::Dataset*>(ds)->cameraPose(cameraIndex, camToWorld);
}

MsplatTrainer msplat_trainer_create(MsplatDataset ds, MsplatConfig config) {
    auto* dataset = static_cast<msplat::Dataset*>(ds);
    auto cfg = configFromC(config);
    auto* trainer = new msplat::Trainer(*dataset, cfg);
    return static_cast<MsplatTrainer>(trainer);
}

void msplat_trainer_destroy(MsplatTrainer t) {
    delete static_cast<msplat::Trainer*>(t);
}

MsplatStats msplat_trainer_step(MsplatTrainer t) {
    auto stats = static_cast<msplat::Trainer*>(t)->step();
    return MsplatStats{stats.iteration, stats.splatCount, stats.msPerStep};
}

void msplat_trainer_train(MsplatTrainer t) {
    static_cast<msplat::Trainer*>(t)->train(0);
}

MsplatEvalMetrics msplat_trainer_evaluate(MsplatTrainer t) {
    auto m = static_cast<msplat::Trainer*>(t)->evaluate();
    return MsplatEvalMetrics{m.psnr, m.ssim, m.l1, m.numTest, m.numGaussians};
}

MsplatPixelBuffer msplat_trainer_render(MsplatTrainer t, int cameraIndex, bool useTest) {
    auto buf = static_cast<msplat::Trainer*>(t)->render(cameraIndex, useTest);
    MsplatPixelBuffer result{buf.data, buf.width, buf.height};
    buf.data = nullptr; // Transfer ownership to caller
    return result;
}

MsplatPixelBuffer msplat_trainer_render_pose(MsplatTrainer t, const float camToWorld[16], int refCameraIndex) {
    auto buf = static_cast<msplat::Trainer*>(t)->renderFromPose(camToWorld, refCameraIndex);
    MsplatPixelBuffer result{buf.data, buf.width, buf.height};
    buf.data = nullptr;
    return result;
}

void msplat_trainer_render_pose_to_buffer(MsplatTrainer t, const float camToWorld[16],
                                      int refCameraIndex, uint8_t* outRGBA,
                                      int* outWidth, int* outHeight) {
    static_cast<msplat::Trainer*>(t)->renderFromPoseToBuffer(
        camToWorld, refCameraIndex, outRGBA, outWidth, outHeight);
}

void msplat_trainer_export_ply(MsplatTrainer t, const char* path) {
    static_cast<msplat::Trainer*>(t)->exportPly(std::string(path));
}

void msplat_trainer_export_splat(MsplatTrainer t, const char* path) {
    static_cast<msplat::Trainer*>(t)->exportSplat(std::string(path));
}

void msplat_trainer_save_checkpoint(MsplatTrainer t, const char* path) {
    static_cast<msplat::Trainer*>(t)->saveCheckpoint(std::string(path));
}

int msplat_trainer_load_checkpoint(MsplatTrainer t, const char* path) {
    return static_cast<msplat::Trainer*>(t)->loadCheckpoint(std::string(path));
}

int msplat_trainer_splat_count(MsplatTrainer t) {
    return static_cast<msplat::Trainer*>(t)->splatCount();
}

int msplat_trainer_iteration(MsplatTrainer t) {
    return static_cast<msplat::Trainer*>(t)->iteration();
}

void msplat_sync(void) { msplat::sync(); }
void msplat_cleanup(void) { msplat::cleanup(); }
