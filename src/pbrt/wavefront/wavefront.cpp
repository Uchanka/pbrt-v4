// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/wavefront/wavefront.h>

#include <pbrt/cameras.h>
#include <pbrt/film.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/memory.h>
#include <pbrt/gpu/util.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#include <pbrt/options.h>
#include <pbrt/samplers.h>
#include <pbrt/util/math.h>
#include <pbrt/util/print.h>
#include <pbrt/util/transform.h>
#include <pbrt/wavefront/integrator.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace pbrt {

namespace {

struct BatchFrameSpec {
    int spp = 1;
    int seed = 0;
    int disableJitter = 0;
    Float jx = 0, jy = 0;
    Float fov = 45;
    Transform worldFromCamera;
    std::string outfile;
};

static std::vector<BatchFrameSpec> LoadBatchManifest(const std::string &path) {
    std::ifstream in(path);
    if (!in.is_open())
        ErrorExit("Unable to open --batch-manifest file: %s", path);

    std::vector<BatchFrameSpec> jobs;
    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
                                 line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#')
            continue;

        std::istringstream iss(line.substr(first));
        std::string mode;
        BatchFrameSpec job;
        if (!(iss >> mode >> job.spp >> job.seed >> job.disableJitter >> job.jx >>
              job.jy >> job.fov))
            ErrorExit("%s:%d: malformed batch header", path, lineNo);

        if (mode == "LOOKAT") {
            Float ex, ey, ez, lx, ly, lz, ux, uy, uz;
            if (!(iss >> ex >> ey >> ez >> lx >> ly >> lz >> ux >> uy >> uz))
                ErrorExit("%s:%d: expected 9 floats for LOOKAT", path, lineNo);
            job.worldFromCamera =
                Inverse(LookAt(Point3f(ex, ey, ez), Point3f(lx, ly, lz),
                               Vector3f(ux, uy, uz)));
        } else if (mode == "MATRIX") {
            Float m[4][4];
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    if (!(iss >> m[r][c]))
                        ErrorExit("%s:%d: expected 16 floats for MATRIX", path, lineNo);
            job.worldFromCamera = Transform(SquareMatrix<4>(m));
        } else {
            ErrorExit("%s:%d: unknown mode '%s' (expected LOOKAT or MATRIX)", path,
                      lineNo, mode);
        }

        std::getline(iss >> std::ws, job.outfile);
        if (job.outfile.empty())
            ErrorExit("%s:%d: missing output filename", path, lineNo);
        jobs.push_back(std::move(job));
    }
    return jobs;
}

}  // namespace

void RenderWavefront(BasicScene &scene) {
    WavefrontPathIntegrator *integrator = nullptr;

#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU) {
#ifdef PBRT_IS_WINDOWS
        integrator =
            new WavefrontPathIntegrator(&CUDATrackedMemoryResource::singleton, scene);
#else
        Allocator alloc(&CUDATrackedMemoryResource::singleton);
        integrator = alloc.new_object<WavefrontPathIntegrator>(
            &CUDATrackedMemoryResource::singleton, scene);
#endif
    } else
#endif  // PBRT_BUILD_GPU_RENDERER
        integrator =
            new WavefrontPathIntegrator(pstd::pmr::get_default_resource(), scene);

    if (!Options->batchManifest.empty()) {
        std::vector<BatchFrameSpec> jobs = LoadBatchManifest(Options->batchManifest);
        Printf("[MTAGR-Batch] Scene loaded on %s; executing %d persistent render jobs...\n",
               Options->useGPU ? "GPU" : "CPU", int(jobs.size()));
        std::fflush(stdout);

        Point2i fullRes = integrator->film.FullResolution();
        Float frame = Float(fullRes.x) / Float(fullRes.y);
        Bounds2f screen = (frame > 1.f)
                              ? Bounds2f(Point2f(-frame, -1.f), Point2f(frame, 1.f))
                              : Bounds2f(Point2f(-1.f, -1.f / frame),
                                         Point2f(1.f, 1.f / frame));

        for (size_t idx = 0; idx < jobs.size(); ++idx) {
            const BatchFrameSpec &job = jobs[idx];
#ifdef PBRT_BUILD_GPU_RENDERER
            if (Options->useGPU)
                GPUWait();
#endif
            Options->seed = job.seed;
            Options->disablePixelJitter = (job.disableJitter != 0);
            Options->pixelJitterX = job.jx;
            Options->pixelJitterY = job.jy;
#ifdef PBRT_BUILD_GPU_RENDERER
            if (Options->useGPU)
                CopyOptionsToGPU();
#endif

            Allocator alloc(integrator->memoryResource);
            int spp = std::max(1, job.spp);
            if (IsPowerOf2(spp))
                integrator->sampler = alloc.new_object<ZSobolSampler>(
                    spp, fullRes, RandomizeStrategy::FastOwen, job.seed);
            else
                integrator->sampler = alloc.new_object<PaddedSobolSampler>(
                    spp, RandomizeStrategy::FastOwen, job.seed);
            integrator->samplesPerPixel = spp;

            CameraTransform camXf(
                AnimatedTransform(job.worldFromCamera, 0.f, job.worldFromCamera, 1.f));
            CameraBaseParameters cbp;
            cbp.cameraTransform = camXf;
            cbp.shutterOpen = 0.f;
            cbp.shutterClose = 1.f;
            cbp.film = integrator->film;
            cbp.medium = Medium();
            integrator->camera =
                alloc.new_object<PerspectiveCamera>(cbp, job.fov, screen, 0.f, 1e6f);

            if (RGBFilm *rf = integrator->film.CastOrNullptr<RGBFilm>()) {
                rf->SetFilename(job.outfile);
            } else if (GBufferFilm *gb = integrator->film.CastOrNullptr<GBufferFilm>()) {
                gb->SetFilename(job.outfile);
                gb->UpdateOutputFromRender(camXf.RenderFromCamera());
            } else if (SpectralFilm *sf = integrator->film.CastOrNullptr<SpectralFilm>()) {
                sf->SetFilename(job.outfile);
            }

            integrator->ResetFilmPixels();
#ifdef PBRT_BUILD_GPU_RENDERER
            if (Options->useGPU)
                GPUWait();
#endif

            Float seconds = integrator->Render();

#ifdef PBRT_BUILD_GPU_RENDERER
            if (Options->useGPU) {
                std::vector<GPULogItem> logs = ReadGPULogs();
                for (const auto &item : logs)
                    Log(item.level, item.file, item.line, item.message);
            }
#endif
            ImageMetadata metadata;
            integrator->camera.InitMetadata(&metadata);
            metadata.renderTimeSeconds = seconds;
            metadata.samplesPerPixel = spp;
            integrator->film.WriteImage(metadata);

            Printf("[MTAGR-Batch %d/%d] %.3f s (spp=%d, jit=(%.4f,%.4f)) -> %s\n",
                   int(idx + 1), int(jobs.size()), seconds, spp, job.jx, job.jy,
                   job.outfile);
            std::fflush(stdout);
        }
        return;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Render!
    Float seconds = integrator->Render();

    LOG_VERBOSE("Total rendering time: %.3f s", seconds);

    if (Options->printStatistics) {
#ifdef PBRT_BUILD_GPU_RENDERER
        if (Options->useGPU)
            ReportKernelStats();
#endif  // PBRT_BUILD_GPU_RENDERER

        Printf("Wavefront integrator statistics:\n");
        Printf("%s\n", integrator->stats->Print());
    }

#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU) {
        std::vector<GPULogItem> logs = ReadGPULogs();
        for (const auto &item : logs)
            Log(item.level, item.file, item.line, item.message);
    }
#endif  // PBRT_BUILD_GPU_RENDERER

    ImageMetadata metadata;
    integrator->camera.InitMetadata(&metadata);
    metadata.renderTimeSeconds = seconds;
    metadata.samplesPerPixel = integrator->sampler.SamplesPerPixel();
    integrator->film.WriteImage(metadata);
}

}  // namespace pbrt
