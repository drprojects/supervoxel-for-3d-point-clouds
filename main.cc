#include <chrono>
#include <cstdlib> // for atof
#include <filesystem>
#include "codelibrary/base/log.h"
#include "codelibrary/geometry/io/xyz_io.h"
#include "codelibrary/geometry/point_cloud/pca_estimate_normals.h"
#include "codelibrary/geometry/point_cloud/supervoxel_segmentation.h"
#include "codelibrary/geometry/util/distance_3d.h"
#include "codelibrary/util/tree/kd_tree.h"

#include "vccs_knn_supervoxel.h"
#include "vccs_supervoxel.h"

struct PointWithNormal : cl::RPoint3D {
    PointWithNormal() {}
    cl::RVector3D normal;
};

class VCCSMetric {
public:
    explicit VCCSMetric(double resolution)
        : resolution_(resolution) {}
    double operator() (const PointWithNormal& p1,
                       const PointWithNormal& p2) const {
        return 1.0 - std::fabs(p1.normal * p2.normal) +
               cl::geometry::Distance(p1, p2) / resolution_ * 0.4;
    }
private:
    double resolution_;
};

void WritePoints(const char* filename,
                 int n_supervoxels,
                 const cl::Array<cl::RPoint3D>& points,
                 const cl::Array<int>& labels) {
    std::ofstream ofs(filename);
    if (!ofs) {
        LOG(ERROR) << "Failed to open output file: " << filename;
        return;
    }

    // Assign a random color to each supervoxel
    std::mt19937 random;
    cl::Array<cl::RGB32Color> supervoxel_colors(n_supervoxels);
    for (int i = 0; i < n_supervoxels; ++i) {
        supervoxel_colors[i] = cl::RGB32Color(random());
    }

    // Write points with their color and supervoxel index
    for (int i = 0; i < points.size(); ++i) {
        const auto& color = supervoxel_colors[labels[i]];
        ofs << points[i].x << " "
            << points[i].y << " "
            << points[i].z << " "
            << static_cast<int>(color.red()) << " "
            << static_cast<int>(color.green()) << " "
            << static_cast<int>(color.blue()) << " "
            << labels[i] << "\n";
    }

    LOG(INFO) << "The points with supervoxel indices are written into " << filename;
}

int main(int argc, char** argv) {
    LOG_ON(INFO);
    using Clock = std::chrono::high_resolution_clock;

    // Defaults
    std::string filename = "test.xyz";
    double resolution = 1.0;
    bool save_outputs = false;

    // Parse optional arguments
    if (argc > 1) filename = argv[1];
    if (argc > 2) {
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--resolution") {
                if (i + 1 < argc) {
                    resolution = std::stof(argv[++i]);
                }
            } else if (std::string(argv[i]) == "--save") {
                save_outputs = true;
            }
        }
    }

    LOG(INFO) << "Using file: " << filename;
    LOG(INFO) << "Using resolution: " << resolution;

    std::filesystem::path input_path(filename);
    std::string stem = input_path.stem().string(); // without extension
    std::string dir  = input_path.parent_path().string();

    cl::Array<cl::RPoint3D> points;
    cl::Array<cl::RGB32Color> colors;

    // Time: Reading points
    auto t0 = Clock::now();
    LOG(INFO) << "Reading points from " << filename << "...";
    if (!cl::geometry::io::ReadXYZPoints(filename.c_str(), &points, &colors)) {
        LOG(ERROR) << "Cannot read file: " << filename;
        return 1;
    }
    auto t1 = Clock::now();
    int n_points = points.size();
    LOG(INFO) << n_points << " points are imported.";
    LOG(INFO) << "Time for reading points: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms";

    // KD-tree + normals
    auto t2 = Clock::now();
    LOG(INFO) << "Building KD tree...";
    cl::KDTree<cl::RPoint3D> kdtree;
    kdtree.SwapPoints(&points);

    const int k_neighbors = 15;
    cl::Array<cl::RVector3D> normals(n_points);
    cl::Array<cl::Array<int>> neighbors(n_points);
    cl::Array<cl::RPoint3D> neighbor_points(k_neighbors);
    for (int i = 0; i < n_points; ++i) {
        kdtree.FindKNearestNeighbors(kdtree.points()[i], k_neighbors, &neighbors[i]);
        for (int k = 0; k < k_neighbors; ++k) {
            neighbor_points[k] = kdtree.points()[neighbors[i][k]];
        }
        cl::geometry::point_cloud::PCAEstimateNormal(neighbor_points.begin(),
                                                     neighbor_points.end(),
                                                     &normals[i]);
    }
    kdtree.SwapPoints(&points);
    auto t3 = Clock::now();
    LOG(INFO) << "Time for KD-tree + KNN + normals: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " ms";

    // SupervoxelSegmentation
    auto t4 = Clock::now();
    LOG(INFO) << "Start SupervoxelSegmentation...";
    cl::Array<PointWithNormal> oriented_points(n_points);
    for (int i = 0; i < n_points; ++i) {
        oriented_points[i].x = points[i].x;
        oriented_points[i].y = points[i].y;
        oriented_points[i].z = points[i].z;
        oriented_points[i].normal = normals[i];
    }
    VCCSMetric metric(resolution);
    cl::Array<int> labels, supervoxels;
    cl::geometry::point_cloud::SupervoxelSegmentation(oriented_points,
                                                      neighbors,
                                                      resolution,
                                                      metric,
                                                      &supervoxels,
                                                      &labels);
    auto t5 = Clock::now();
    LOG(INFO) << "Time for SupervoxelSegmentation: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count()
              << " ms";
    LOG(INFO) << "Input points: " << n_points
              << ", Output superpoints: " << supervoxels.size();

    // VCCS
    auto t6 = Clock::now();
    LOG(INFO) << "Start VCCS supervoxel segmentation...";
    const double voxel_resolution = 0.03;
    VCCSSupervoxel vccs(points.begin(), points.end(),
                        voxel_resolution,
                        resolution);
    cl::Array<int> vccs_labels;
    cl::Array<VCCSSupervoxel::Supervoxel> vccs_supervoxels;
    vccs.Segment(&vccs_labels, &vccs_supervoxels);
    auto t7 = Clock::now();
    LOG(INFO) << "Time for VCCS: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6).count()
              << " ms";
    LOG(INFO) << "Input points: " << n_points
              << ", Output superpoints: " << vccs_supervoxels.size();

    // VCCS-KNN
    auto t8 = Clock::now();
    LOG(INFO) << "Start KNN variant of VCCS supervoxel segmentation...";
    kdtree.SwapPoints(&points);
    VCCSKNNSupervoxel vccs_knn(kdtree, resolution);
    cl::Array<int> vccs_knn_labels;
    cl::Array<VCCSKNNSupervoxel::Supervoxel> vccs_knn_supervoxels;
    vccs_knn.Segment(&vccs_knn_labels, &vccs_knn_supervoxels);
    kdtree.SwapPoints(&points);
    auto t9 = Clock::now();
    LOG(INFO) << "Time for VCCS-KNN: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t9 - t8).count()
              << " ms";
    LOG(INFO) << "Input points: " << n_points
              << ", Output superpoints: " << vccs_knn_supervoxels.size();

     // Saving outputs, if required
     if (save_outputs) {
        WritePoints((std::filesystem::path(dir) / (stem + "_out.xyz")).string().c_str(),
                    supervoxels.size(), points, labels);

        WritePoints((std::filesystem::path(dir) / (stem + "_out_vccs.xyz")).string().c_str(),
                    vccs_supervoxels.size(), points, vccs_labels);

        WritePoints((std::filesystem::path(dir) / (stem + "_out_vccs_knn.xyz")).string().c_str(),
                    vccs_knn_supervoxels.size(), points, vccs_knn_labels);
    }

    return 0;
}
