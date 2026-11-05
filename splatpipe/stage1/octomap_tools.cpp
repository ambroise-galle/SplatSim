// Outil octomap minimal reproduisant le protocole du benchmark de reconstruction Oxford Spires
// (oxspires_tools/cpp/src/pcd2bt.cpp et filter_pcd_with_bt.cpp, commit b456e1e) sans PCL :
//
//   octomap_tools build <points_f32.bin> <ox> <oy> <oz> <resolution> <out.bt>
//       Un nuage (float32 x,y,z contigus, repère world) + son point de vue → OcTree(resolution),
//       tree.insertPointCloud(cloud, origin, maxrange=-1, lazy_eval=true, discretize=true), tree.writeBinary(out).
//       Identique à processPCDFolder pour un seul nuage (le benchmark insère les nuages l'un après l'autre dans
//       le même arbre ; l'ensemble des voxels connus est l'union, donc identique à l'union des arbres par station).
//
//   octomap_tools query <tree.bt> <points_f32.bin> <mask_u8.out>
//       Pour chaque point : tree.search(point) != NULL → 1 (connu : libre ou occupé), sinon 0 (inconnu).
//       Identique à removeUnknownPoints (qui garde les points connus). Parallélisé par OpenMP (lecture seule).
#include <octomap/octomap.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static std::vector<float> read_f32(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "cannot open " << path << std::endl;
    std::exit(2);
  }
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<float> buf(n / sizeof(float));
  f.read(reinterpret_cast<char *>(buf.data()), buf.size() * sizeof(float));
  return buf;
}

int main(int argc, char **argv) {
  using clk = std::chrono::steady_clock;
  if (argc < 2) {
    std::cerr << "usage: build|query ..." << std::endl;
    return 1;
  }
  std::string mode = argv[1];
  auto t0 = clk::now();
  if (mode == "build" && argc == 8) {
    std::vector<float> pts = read_f32(argv[2]);
    octomap::point3d origin(std::atof(argv[3]), std::atof(argv[4]), std::atof(argv[5]));
    double res = std::atof(argv[6]);
    octomap::Pointcloud cloud;
    cloud.reserve(pts.size() / 3);
    for (size_t i = 0; i + 2 < pts.size(); i += 3) cloud.push_back(pts[i], pts[i + 1], pts[i + 2]);
    octomap::OcTree tree(res);
    tree.insertPointCloud(cloud, origin, -1, true, true);
    auto t1 = clk::now();
    size_t leafs = tree.getNumLeafNodes();
    tree.writeBinary(argv[7]);
    auto t2 = clk::now();
    std::printf("{\"n_points\": %zu, \"leafs_before_prune\": %zu, \"leafs_after_prune\": %zu, \"insert_s\": %.1f, \"write_s\": %.1f}\n",
                cloud.size(), leafs, tree.getNumLeafNodes(),
                std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count());
    return 0;
  }
  if (mode == "query" && argc == 5) {
    octomap::OcTree tree(argv[2]);
    if (tree.size() == 0) {
      std::cerr << "empty tree " << argv[2] << std::endl;
      return 3;
    }
    std::vector<float> pts = read_f32(argv[3]);
    size_t n = pts.size() / 3;
    std::vector<unsigned char> mask(n, 0);
    auto t1 = clk::now();
    long long known = 0;
    #pragma omp parallel for schedule(static) reduction(+:known)
    for (long long i = 0; i < (long long)n; ++i) {
      octomap::OcTreeNode *node = tree.search(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);
      if (node) {
        mask[i] = 1;
        known += 1;
      }
    }
    auto t2 = clk::now();
    std::ofstream f(argv[4], std::ios::binary);
    f.write(reinterpret_cast<char *>(mask.data()), n);
    std::printf("{\"n_points\": %zu, \"n_known\": %lld, \"load_s\": %.1f, \"query_s\": %.1f}\n", n, known,
                std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count());
    return 0;
  }
  std::cerr << "bad arguments" << std::endl;
  return 1;
}
