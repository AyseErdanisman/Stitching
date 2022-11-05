#include "warper.h"
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif
#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

std::vector<cv::String> img_names = {{"cap1img.png"}, {"cap2img.png"}};
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
#ifdef HAVE_OPENCV_XFEATURES2D
std::string features_type = "sift";
float match_conf = 0.65f;
#else
string features_type = "orb";
float match_conf = 0.3f;
#endif
std::string matcher_type = "homography";
std::string estimator_type = "homography";
std::string ba_cost_func = "ray";
std::string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
cv::detail::WaveCorrectKind wave_correct = cv::detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
std::string warp_type = "spherical";
int expos_comp_type = cv::detail::ExposureCompensator::GAIN_BLOCKS;
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 32;
std::string seam_find_type = "gc_color";
int blend_type = cv::detail::Blender::MULTI_BAND;
int timelapse_type = cv::detail::Timelapser::AS_IS;
float blend_strength = 5;
std::string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;

Warper::Warper()
{

}
