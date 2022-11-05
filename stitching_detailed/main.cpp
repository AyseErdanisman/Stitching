#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

#include <fstream>
#include <string>
#include <opencv4/opencv2/opencv_modules.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

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


static void printUsage(char** argv)
{
    std::cout <<
        "Rotation model images stitcher.\n\n"
         << argv[0] << " img1 img2 [...imgN] [flags]\n\n"
        "Flags:\n"
        "  --preview\n"
        "      Run stitching in the preview mode. Works faster than usual mode,\n"
        "      but output image will have lower resolution.\n"
        "  --try_cuda (yes|no)\n"
        "      Try to use CUDA. The default value is 'no'. All default values\n"
        "      are for CPU mode.\n"
        "\nMotion Estimation Flags:\n"
        "  --work_megapix <float>\n"
        "      Resolution for image registration step. The default is 0.6 Mpx.\n"
        "  --features (surf|orb|sift|akaze)\n"
        "      Type of features used for images matching.\n"
        "      The default is surf if available, orb otherwise.\n"
        "  --matcher (homography|affine)\n"
        "      Matcher used for pairwise image matching.\n"
        "  --estimator (homography|affine)\n"
        "      Type of estimator used for transformation estimation.\n"
        "  --match_conf <float>\n"
        "      Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.\n"
        "  --conf_thresh <float>\n"
        "      Threshold for two images are from the same panorama confidence.\n"
        "      The default is 1.0.\n"
        "  --ba (no|reproj|ray|affine)\n"
        "      Bundle adjustment cost function. The default is ray.\n"
        "  --ba_refine_mask (mask)\n"
        "      Set refinement mask for bundle adjustment. It looks like 'x_xxx',\n"
        "      where 'x' means refine respective parameter and '_' means don't\n"
        "      refine one, and has the following format:\n"
        "      <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle\n"
        "      adjustment doesn't support estimation of selected parameter then\n"
        "      the respective flag is ignored.\n"
        "  --wave_correct (no|horiz|vert)\n"
        "      Perform wave effect correction. The default is 'horiz'.\n"
        "  --save_graph <file_name>\n"
        "      Save matches graph represented in DOT language to <file_name> file.\n"
        "      Labels description: Nm is number of matches, Ni is number of inliers,\n"
        "      C is confidence.\n"
        "\nCompositing Flags:\n"
        "  --warp (affine|plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)\n"
        "      Warp surface type. The default is 'spherical'.\n"
        "  --seam_megapix <float>\n"
        "      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
        "  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
        "      Seam estimation method. The default is 'gc_color'.\n"
        "  --compose_megapix <float>\n"
        "      Resolution for compositing step. Use -1 for original resolution.\n"
        "      The default is -1.\n"
        "  --expos_comp (no|gain|gain_blocks|channels|channels_blocks)\n"
        "      Exposure compensation method. The default is 'gain_blocks'.\n"
        "  --expos_comp_nr_feeds <int>\n"
        "      Number of exposure compensation feed. The default is 1.\n"
        "  --expos_comp_nr_filtering <int>\n"
        "      Number of filtering iterations of the exposure compensation gains.\n"
        "      Only used when using a block exposure compensation method.\n"
        "      The default is 2.\n"
        "  --expos_comp_block_size <int>\n"
        "      BLock size in pixels used by the exposure compensator.\n"
        "      Only used when using a block exposure compensation method.\n"
        "      The default is 32.\n"
        "  --blend (no|feather|multiband)\n"
        "      Blending method. The default is 'multiband'.\n"
        "  --blend_strength <float>\n"
        "      Blending strength from [0,100] range. The default is 5.\n"
        "  --output <result_img>\n"
        "      The default is 'result.jpg'.\n"
        "  --timelapse (as_is|crop) \n"
        "      Output warped images separately as frames of a time lapse movie, with 'fixed_' prepended to input file names.\n"
        "  --rangewidth <int>\n"
        "      uses range_width to limit number of images to match with.\n";
}
// Default command line args
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

static int parseCmdArgs(int argc, char** argv)
{
//    if (argc == 1)
//    {
//        printUsage(argv);
//        return -1;
//    }
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "/?")
        {
            printUsage(argv);
            return -1;
        }
        else if (std::string(argv[i]) == "--preview")
        {
            preview = true;
        }
        else if (std::string(argv[i]) == "--try_cuda")
        {
            if (std::string(argv[i + 1]) == "no")
                try_cuda = false;
            else if (std::string(argv[i + 1]) == "yes")
                try_cuda = true;
            else
            {
                std::cout << "Bad --try_cuda flag value\n";
                return -1;
            }
            i++;
        }
        else if (std::string(argv[i]) == "--work_megapix")
        {
            work_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (std::string(argv[i]) == "--seam_megapix")
        {
            seam_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (std::string(argv[i]) == "--compose_megapix")
        {
            compose_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (std::string(argv[i]) == "--result")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (std::string(argv[i]) == "--features")
        {
            features_type = argv[i + 1];
            if (std::string(features_type) == "orb")
                match_conf = 0.3f;
            i++;
        }
        else if (std::string(argv[i]) == "--matcher")
        {
            if (std::string(argv[i + 1]) == "homography" || std::string(argv[i + 1]) == "affine")
                matcher_type = argv[i + 1];
            else
            {
                std::cout << "Bad --matcher flag value\n";
                return -1;
            }
            i++;
        }
        else if (std::string(argv[i]) == "--estimator")
        {
            if (std::string(argv[i + 1]) == "homography" || std::string(argv[i + 1]) == "affine")
                estimator_type = argv[i + 1];
            else
            {
                std::cout << "Bad --estimator flag value\n";
                return -1;
            }
            i++;
        }
        else if (std::string(argv[i]) == "--match_conf")
        {
            match_conf = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (std::string(argv[i]) == "--conf_thresh")
        {
            conf_thresh = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (std::string(argv[i]) == "--ba")
        {
            ba_cost_func = argv[i + 1];
            i++;
        }
        else if (std::string(argv[i]) == "--ba_refine_mask")
        {
            ba_refine_mask = argv[i + 1];
            if (ba_refine_mask.size() != 5)
            {
                std::cout << "Incorrect refinement mask length.\n";
                return -1;
            }
            i++;
        }
        else if (std::string(argv[i]) == "--wave_correct")
        {
            if (std::string(argv[i + 1]) == "no")
                do_wave_correct = false;
            else if (std::string(argv[i + 1]) == "horiz")
            {
                do_wave_correct = true;
                wave_correct = cv::detail::WAVE_CORRECT_HORIZ;
            }
            else if (std::string(argv[i + 1]) == "vert")
            {
                do_wave_correct = true;
                wave_correct = cv::detail::WAVE_CORRECT_VERT;
            }
            else
            {
                std::cout << "Bad --wave_correct flag value\n";
                return -1;
            }
            i++;
        }
        else if (std::string(argv[i]) == "--save_graph")
        {
            save_graph = true;
            save_graph_to = argv[i + 1];
            i++;
        }
        else if (std::string(argv[i]) == "--warp")
        {
            warp_type = std::string(argv[i + 1]);
            i++;
        }
        else if (std::string(argv[i]) == "--expos_comp")
        {
            if (std::string(argv[i + 1]) == "no")
                expos_comp_type = cv::detail::ExposureCompensator::NO;
            else if (std::string(argv[i + 1]) == "gain")
                expos_comp_type = cv::detail::ExposureCompensator::GAIN;
            else if (std::string(argv[i + 1]) == "gain_blocks")
                expos_comp_type = cv::detail::ExposureCompensator::GAIN_BLOCKS;
            else if (std::string(argv[i + 1]) == "channels")
                expos_comp_type = cv::detail::ExposureCompensator::CHANNELS;
            else if (std::string(argv[i + 1]) == "channels_blocks")
                expos_comp_type = cv::detail::ExposureCompensator::CHANNELS_BLOCKS;
            else
            {
                std::cout << "Bad exposure compensation method\n";
                return -1;
            }
            i++;
        }
        else if (std::string(argv[i]) == "--expos_comp_nr_feeds")
        {
            expos_comp_nr_feeds = atoi(argv[i + 1]);
            i++;
        }
        else if (std::string(argv[i]) == "--expos_comp_nr_filtering")
        {
            expos_comp_nr_filtering = atoi(argv[i + 1]);
            i++;
        }
        else if (std::string(argv[i]) == "--expos_comp_block_size")
        {
            expos_comp_block_size = atoi(argv[i + 1]);
            i++;
        }
        else if (std::string(argv[i]) == "--seam")
        {
            if (std::string(argv[i + 1]) == "no" ||
                std::string(argv[i + 1]) == "voronoi" ||
                std::string(argv[i + 1]) == "gc_color" ||
                std::string(argv[i + 1]) == "gc_colorgrad" ||
                std::string(argv[i + 1]) == "dp_color" ||
                std::string(argv[i + 1]) == "dp_colorgrad")
                seam_find_type = argv[i + 1];
            else
            {
                std::cout << "Bad seam finding method\n";
                return -1;
            }
            i++;
        }
        else if (std::string(argv[i]) == "--blend")
        {
            if (std::string(argv[i + 1]) == "no")
                blend_type = cv::detail::Blender::NO;
            else if (std::string(argv[i + 1]) == "feather")
                blend_type = cv::detail::Blender::FEATHER;
            else if (std::string(argv[i + 1]) == "multiband")
                blend_type = cv::detail::Blender::MULTI_BAND;
            else
            {
                std::cout << "Bad blending method\n";
                return -1;
            }
            i++;
        }
        else if (std::string(argv[i]) == "--timelapse")
        {
            timelapse = true;
            if (std::string(argv[i + 1]) == "as_is")
                timelapse_type = cv::detail::Timelapser::AS_IS;
            else if (std::string(argv[i + 1]) == "crop")
                timelapse_type = cv::detail::Timelapser::CROP;
            else
            {
                std::cout << "Bad timelapse method\n";
                return -1;
            }
            i++;
        }
        else if (std::string(argv[i]) == "--rangewidth")
        {
            range_width = atoi(argv[i + 1]);
            i++;
        }
        else if (std::string(argv[i]) == "--blend_strength")
        {
            blend_strength = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (std::string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else
            img_names.push_back(argv[i]);
    }
    if (preview)
    {
        compose_megapix = 0.6;
    }
    return 0;
}
int main(int argc, char* argv[])
{

//----------ben ekledim-begin---------------------------------------------------------------------

    cv::Mat capture1;
    cv::Mat capture2;
    cv::namedWindow("video-1");//Declaring the video to show the video//
    cv::namedWindow("video-2");
    cv::VideoCapture cap1(4);
    cv::VideoCapture cap2(2);
    if(!cap1.isOpened())
    {
        std::cout << "No video stream detected" << std::endl;
        system("pause");
        return-1;
    }

    if(!cap2.isOpened())
    {
        std::cout <<"No video stream detected" << std::endl;
        system("pause");
        return-1;
    }
    cap1 >> capture1;
    cap2 >> capture2;
    cv::imwrite("cap1img.png",capture1);
    cv::imwrite("cap2img.png",capture2);

//----------ben ekledim-end---------------------------------------------------------------------

#if ENABLE_LOG
    int64 app_start_time = cv::getTickCount();
#endif
#if 0
    cv::setBreakOnError(true);
#endif
    int retval = parseCmdArgs(argc, argv);
    if (retval)
        return retval;
    // Check if have enough images
    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }
    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
    LOGLN("Finding features...");
#if ENABLE_LOG
    int64 t = cv::getTickCount();
#endif
    cv::Ptr<cv::Feature2D> finder;
    if (features_type == "orb")
    {
        finder = cv::ORB::create();
    }
    else if (features_type == "akaze")
    {
        finder = cv::AKAZE::create();
    }
#ifdef HAVE_OPENCV_XFEATURES2D
    else if (features_type == "surf")
    {
        finder = cv::xfeatures2d::SURF::create();
    }
#endif
    else if (features_type == "sift")
    {
        finder = cv::SIFT::create();
    }
    else
    {
        std::cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return -1;
    }
    cv::Mat full_img, img;
    std::vector<cv::detail::ImageFeatures> features(num_images);
    std::vector<cv::Mat> images(num_images);
    std::vector<cv::Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;
    for (int i = 0; i < num_images; ++i)
    {
        full_img = cv::imread(cv::samples::findFile(img_names[i]));
        full_img_sizes[i] = full_img.size();
        if (full_img.empty())
        {
            LOGLN("Can't open image " << img_names[i]);
            return -1;
        }
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = cv::min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, cv::Size(), work_scale, work_scale, cv::INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = cv::min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }
        computeImageFeatures(finder, img, features[i]);
        features[i].img_idx = i;
        LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());
        resize(full_img, img, cv::Size(), seam_scale, seam_scale, cv::INTER_LINEAR_EXACT);
        images[i] = img.clone();
    }
    full_img.release();
    img.release();
    LOGLN("Finding features, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");
    LOG("Pairwise matching");
#if ENABLE_LOG
    t = cv::getTickCount();
#endif
    std::vector<cv::detail::MatchesInfo> pairwise_matches;
    cv::Ptr<cv::detail::FeaturesMatcher> matcher;
    if (matcher_type == "affine")
        matcher = cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
    else if (range_width==-1)
        matcher = cv::makePtr<cv::detail::BestOf2NearestMatcher>(try_cuda, match_conf);
    else
        matcher = cv::makePtr<cv::detail::BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);
    (*matcher)(features, pairwise_matches);
    matcher->collectGarbage();
    LOGLN("Pairwise matching, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");
    // Check if we should save matches graph
    if (save_graph)
    {
        LOGLN("Saving matches graph...");
        std::ofstream f(save_graph_to.c_str());
        f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
    }
    // Leave only images we are sure are from the same panorama
    std::vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    std::vector<cv::Mat> img_subset;
    std::vector<cv::String> img_names_subset;
    std::vector<cv::Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }
    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;
    // Check if we still have enough images
    num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }
    cv::Ptr<cv::detail::Estimator> estimator;
    if (estimator_type == "affine")
        estimator = cv::makePtr<cv::detail::AffineBasedEstimator>();
    else
        estimator = cv::makePtr<cv::detail::HomographyBasedEstimator>();
    std::vector<cv::detail::CameraParams> cameras;
    if (!(*estimator)(features, pairwise_matches, cameras))
    {
        std::cout << "Homography estimation failed.\n";
        return -1;
    }
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        cv::Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        LOGLN("Initial camera intrinsics #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
    }
    cv::Ptr<cv::detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = cv::makePtr<cv::detail::BundleAdjusterReproj>();
    else if (ba_cost_func == "ray") adjuster = cv::makePtr<cv::detail::BundleAdjusterRay>();
    else if (ba_cost_func == "affine") adjuster = cv::makePtr<cv::detail::BundleAdjusterAffinePartial>();
    else if (ba_cost_func == "no") adjuster = cv::makePtr<cv::detail::NoBundleAdjuster>();
    else
    {
        std::cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
        return -1;
    }
    adjuster->setConfThresh(conf_thresh);
    cv::Mat_<uchar> refine_mask = cv::Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
    adjuster->setRefinementMask(refine_mask);
    if (!(*adjuster)(features, pairwise_matches, cameras))
    {
        std::cout << "Camera parameters adjusting failed.\n";
        return -1;
    }
    // Find median focal length
    std::vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        LOGLN("Camera #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
        focals.push_back(cameras[i].focal);
    }
    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
    if (do_wave_correct)
    {
        std::vector<cv::Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }
    LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
    t = cv::getTickCount();
#endif
    std::vector<cv::Point> corners(num_images);
    std::vector<cv::UMat> masks_warped(num_images);
    std::vector<cv::UMat> images_warped(num_images);
    std::vector<cv::Size> sizes(num_images);
    std::vector<cv::UMat> masks(num_images);
    // Prepare images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(cv::Scalar::all(255));
    }
    // Warp images and their masks
    cv::Ptr<cv::WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_CUDAWARPING
    if (try_cuda && cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane")
            warper_creator = cv::makePtr<cv::PlaneWarperGpu>();
        else if (warp_type == "cylindrical")
            warper_creator = cv::makePtr<cv::CylindricalWarperGpu>();
        else if (warp_type == "spherical")
            warper_creator = cv::makePtr<cv::SphericalWarperGpu>();
    }
    else
#endif
    {
        if (warp_type == "plane")
            warper_creator = cv::makePtr<cv::PlaneWarper>();
        else if (warp_type == "affine")
            warper_creator = cv::makePtr<cv::AffineWarper>();
        else if (warp_type == "cylindrical")
            warper_creator = cv::makePtr<cv::CylindricalWarper>();
        else if (warp_type == "spherical")
            warper_creator = cv::makePtr<cv::SphericalWarper>();
        else if (warp_type == "fisheye")
            warper_creator = cv::makePtr<cv::FisheyeWarper>();
        else if (warp_type == "stereographic")
            warper_creator = cv::makePtr<cv::StereographicWarper>();
        else if (warp_type == "compressedPlaneA2B1")
            warper_creator = cv::makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlaneA1.5B1")
            warper_creator = cv::makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA2B1")
            warper_creator = cv::makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA1.5B1")
            warper_creator = cv::makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniA2B1")
            warper_creator = cv::makePtr<cv::PaniniWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniA1.5B1")
            warper_creator = cv::makePtr<cv::PaniniWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniPortraitA2B1")
            warper_creator = cv::makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniPortraitA1.5B1")
            warper_creator = cv::makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "mercator")
            warper_creator = cv::makePtr<cv::MercatorWarper>();
        else if (warp_type == "transverseMercator")
            warper_creator = cv::makePtr<cv::TransverseMercatorWarper>();
    }
    if (!warper_creator)
    {
        std::cout << "Can't create the following warper '" << warp_type << "'\n";
        return 1;
    }
    cv::Ptr<cv::detail::RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
    for (int i = 0; i < num_images; ++i)
    {
        cv::Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;
        corners[i] = warper->warp(images[i], K, cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();
        warper->warp(masks[i], K, cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, masks_warped[i]);
    }
    std::vector<cv::UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);
    LOGLN("Warping images, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");
    LOGLN("Compensating exposure...");
#if ENABLE_LOG
    t = cv::getTickCount();
#endif
    cv::Ptr<cv::detail::ExposureCompensator> compensator = cv::detail::ExposureCompensator::createDefault(expos_comp_type);
    if (dynamic_cast<cv::detail::GainCompensator*>(compensator.get()))
    {
        cv::detail::GainCompensator* gcompensator = dynamic_cast<cv::detail::GainCompensator*>(compensator.get());
        gcompensator->setNrFeeds(expos_comp_nr_feeds);
    }
    if (dynamic_cast<cv::detail::ChannelsCompensator*>(compensator.get()))
    {
        cv::detail::ChannelsCompensator* ccompensator = dynamic_cast<cv::detail::ChannelsCompensator*>(compensator.get());
        ccompensator->setNrFeeds(expos_comp_nr_feeds);
    }
    if (dynamic_cast<cv::detail::BlocksCompensator*>(compensator.get()))
    {
        cv::detail::BlocksCompensator* bcompensator = dynamic_cast<cv::detail::BlocksCompensator*>(compensator.get());
        bcompensator->setNrFeeds(expos_comp_nr_feeds);
        bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
        bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
    }
    compensator->feed(corners, images_warped, masks_warped);
    LOGLN("Compensating exposure, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");
    LOGLN("Finding seams...");
#if ENABLE_LOG
    t = cv::getTickCount();
#endif
    cv::Ptr<cv::detail::SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = cv::makePtr<cv::detail::NoSeamFinder>();
    else if (seam_find_type == "voronoi")
        seam_finder = cv::makePtr<cv::detail::VoronoiSeamFinder>();
    else if (seam_find_type == "gc_color")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_cuda && cv::cuda::getCudaEnabledDeviceCount() > 0)
            seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinderGpu>(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
        else
#endif
            seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_cuda && cv::cuda::getCudaEnabledDeviceCount() > 0)
            seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinderGpu>(cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
        else
#endif
            seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (seam_find_type == "dp_color")
        seam_finder = cv::makePtr<cv::detail::DpSeamFinder>(cv::detail::DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = cv::makePtr<cv::detail::DpSeamFinder>(cv::detail::DpSeamFinder::COLOR_GRAD);
    if (!seam_finder)
    {
        std::cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
        return 1;
    }
    seam_finder->find(images_warped_f, corners, masks_warped);
    LOGLN("Finding seams, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");
    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();
    LOGLN("Compositing...");
#if ENABLE_LOG
    t = cv::getTickCount();
#endif
    cv::Mat img_warped, img_warped_s;
    cv::Mat dilated_mask, seam_mask, mask, mask_warped;
    cv::Ptr<cv::detail::Blender> blender;
    cv::Ptr<cv::detail::Timelapser> timelapser;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        LOGLN("Compositing image #" << indices[img_idx]+1);
        // Read image and resize it if necessary
        full_img = cv::imread(cv::samples::findFile(img_names[img_idx]));
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = cv::min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;
            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;
            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);
            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;
                // Update corner and size
                cv::Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }
                cv::Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                cv::Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, cv::Size(), compose_scale, compose_scale, cv::INTER_LINEAR_EXACT);
        else
            img = full_img;
        full_img.release();
        cv::Size img_size = img.size();
        cv::Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);
        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, img_warped);
        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(cv::Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, mask_warped);
        // Compensate exposure
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();
        dilate(masks_warped[img_idx], dilated_mask, cv::Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, cv::INTER_LINEAR_EXACT);
        mask_warped = seam_mask & mask_warped;
        if (!blender && !timelapse)
        {
            blender = cv::detail::Blender::createDefault(blend_type, try_cuda);
            cv::Size dst_sz = cv::detail::resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = cv::detail::Blender::createDefault(cv::detail::Blender::NO, try_cuda);
            else if (blend_type == cv::detail::Blender::MULTI_BAND)
            {
                cv::detail::MultiBandBlender* mb = dynamic_cast<cv::detail::MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
                LOGLN("Multi-band blender, number of bands: " << mb->numBands());
            }
            else if (blend_type == cv::detail::Blender::FEATHER)
            {
                cv::detail::FeatherBlender* fb = dynamic_cast<cv::detail::FeatherBlender*>(blender.get());
                fb->setSharpness(1.f/blend_width);
                LOGLN("Feather blender, sharpness: " << fb->sharpness());
            }
            blender->prepare(corners, sizes);
        }
        else if (!timelapser && timelapse)
        {
            timelapser = cv::detail::Timelapser::createDefault(timelapse_type);
            timelapser->initialize(corners, sizes);
        }
        // Blend the current image
        if (timelapse)
        {
            timelapser->process(img_warped_s, cv::Mat::ones(img_warped_s.size(), CV_8UC1), corners[img_idx]);
            cv::String fixedFileName;
            size_t pos_s = cv::String(img_names[img_idx]).find_last_of("/\\");
            if (pos_s == cv::String::npos)
            {
                fixedFileName = "fixed_" + img_names[img_idx];
            }
            else
            {
                fixedFileName = "fixed_" + cv::String(img_names[img_idx]).substr(pos_s + 1, cv::String(img_names[img_idx]).length() - pos_s);
            }
            imwrite(fixedFileName, timelapser->getDst());
        }
        else
        {
            blender->feed(img_warped_s, mask_warped, corners[img_idx]);
        }
    }
    if (!timelapse)
    {

        cv::Mat result, result_mask;
        blender->blend(result, result_mask);
        LOGLN("Compositing, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");
        imwrite(result_name, result);

//----------ben ekledim-begin---------------------------------------------------------------------

        cv::Mat result1 = cv::imread("result.jpg");
        cv::imshow("result", result1);

//----------ben ekledim-end---------------------------------------------------------------------

    }
    LOGLN("Finished, total time: " << ((cv::getTickCount() - app_start_time) / cv::getTickFrequency()) << " sec");

//----------ben ekledim-begin---------------------------------------------------------------------

    while(true)
    {
        //Taking an everlasting loop to show the video//
        cap1 >> capture1;
        cap2 >> capture2;

        //Video capture resize
        cv::resize(capture1, capture1, cv::Size(854, 480));
        cv::resize(capture2, capture2, cv::Size(854, 480));

        if(capture1.empty()) break;
        if(capture2.empty()) break;

        //Showing the video//
        cv::imshow("video-1", capture1);
        cv::imshow("video-2", capture2);

        //Allowing 25 milliseconds frame processing time and initiating break condition//
        char c = (char)cv::waitKey(25);
        if (c == 27)
        {
            //If 'Esc' is entered break the loop//
            break;
        }

        //Releasing the buffer memory//

    }
    cap1.release();
    cap2.release();

//----------ben ekledim-end---------------------------------------------------------------------

    return 0;
}
