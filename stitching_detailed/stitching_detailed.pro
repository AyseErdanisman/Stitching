TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_aruco
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_barcode
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_bgsegm
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_bioinspired
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_calib3d
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_ccalib
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_core
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_cudaarithm
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_cudabgsegm
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_cudacodec
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_cudafeatures2d
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_cudafilters
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_cudaimgproc
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_cudalegacy
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_cudaobjdetect
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_cudaoptflow
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_cudastereo
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_cudawarping
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_cudev
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_datasets
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_dnn_objdetect
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_dnn
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_dnn_superres
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_dpm
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_face
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_features2d
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_flann
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_freetype
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_fuzzy
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_gapi
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_hfs
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_highgui
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_imgcodecs
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_img_hash
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_imgproc
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_intensity_transform
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_line_descriptor
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_mcc
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_ml
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_objdetect
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_optflow
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_phase_unwrapping
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_photo
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_plot
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_quality
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_rapid
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_reg
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_rgbd
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_saliency
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_shape
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_stereo
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_stitching
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_structured_light
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_superres
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_surface_matching
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_text
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_tracking
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_videoio
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_video
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_videostab
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_wechat_qrcode
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_xfeatures2d
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_ximgproc
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_xobjdetect
unix|win32: LIBS += -L$$PWD/../../../../../../usr/local/lib/ -lopencv_xphoto

INCLUDEPATH += $$PWD/../../../../../../usr/local/include/opencv4
DEPENDPATH += $$PWD/../../../../../../usr/local/include/opencv4

HEADERS +=