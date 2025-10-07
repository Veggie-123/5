#ifndef Yolo_H
#define Yolo_H

#include <iostream> // 标准输入输出流库
#include <cstdlib> // 标准库
#include <unistd.h> // Unix标准库
#include <vector> // 向量容器库
#include <sys/time.h> // 时间库

#include <opencv2/opencv.hpp> // OpenCV主头文件
#include <opencv4/opencv2/core/core.hpp> // OpenCV核心功能
#include <opencv4/opencv2/highgui.hpp> // OpenCV高层GUI功能
#include <opencv4/opencv2/imgproc/imgproc_c.h> // OpenCV图像处理功能

#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>

/*****************************yolo-fastestv2**************************************/
typedef struct {
    int width;
    int height;
} YoloSize;

typedef struct {
    std::string name;
    int stride;
    std::vector<YoloSize> anchors;
} YoloLayerData;

typedef struct {
    int x1,y1,x2,y2,label,id;
    float score;
} BoxInfo;

/*****************************yolov5-lite**************************************/
typedef struct {
    float x1, y1, x2, y2, score;
    int label;
} BoxInfo_v5lite;

typedef struct {
    int inpSize, maxSide, Padw, Padh;
    float ratio;
} MatInfo;
/******************************************************************************/

class yolo_fv2_mnn {
public:
/*****************************yolo-fastestv2**************************************/
    yolo_fv2_mnn(float input_threshold, 
                 float input_nms_threshold, 
                 std::string input_model_name, 
                 int input_num_classes,
                 std::vector<std::string> input_labels);

    std::vector<BoxInfo> decode_infer(MNN::Tensor & data, int stride, const std::vector<YoloSize> &anchors);

    void nms(std::vector<BoxInfo> &result);

    void scale_coords(std::vector<BoxInfo> &boxes, int w_from, int h_from, int w_to, int h_to);

    cv::Mat draw_box(cv::Mat & cv_mat, std::vector<BoxInfo> &boxes);
    void draw_box(std::vector<BoxInfo> &boxes);

    std::vector<BoxInfo> detect(cv::Mat &src);

/*****************************yolov5-lite**************************************/
    yolo_fv2_mnn(float input_nms_threshold);

    cv::Mat preprocess_v5lite(cv::Mat &cv_mat);

    std::vector<BoxInfo_v5lite> decode_v5lite(cv::Mat &raw_image);

    void nms_v5lite(std::vector<BoxInfo_v5lite> &result);

    void draw_box_v5lite(cv::Mat &cv_mat, std::vector<BoxInfo_v5lite> &boxes);
/******************************************************************************/
    ~yolo_fv2_mnn();

private:
    float nms_threshold;

    std::string model_name;
    std::shared_ptr<MNN::Interpreter> net;
    MNN::ScheduleConfig config;//进程配置
    MNN::BackendConfig backendConfig;//后端配置
    MNN::Session *session;//创建进程
/*****************************yolo-fastestv2**************************************/
    int num_classes;//类别数
    std::vector<std::string> labels; //定义类别标签

    int input_size = 320;//输入尺寸
    YoloSize yolosize = YoloSize{input_size, input_size};

    float threshold;

    std::vector<YoloLayerData> layers{//设置yolov5s的层信息，包括层名，步长，锚点信息
            {"772", 32, {{69,  28}, {92,  128},  {249,  190}}},//10,320
            {"770", 16,  {{16,  11}, {26,  32},  {48,  78}}},//20,320
    };
/*****************************yolov5-lite**************************************/
    MatInfo mmat_objection;
/******************************************************************************/
};

#endif //Yolo_H
