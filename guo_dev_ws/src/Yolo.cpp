#include "Yolo.h"

using namespace MNN;

#define display 0
#define camera  1

yolo_fv2_mnn::yolo_fv2_mnn( float input_threshold, 
                            float input_nms_threshold, 
                            std::string input_model_name,
                            int input_num_classes,
                            std::vector<std::string> input_labels) {
    threshold = input_threshold;
    nms_threshold = input_nms_threshold;
    model_name = input_model_name;
    num_classes = input_num_classes;
    labels = input_labels;
    net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_name.c_str()));//创建解释器
    if (nullptr == net) {//判断解释器是否为空
        std::cout << "Interpreter create failed!" << std::endl;
    }
    config.numThread = 4;//4个线程
    config.type      = static_cast<MNNForwardType>(MNN_FORWARD_CPU);//前向类型为CPU
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)1;//意思是Precision_Normal，用来设置精度
    // backendConfig.precision = MNN::BackendConfig::Precision_Normal;
    backendConfig.memory = MNN::BackendConfig::Memory_Low;
    config.backendConfig = &backendConfig;//设置后端配置
    session = net->createSession(config);//创建进程
}

yolo_fv2_mnn::~yolo_fv2_mnn() {
    if (session) {
        net->releaseSession(session);
        session = nullptr;
    }
}

std::vector<BoxInfo> yolo_fv2_mnn::decode_infer(MNN::Tensor &data, int stride, const std::vector<YoloSize> &anchors) {
    std::vector<BoxInfo> result;
    int numAnchor = 3;
    int dataB = data.shape()[0];//b
    int dataH = data.shape()[1];//h
    int dataW = data.shape()[2];//w
    int dataC = data.shape()[3];//p

    auto data_ptr = data.host<float>();
    for (int b = 0; b < dataB; b++) {
        auto batch_ptr = data_ptr +b * dataH * dataW * dataC;
        for (int h = 0; h < dataH; h++) {
            auto height_ptr = batch_ptr + h * dataW * dataC;
            for (int w = 0; w < dataW; w++) {
                auto pred_ptr = height_ptr + w * dataC;
                for (int p = 0; p < numAnchor; p++) {
                    int category = -1;
                    float score = -1;
                    float objScore = pred_ptr[4 * numAnchor + p];
                    float tmp = 0;
                    for (int j = 0; j < num_classes; j++) {
                        float clsScore = pred_ptr[4 * numAnchor + numAnchor + j];
                        clsScore *= objScore;
                        if (clsScore > tmp) {
                            score = clsScore;
                            category = j;
                            tmp = clsScore;
                        }
                    }
                    if (score > threshold) {
                        float bcx = (pred_ptr[p * 4 + 0] * 2. - 0.5 + w) * stride;
                        float bcy = (pred_ptr[p * 4 + 1] * 2. - 0.5 + h) * stride;
                        float bw = pow((pred_ptr[p * 4 + 2] * 2.), 2) * anchors[p].width;
                        float bh = pow((pred_ptr[p * 4 + 3] * 2.), 2) * anchors[p].height;

                        BoxInfo box;

                        box.x1 = std::max(0, std::min(yolosize.width, int((bcx - 0.5 * bw))));
                        box.y1 = std::max(0, std::min(yolosize.height, int((bcy - 0.5 * bh))));
                        box.x2 = std::max(0, std::min(yolosize.width, int((bcx + 0.5 * bw))));
                        box.y2 = std::max(0, std::min(yolosize.height, int((bcy + 0.5 * bh))));

                        box.score = score;
                        box.label = category;
                        result.push_back(box);
                    }
                }
            }
        }
    }
    return result;
}

void yolo_fv2_mnn::nms(std::vector<BoxInfo> &input_boxes) {
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= nms_threshold) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}

//解码，将由调整后的图像推理出的框坐标转换为原图像的坐标，形参为框，调整后的图像尺寸，原图像尺寸
void yolo_fv2_mnn::scale_coords(std::vector<BoxInfo> &boxes, int w_from, int h_from, int w_to, int h_to){
    float w_ratio = float(w_to)/float(w_from);
    float h_ratio = float(h_to)/float(h_from);
 
    for(auto &box: boxes){//遍历所有的box
        box.x1 *= w_ratio;
        box.x2 *= w_ratio;
        box.y1 *= h_ratio;
        box.y2 *= h_ratio;
    }
}

//画框,形参是cv::Mat类型的图像，std::vector<BoxInfo>类型的boxes，std::vector<std::string>类型的labels
cv::Mat yolo_fv2_mnn::draw_box(cv::Mat & cv_mat, std::vector<BoxInfo> &boxes){
    std::cout<<"objects : "<<boxes.size()<<std::endl;
    int CNUM = 6;//意思是有4类
    cv::RNG rng(0xFFFFFFFF);//意思是随机数种子，用来生成随机数
    cv::Scalar_<int> randColor[CNUM];//意思是生成4个随机颜色
    for (int i = 0; i < CNUM; i++)//生成4个随机颜色
        rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
 
    for(auto box : boxes){//遍历所有的box
        int width = box.x2-box.x1;//box的宽度
        int height = box.y2-box.y1;//box的高度
        // int id = box.id;//box的id
        cv::Point p = cv::Point(box.x1, box.y1);//box左上角坐标
        cv::Rect rect = cv::Rect(box.x1, box.y1, width, height);//定义box的矩形框
        cv::rectangle(cv_mat, rect, randColor[box.label]);//画矩形框
        // std::string text = labels[box.label] + ":" + std::to_string(box.score) + " ID:" + std::to_string(id);//定义文字
        std::string text = labels[box.label] + ":" + std::to_string(box.score);//定义文字
        cv::putText(cv_mat, text, p, cv::FONT_HERSHEY_PLAIN, 1, randColor[box.label]);//在图像上标注文字

        //输出推理信息
        std::cout <<"class : "    << labels[box.label] <<"  "<<box.score<<
                    " center_x = "<< (box.x1+box.x2)/2    << "  center_y = "<< (box.y1+box.y2)/2<<std::endl;
    }
    return cv_mat;//返回图像
}

void yolo_fv2_mnn::draw_box(std::vector<BoxInfo> &boxes){//输出推理信息
    // if(labels[0]=="zhuitong") {
    //     std::vector<BoxInfo> result_zhuitong;
    //     int index = 0;
    //     float p = boxes[0].score;
    //     for (auto box : boxes) {
    //         if (box.score > p) {
    //             p = box.score;
    //             index = box.label;
    //         }
    //     }
    //     result_zhuitong.push_back(boxes[index]);
    //     boxes = result_zhuitong;
    // }

    std::cout<<"objects : "<<boxes.size()<<std::endl;
    for(auto box : boxes){//遍历所有的box
        std::cout <<"class : "    << labels[box.label] <<"  "<<box.score<<
                    " center_x = "<< (box.x1+box.x2)/2    << "  center_y = "<< (box.y1+box.y2)/2<<std::endl;
    }
}

std::vector<BoxInfo> yolo_fv2_mnn::detect( cv::Mat &src ) {

    double t = (double)cv::getTickCount();//fps计时

    // preprocessing
    cv::Mat image;//初始化图像
    cv::resize(src, image, cv::Size(input_size, input_size));//调整图像尺寸
    image.convertTo(image, CV_32FC3);
    image = image /255.0f;
 
    // wrapping input tensor, convert nhwc to nchw
    std::vector<int> dims{1, 320, 320, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    std::memcpy(nhwc_data, image.data, nhwc_size);
    auto inputTensor = net->getSessionInput(session, nullptr);
    inputTensor->copyFromHostTensor(nhwc_Tensor);

    // run network
    net->runSession(session);

    // get output data
    std::vector<BoxInfo> boxes;
    std::vector<BoxInfo> result;

    std::string output_tensor_name1 = layers[1].name ;
    std::string output_tensor_name2 = layers[0].name ;

    MNN::Tensor *tensor_boxes   = net->getSessionOutput(session, output_tensor_name1.c_str());
    MNN::Tensor *tensor_anchors = net->getSessionOutput(session, output_tensor_name2.c_str());

    MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
    MNN::Tensor tensor_anchors_host(tensor_anchors, tensor_anchors->getDimensionType());

    tensor_boxes->copyToHostTensor(&tensor_boxes_host);
    tensor_anchors->copyToHostTensor(&tensor_anchors_host);

    boxes = decode_infer(tensor_boxes_host, layers[1].stride, layers[1].anchors);
    result.insert(result.begin(), boxes.begin(), boxes.end());
    boxes = decode_infer(tensor_anchors_host, layers[0].stride, layers[0].anchors);
    result.insert(result.begin(), boxes.begin(), boxes.end());

    nms(result);

    scale_coords(result, input_size, input_size, src.cols, src.rows);

    t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
    double fps = 1.0/t;
    std::cout << "fps : "<< fps << std::endl;
    
    delete nhwc_Tensor;

#if display
    cv::Mat frame_show = draw_box(src, result);
    char str[10];
    sprintf(str, "%.2f", fps);
    std::string fpsString("FPS:");
    fpsString += str;
    cv::putText(frame_show, fpsString, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
#if camera
    cv::imshow("result", frame_show);
#else
    cv::imwrite("/home/pi/yolo_fv2_mnn/output.jpg", frame_show);
#endif
#else
    draw_box(result);
#endif
    return result;
}

yolo_fv2_mnn::yolo_fv2_mnn(float input_nms_threshold){
    nms_threshold = input_nms_threshold;
    mmat_objection.inpSize = 320;

    model_name = "/home/pi/model/ab_fp32.mnn";

    net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_name.c_str()));
    if (nullptr == net) {
    }

    config.numThread = 4;
    config.type = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
    backendConfig.precision = MNN::BackendConfig::Precision_Normal;
    backendConfig.memory = MNN::BackendConfig::Memory_Low;
    config.backendConfig = &backendConfig;
    session = net->createSession(config);
}

cv::Mat yolo_fv2_mnn::preprocess_v5lite(cv::Mat &cv_mat) {
    cv::Mat img, dstimg;

    cv::cvtColor(cv_mat, dstimg, cv::COLOR_BGR2RGB);

    mmat_objection.maxSide = cv_mat.rows > cv_mat.cols ? cv_mat.rows : cv_mat.cols;
    mmat_objection.ratio = float(mmat_objection.inpSize) / float(mmat_objection.maxSide);
    int fx = int(cv_mat.cols * mmat_objection.ratio);
    int fy = int(cv_mat.rows * mmat_objection.ratio);
    mmat_objection.Padw = int((mmat_objection.inpSize - fx) * 0.5);
    mmat_objection.Padh = int((mmat_objection.inpSize - fy) * 0.5);
    cv::resize(dstimg, img, cv::Size(fx, fy));
    cv::copyMakeBorder(img, img, mmat_objection.Padh, mmat_objection.Padh, mmat_objection.Padw,
                       mmat_objection.Padw, cv::BORDER_CONSTANT, cv::Scalar::all(127));

    img.convertTo(img, CV_32FC3);
    img = img / 255.0f;

    return img;
}

std::vector<BoxInfo_v5lite> yolo_fv2_mnn::decode_v5lite(cv::Mat &raw_image)
{
    struct timespec begin, end;
    long time;
    clock_gettime(CLOCK_MONOTONIC, &begin);

    cv::Mat cv_mat = preprocess_v5lite(raw_image);

    std::vector<int> dims{1, mmat_objection.inpSize, mmat_objection.inpSize, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data = nhwc_Tensor->host<float>();
    auto nhwc_size = nhwc_Tensor->size();
    std::memcpy(nhwc_data, cv_mat.data, nhwc_size);

    auto inputTensor = net->getSessionInput(session, nullptr);
    inputTensor->copyFromHostTensor(nhwc_Tensor);

    net->runSession(session);
    MNN::Tensor *tensor_scores = net->getSessionOutput(session, "outputs");
    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    tensor_scores->copyToHostTensor(&tensor_scores_host);
    auto pred_dims = tensor_scores_host.shape();

    const unsigned int num_proposals = pred_dims.at(1);
    const unsigned int num_classes = pred_dims.at(2) - 5;
    std::vector<BoxInfo_v5lite> result;

    for (unsigned int i = 0; i < num_proposals; ++i) {
        const float *offset_obj_cls_ptr = tensor_scores_host.host<float>() + (i * (num_classes + 5)); // row ptr
        float obj_conf = offset_obj_cls_ptr[4];
        if (obj_conf < 0.5)
            continue;

        float cls_conf = offset_obj_cls_ptr[5];
        unsigned int label = 0;
        for (unsigned int j = 0; j < num_classes; ++j) {
            float tmp_conf = offset_obj_cls_ptr[j + 5];
            if (tmp_conf > cls_conf) {
                cls_conf = tmp_conf;
                label = j;
            }
        }

        float conf = obj_conf * cls_conf; 
        if (conf < 0.50)
            continue;

        float cx = offset_obj_cls_ptr[0];
        float cy = offset_obj_cls_ptr[1];
        float w = offset_obj_cls_ptr[2];
        float h = offset_obj_cls_ptr[3];

        float x1 = (cx - w / 2.f);
        float y1 = (cy - h / 2.f);
        float x2 = (cx + w / 2.f);
        float y2 = (cy + h / 2.f);

        BoxInfo_v5lite box;
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = std::min(x2, (float)mmat_objection.inpSize - 1.f);
        box.y2 = std::min(y2, (float)mmat_objection.inpSize - 1.f);
        box.score = conf;
        box.label = label;
        result.push_back(box);
    }

    delete nhwc_Tensor;

    nms_v5lite(result);

    draw_box_v5lite(raw_image, result);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec);
    if(time > 0) printf(">> fps : %lf \n", 1000000000 / (double)time);
    return result;
}

void yolo_fv2_mnn::nms_v5lite(std::vector<BoxInfo_v5lite> &input_boxes) {
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo_v5lite a, BoxInfo_v5lite b)
              { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= nms_threshold) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
                j++;
        }
    }
}

void yolo_fv2_mnn::draw_box_v5lite(cv::Mat &cv_mat, std::vector<BoxInfo_v5lite> &boxes) {
    static const char *class_names[] = {"A", "B"};

    char text[256];

    for (auto box :boxes) {
        box.x1 = (box.x1 - mmat_objection.Padw) / mmat_objection.ratio;
        box.x2 = (box.x2 - mmat_objection.Padw) / mmat_objection.ratio;
        box.y1 = (box.y1 - mmat_objection.Padh) / mmat_objection.ratio;
        box.y2 = (box.y2 - mmat_objection.Padh) / mmat_objection.ratio;

        printf("class: %s %f, center_x = %f, center_y = %f\n",class_names[box.label], box.score, (box.x1 + box.x2)/2, (box.y1 + box.y2)/2);
#if display
        cv::Point pos = cv::Point(box.x1, box.y1 - 5);
        cv::Rect rect = cv::Rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
        cv::rectangle(cv_mat, rect, cv::Scalar(0, 255, 0),4);
        sprintf(text, "%s %.1f%%", class_names[box.label], box.score * 100);
        cv::putText(cv_mat, text, pos, cv::FONT_HERSHEY_SIMPLEX, (box.y2 - box.y1) / mmat_objection.maxSide*5,
                    cv::Scalar(0, 0, 255), 4);
#endif
    }
#if display
#if camera
    cv::imshow("Fourcc", cv_mat);
#else
    cv::imwrite("/home/pi/yolo_fv2_mnn/result.jpg", cv_mat);
#endif
#endif
}

// double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt) {
// 	float in = (bb_test & bb_gt).area();
// 	float un = bb_test.area() + bb_gt.area() - in;

// 	if (un < DBL_EPSILON)
// 		return 0;

// 	return (double)(in / un);
// }
