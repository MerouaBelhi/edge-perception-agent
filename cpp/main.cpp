// Edge Perception Agent — C++ inference engine
//
// Runs a YOLOv8 ONNX model with the ONNX Runtime C++ API on CPU: letterbox
// pre-processing, output decoding, class-aware NMS, drawing, and per-stage
// latency measurement. Same algorithm as src/epa/detector_onnx.py,
// re-implemented with no Python. OpenCV is used only for image I/O and NMS.
//
// Usage:
//   ./epa_infer <model.onnx> <image|video|camera_index> [out_path] [conf] [iou]
//   ./epa_infer models/site11hh_v2_int8.onnx path/to/site.jpg results/cpp_site.jpg
//   ./epa_infer models/site11hh_v2_int8.onnx path/to/site.mp4 results/cpp_site.mp4

#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <memory>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

const std::vector<std::string> kCoco = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"};

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Detection {
    cv::Rect2f box;
    float score;
    int cls;
};

struct Letterbox {
    cv::Mat image;
    float scale;
    int pad_x, pad_y;
};

Letterbox letterbox(const cv::Mat& img, int size) {
    float scale = std::min(static_cast<float>(size) / img.rows,
                           static_cast<float>(size) / img.cols);
    int nw = static_cast<int>(std::round(img.cols * scale));
    int nh = static_cast<int>(std::round(img.rows * scale));
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    Letterbox lb{cv::Mat(size, size, CV_8UC3, cv::Scalar(114, 114, 114)), scale,
                 (size - nw) / 2, (size - nh) / 2};
    resized.copyTo(lb.image(cv::Rect(lb.pad_x, lb.pad_y, nw, nh)));
    return lb;
}

class YoloDetector {
public:
    YoloDetector(const std::string& model, float conf, float iou, int threads = 0)
        : env_(ORT_LOGGING_LEVEL_WARNING, "epa"), conf_(conf), iou_(iou) {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (threads > 0) opts.SetIntraOpNumThreads(threads);
        // ORTCHAR_T is wchar_t on Windows and char elsewhere: widen the path portably
        // (keep the project in a folder whose path has only ASCII characters)
        const std::basic_string<ORTCHAR_T> model_path(model.begin(), model.end());
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
        Ort::AllocatorWithDefaultOptions alloc;
        in_name_ = session_->GetInputNameAllocated(0, alloc).get();
        out_name_ = session_->GetOutputNameAllocated(0, alloc).get();

        // Input size from the model itself (e.g. 640 for COCO, 416 for the PPE model)
        auto in_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        size_ = (in_shape.size() == 4 && in_shape[2] > 0) ? static_cast<int>(in_shape[2]) : 640;

        // Class names: Ultralytics writes them into the ONNX metadata as
        //   names = "{0: 'person', 1: 'helmet', 2: 'no_helmet'}"
        Ort::ModelMetadata meta = session_->GetModelMetadata();
        auto raw = meta.LookupCustomMetadataMapAllocated("names", alloc);
        if (raw) {
            const std::string s = raw.get();
            for (size_t i = s.find('\''); i != std::string::npos; i = s.find('\'', i)) {
                size_t j = s.find('\'', i + 1);
                if (j == std::string::npos) break;
                names_.push_back(s.substr(i + 1, j - i - 1));
                i = j + 1;
            }
        }
        if (names_.empty()) names_ = kCoco;
    }

    int input_size() const { return size_; }
    const std::string& name(int cls) const {
        static const std::string unknown = "?";
        return cls >= 0 && cls < static_cast<int>(names_.size()) ? names_[cls] : unknown;
    }

    std::vector<Detection> detect(const cv::Mat& frame, double t[3]) {
        auto t0 = Clock::now();
        Letterbox lb = letterbox(frame, size_);
        // HWC uint8 BGR -> NCHW float RGB in [0,1]
        cv::Mat blob = cv::dnn::blobFromImage(lb.image, 1.0 / 255.0, cv::Size(size_, size_),
                                              cv::Scalar(), /*swapRB=*/true, /*crop=*/false);
        t[0] = ms_since(t0);

        auto t1 = Clock::now();
        std::array<int64_t, 4> in_shape{1, 3, size_, size_};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            mem, blob.ptr<float>(), blob.total(), in_shape.data(), in_shape.size());
        const char* in_names[] = {in_name_.c_str()};
        const char* out_names[] = {out_name_.c_str()};
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
        t[1] = ms_since(t1);

        auto t2 = Clock::now();
        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();   // (1, 84, 8400)
        const int rows = static_cast<int>(shape[1]);     // 4 + num_classes
        const int anchors = static_cast<int>(shape[2]);  // 8400
        float* out_data = outputs[0].GetTensorMutableData<float>();
        cv::Mat preds(rows, anchors, CV_32F, out_data);
        cv::Mat predsT = preds.t();             // (8400, 84)

        std::vector<cv::Rect> nms_boxes;
        std::vector<cv::Rect2f> boxes;
        std::vector<float> scores;
        std::vector<int> classes;
        for (int i = 0; i < anchors; ++i) {
            const float* r = predsT.ptr<float>(i);
            cv::Mat cls_scores(1, rows - 4, CV_32F, const_cast<float*>(r + 4));
            cv::Point max_loc;
            double max_score;
            cv::minMaxLoc(cls_scores, nullptr, &max_score, nullptr, &max_loc);
            if (max_score < conf_) continue;
            float cx = r[0], cy = r[1], w = r[2], h = r[3];
            float x1 = (cx - w / 2 - lb.pad_x) / lb.scale;
            float y1 = (cy - h / 2 - lb.pad_y) / lb.scale;
            float bw = w / lb.scale, bh = h / lb.scale;
            x1 = std::clamp(x1, 0.f, static_cast<float>(frame.cols));
            y1 = std::clamp(y1, 0.f, static_cast<float>(frame.rows));
            bw = std::min(bw, frame.cols - x1);
            bh = std::min(bh, frame.rows - y1);
            boxes.emplace_back(x1, y1, bw, bh);
            // class-aware NMS: shift each class into its own region of the plane
            int off = max_loc.x * 4096;
            nms_boxes.emplace_back(static_cast<int>(x1) + off, static_cast<int>(y1),
                                   static_cast<int>(bw), static_cast<int>(bh));
            scores.push_back(static_cast<float>(max_score));
            classes.push_back(max_loc.x);
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(nms_boxes, scores, conf_, iou_, keep);
        std::vector<Detection> dets;
        for (int k : keep) dets.push_back({boxes[k], scores[k], classes[k]});
        t[2] = ms_since(t2);
        return dets;
    }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::string in_name_, out_name_;
    std::vector<std::string> names_;
    int size_ = 640;
    float conf_, iou_;
};

void draw(cv::Mat& img, const std::vector<Detection>& dets, const YoloDetector& det, double fps) {
    for (const auto& d : dets) {
        cv::Scalar col((d.cls * 67) % 255, (d.cls * 131 + 80) % 255, (d.cls * 29 + 160) % 255);
        cv::rectangle(img, d.box, col, 2);
        char label[64];
        std::snprintf(label, sizeof(label), "%s %.2f",
                      det.name(d.cls).c_str(), d.score);
        int base = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        cv::Point tl(static_cast<int>(d.box.x), static_cast<int>(d.box.y));
        cv::rectangle(img, tl - cv::Point(0, ts.height + 6), tl + cv::Point(ts.width + 4, 0), col,
                      cv::FILLED);
        cv::putText(img, label, tl + cv::Point(2, -4), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    char hud[64];
    std::snprintf(hud, sizeof(hud), "C++ ONNX Runtime | FPS %.1f", fps);
    cv::rectangle(img, cv::Point(8, 8), cv::Point(290, 38), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(img, hud, cv::Point(16, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

bool is_image(const std::string& p) {
    std::string s = p;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    for (const char* ext : {".jpg", ".jpeg", ".png", ".bmp"})
        if (s.size() >= std::strlen(ext) && s.compare(s.size() - std::strlen(ext), std::string::npos, ext) == 0)
            return true;
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> <image|video|cam_index> [out_path] [conf=0.3] [iou=0.45]\n";
        return 1;
    }
    const std::string model = argv[1], source = argv[2];
    const std::string out_path = argc > 3 ? argv[3] : "";
    const float conf = argc > 4 ? std::stof(argv[4]) : 0.3f;
    const float iou = argc > 5 ? std::stof(argv[5]) : 0.45f;

    YoloDetector det(model, conf, iou);
    std::printf("model: %s | input %dx%d\n", model.c_str(), det.input_size(), det.input_size());
    double t[3];

    if (is_image(source)) {
        cv::Mat img = cv::imread(source);
        if (img.empty()) { std::cerr << "Cannot read " << source << "\n"; return 1; }
        for (int i = 0; i < 3; ++i) det.detect(img, t);   // warm-up
        std::vector<double> total;
        std::vector<Detection> dets;
        for (int i = 0; i < 20; ++i) {
            dets = det.detect(img, t);
            total.push_back(t[0] + t[1] + t[2]);
        }
        double mean = std::accumulate(total.begin(), total.end(), 0.0) / total.size();
        std::printf("detections: %zu\n", dets.size());
        for (const auto& d : dets)
            std::printf("  %-10s %.2f  [%d %d %d %d]\n", det.name(d.cls).c_str(), d.score,
                        (int)d.box.x, (int)d.box.y, (int)(d.box.x + d.box.width),
                        (int)(d.box.y + d.box.height));
        std::printf("latency (last run): pre %.1f ms | infer %.1f ms | post %.1f ms\n",
                    t[0], t[1], t[2]);
        std::printf("mean end-to-end over 20 runs: %.1f ms (%.1f FPS)\n", mean, 1000.0 / mean);
        if (!out_path.empty()) {
            draw(img, dets, det, 1000.0 / mean);
            cv::imwrite(out_path, img);
        }
        return 0;
    }

    cv::VideoCapture cap;
    if (std::all_of(source.begin(), source.end(), ::isdigit)) cap.open(std::stoi(source));
    else cap.open(source);
    if (!cap.isOpened()) { std::cerr << "Cannot open " << source << "\n"; return 1; }
    const double src_fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 25.0;
    cv::VideoWriter writer;
    cv::Mat frame;
    std::vector<double> lat;
    double ema = 0;
    while (cap.read(frame)) {
        auto dets = det.detect(frame, t);
        double ms = t[0] + t[1] + t[2];
        lat.push_back(ms);
        ema = ema == 0 ? 1000.0 / ms : 0.9 * ema + 0.1 * (1000.0 / ms);
        if (!out_path.empty()) {
            if (!writer.isOpened())
                writer.open(out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), src_fps,
                            frame.size());
            draw(frame, dets, det, ema);
            writer.write(frame);
        }
    }
    if (lat.empty()) return 1;
    std::sort(lat.begin(), lat.end());
    double mean = std::accumulate(lat.begin(), lat.end(), 0.0) / lat.size();
    std::printf("frames: %zu | mean %.1f ms | p50 %.1f ms | p95 %.1f ms | %.1f FPS\n", lat.size(),
                mean, lat[lat.size() / 2], lat[static_cast<size_t>(lat.size() * 0.95)],
                1000.0 / mean);
    return 0;
}
