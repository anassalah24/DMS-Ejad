#include "facedetectioncomponent.h"


//constructor
FaceDetectionComponent::FaceDetectionComponent(ThreadSafeQueue<cv::Mat>& inputQueue, ThreadSafeQueue<cv::Mat>& outputQueue)
: inputQueue(inputQueue), outputQueue(outputQueue), running(false) {}


//destructor
FaceDetectionComponent::~FaceDetectionComponent() {
    stopDetection();
}

// initialize model , choose backend ( CUDA , OPENCV , OPENCL ) 
bool FaceDetectionComponent::initialize(const std::string& modelConfiguration, const std::string& modelWeights) {
    net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);// CUDA , OPENCV
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);// DNN_TARGET_OPENCL , DNN_TARGET_CUDA , DNN_TARGET_CPU 
    if (net.empty()) {
        std::cerr << "Failed to load the model or config file." << std::endl;
        return false;
    }
    return true;
}

// start detection loop in another thread
void FaceDetectionComponent::startDetection() {
    if (running) {
        std::cerr << "Detection is already running." << std::endl;
        return;
    }
    running = true;
    detectionThread = std::thread(&FaceDetectionComponent::detectionLoop, this);
}

// release thread and any needed cleanup
void FaceDetectionComponent::stopDetection() {
    running = false;
    if (detectionThread.joinable()) {
        detectionThread.join();
    }
}


// This loop takes frame from input queue , sends it to detect faces and places it into the output queue
void FaceDetectionComponent::detectionLoop() {
    cv::Mat frame;
    this->fdt = 80;
    while (running) {
        if (inputQueue.tryPop(frame)) {
            detectFaces(frame);
            outputQueue.push(frame);
        }
    }
}


// main YOLO detection algorithm
void FaceDetectionComponent::detectFaces(cv::Mat& frame) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(320, 320), cv::Scalar(0, 0, 0), true, false);// original input size is 416,416
    net.setInput(blob);

    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    float confThreshold = (this->fdt)/100;// Confidence threshold
    for (const auto& out : outs) {
        for (int i = 0; i < out.rows; ++i) {
            const float* detection = out.ptr<float>(i);
            float confidence = detection[4];
            if (confidence > confThreshold) {
                int centerX = static_cast<int>(detection[0] * frame.cols);
                int centerY = static_cast<int>(detection[1] * frame.rows);
                int width = static_cast<int>(detection[2] * frame.cols);
                int height = static_cast<int>(detection[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
		

		// draw rectangle on detected face
                cv::rectangle(frame, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 3);
            }
        }
    }
}



void FaceDetectionComponent::setFDT(int fdt) {
    this->fdt = fdt;
    std::cout << "FDT CHANGED SUCCESSFULLY" << std::endl;
}


