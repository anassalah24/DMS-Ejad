#include <opencv2/opencv.hpp>
#include "threadsafequeue.h"
#include "dmsmanager.h"
#include <benchmark/benchmark.h>


int main() {


    // Initialize thread-safe queues needed for each component
    ThreadSafeQueue<cv::Mat> cameraQueue;
    ThreadSafeQueue<cv::Mat> preprocessingQueue;
    ThreadSafeQueue<cv::Mat> faceDetectionQueue;
    ThreadSafeQueue<cv::Mat> tcpOutputQueue;     
    ThreadSafeQueue<CarState> stateOutputQueue;
    ThreadSafeQueue<int> postOutputQueue;
    ThreadSafeQueue<std::string> commandsQueue;
    ThreadSafeQueue<std::string> faultsQueue;

    int tcpPort = 12345;  // Define the TCP port for the server

    // Initialize the DMSManager with all necessary queues and the TCP port
    DMSManager dmsManager(cameraQueue, preprocessingQueue, faceDetectionQueue, tcpOutputQueue, tcpPort, stateOutputQueue, postOutputQueue,commandsQueue,faultsQueue);

    // Initialize the camera component
    if (!dmsManager.initializeCamera("/dev/video0")) { // use /dev/video0 for camera or /path/to/video/file
        std::cerr << "Failed to initialize camera component." << std::endl;
        return -1;
    }

    // Initialize the face detection component with weights and configurations
    if (!dmsManager.initializeFaceDetection("/home/dms/DMS-main/ModularCode/modelconfigs/face-yolov3-tiny.cfg", "/home/dms/DMS-main/ModularCode/modelconfigs/face-yolov3-tiny_41000.weights")) {
        std::cerr << "Failed to initialize face detection component." << std::endl;
        return -1;
    }


    //******************missing initialization for the vehicle state to pass the file to read from***************************
    //-----------------------------------------------------------------------------------------------------------------------------


    // Start the system
    if (!dmsManager.startSystem()) {
        std::cerr << "Failed to start the system." << std::endl;
        return -1;
    }


    // Main loop (The display code is commented out; uncomment if needed)

    
    // MAT objects to carry frames incase we need to show frames of some output from component while development
    cv::Mat cameraFrame, preprocessedFrame, faceDetectedFrame;
    while (true) {
        //if (cameraQueue.tryPop(cameraFrame) && !cameraFrame.empty()) {
        //   cv::imshow("Camera Frame", cameraFrame);
        //}

        //if (preprocessingQueue.tryPop(preprocessedFrame) && !preprocessedFrame.empty()) {
        //    cv::imshow("Processed Frame", preprocessedFrame);
        //}

        //if (faceDetectionQueue.tryPop(faceDetectedFrame) && !faceDetectedFrame.empty()) {
        //   cv::imshow("Face Detection", faceDetectedFrame);
        //}

        //if (cv::waitKey(1) == 27) break;  // Exit on ESC key
    }

    // Shutdown the system
    dmsManager.stopSystem();
    std::cout << "System stopped." << std::endl;


    return 0;
}

