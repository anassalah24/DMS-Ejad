#include <thread>
#include "threadsafequeue.h"
#include "basiccameracomponent.h"
#include "basicpreprocessingcomponent.h"
#include "facedetectioncomponent.h"
#include "commtcpcomponent.h"
#include "vehiclestatemanager.h"
#include "postprocessingcomponent.h"

class DMSManager {
public:
    DMSManager(ThreadSafeQueue<cv::Mat>& cameraQueue, ThreadSafeQueue<cv::Mat>& preprocessingQueue, ThreadSafeQueue<cv::Mat>& faceDetectionQueue,ThreadSafeQueue<cv::Mat>& tcpOutputQueue, int tcpPort, ThreadSafeQueue<CarState>& stateOutputQueue , ThreadSafeQueue<int>& postOutputQueue,ThreadSafeQueue<std::string>& commandsQueue);
    ~DMSManager();

    bool startSystem();
    void stopSystem();
    bool initializeCamera(const std::string& source);
    bool initializeFaceDetection(const std::string& modelConfiguration, const std::string& modelWeights);
    //bool initializeVehicleState(const std::string& steeringFilePath, const std::string& velocityFilePath);
    bool initializePostProcessing();
    void setCameraFPS(int fps);
    void setFaceFDT(int fdt);


    // function to handle the different types of commands send by the windows app
    void handlecommand(std::string& command);

    

private:
    BasicCameraComponent cameraComponent;
    BasicPreprocessingComponent preprocessingComponent;
    FaceDetectionComponent faceDetectionComponent;
    CommTCPComponent tcpComponent; 
    VehicleStateManager vehicleStateManager;
    PostProcessingComponent postProcessingComponent;

    ThreadSafeQueue<cv::Mat>& cameraQueue;
    ThreadSafeQueue<cv::Mat>& preprocessingQueue;
    ThreadSafeQueue<cv::Mat>& faceDetectionQueue;
    ThreadSafeQueue<cv::Mat> tcpOutputQueue;
    ThreadSafeQueue<CarState>& stateOutputQueue;
    ThreadSafeQueue<int>& postOutputQueue;



    ThreadSafeQueue<std::string>& commandsQueue;


    std::thread cameraThread;
    std::thread preprocessingThread;
    std::thread faceDetectionThread;
    std::thread tcpThread; 
    std::thread vehicleStateThread;
    std::thread postProcessingThread; 
    std::thread commandsThread; 



    int tcpPort; 
    bool running;

    void cameraLoop();
    void preprocessingLoop();
    void faceDetectionLoop();
    void commtcpLoop(); 
    void vehicleStateLoop(); 
    void postprocessingLoop(); 
    void commandsLoop();

};

