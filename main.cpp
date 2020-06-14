// main.cpp 
// Testing OpenCL WorkgroupSize parameter in enqueueNDRangeKernel() for calculation performance

// TODO: �������� �������� �� ������ ��������� ��������� WorkgroupSize
// TODO: �������� ����� ����������� ���-�� ������ ��������� �� ������ ������������

#pragma comment(lib, "opencl.lib")
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <conio.h>
#include <time.h>
#include <windows.h>
#include <algorithm>
#include <numeric>
#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif 
#include "GQPC_Timer/GQPC_Timer.hpp"

using namespace std;

/// <summary>
/// Struct for storing OpenCL device data for benchmarking.
/// </summary>
struct ClDeviceBenchmarkData {
    unsigned nMaxWorkgoupSize;
    cl::Context* pContext;
    cl::CommandQueue* pCmdQueue;
    cl::Kernel* pKernel;
    cl::Buffer* pInputVec1;
    cl::Buffer* pInputVec2;
    cl::Buffer* pOutputVec;
}; //> struct

// global constants
const unsigned DATA_SIZE = 20 * (1 << 20);                      // about 20M items for benchmark
const unsigned int KERNEL_BENCHMARK_LOOPS = 200;                // number of kernel benchmarks on OpenCL device

                                                      
void getClDevices();                                                // get all OpenCL devices installed in the system
int chooseClDevice();                                               // let user select OpenCL device to benchmark
void printClDeviceInfo(int iDev);                                   // print OpenCL device info
int getNvCudaCoresPerSm(cl_uint verMajor, cl_uint verMinor);        // get number of NVidia CUDA cores per one streaming multiprocessor (SM)
void benchmarkClDevice(int iDev);                                   // do single OpenCL device benchmark (top-level function)
void cleanupClDevices();                                            // cleanup OpenCL devices data
void getBenchTimes(vector<double>& timeValues,
    double& minTime, double& maxTime, double& avgTime);             // get benchmark times
void prepareTestData();                                             // prepare data for benchmark
void prepareClDevices();                                            // prepare OpenCL devices for benchmark
void freeTestData();                                                // free memory allocated for benchmark
string formatMemSizeInfo(cl_ulong ms);                              // format memory size to user-friendly string
string removeMultiSpaces(string s);                                 // remove multiple spaces in string
string getDeviceTypeDescription(cl_device_type dt);                 // get device type string description

// global variables
float* pInputVector1;                                               // benchmark input data 1
float* pInputVector2;                                               // benchmark input data 2
float* pOutputVector;                                               // benchmark output data
vector <cl::Device> clDevices;                                      // list of all OpenCL devices installed in the system
vector <ClDeviceBenchmarkData> clDevicesData;                       // OpenCL devices data used for benchmarking
GQPC_Timer qpc_timer;                                               // high definition timer for benchmark measurements

// main() funcion =========================================================================================================================
int main() {
    // greetings to user
    cout << "OpenCL benchmark: how WorkGroupSize parameter affects calculation performance\n\n";

    // prepare memory and test data    
    prepareTestData();    

    // get all OpenCL devices in the system 
    getClDevices();

    // prepare OpenCL devices
    prepareClDevices();

    // let user choose which OpenCL device to benchmark
    if (clDevices.size() > 0) {
        int iDev;
        while (iDev = chooseClDevice(), iDev > 0) {
            printClDeviceInfo(iDev - 1);
            benchmarkClDevice(iDev - 1);
        } //> while    
    } else {
        // process "no OpenCL devices" situation
        cout << "No OpenCL devices detected. Press any key to exit\n";
        _getch();
    }//> if

    // free test data
    cleanupClDevices();
    freeTestData();

    cout << "bye\n";
    return 0;
} //> main() 
// ========================================================================================================================================

/// <summary>
/// Get list of all OpenCL devices installed in the system.
/// </summary>
void getClDevices() {
    clDevices.clear();

    // get platforms
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // get devices for each platform
    for (int iPlatf = 0; iPlatf < platforms.size(); iPlatf++) {
        vector<cl::Device> devices;
        platforms[iPlatf].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        clDevices.insert(clDevices.end(), devices.begin(), devices.end());
    } //> for
} //> getClDevices()

/// <summary>
/// Let user choose which OpenCL device to benchmark.
/// </summary>
/// <returns>Index of selected OpenCL device.</returns>
int chooseClDevice() {
    if (clDevices.size() == 0) {        
        return -1;
    } else {
        // choose device to benchmark
        cout << "\nList of all OpenCL devices in the system:\n";
        for (int i = 0; i < clDevices.size(); i++) {
            cout << "\t" << i + 1 << " - " << removeMultiSpaces( clDevices[i].getInfo<CL_DEVICE_NAME>() ) << endl;
        } //> for

        cout << "Choose one to benchmark (or esc to exit): ";
        int ch;
        do {
            ch = _getch();
        } while (!((ch == 27) || ((ch > '0') && (ch <= '0' + clDevices.size()))));
        if (ch == 27)
            cout << "esc\n";
        else
            cout << static_cast<char>(ch) << endl;

        // process esc key
        if (ch == 27)
            return 0;

        // process device selection        
        ch -= '0';
        cout << "\nSelected device: " << clDevices[ch - 1].getInfo<CL_DEVICE_NAME>() << endl;
        return ch;
    } //> else > if
} //> chooseClDevice()

/// <summary>
/// Print information about selected OpenCL device.
/// </summary>
/// <param name="iDev">Index of OpenCL device.</param>
void printClDeviceInfo(int iDev) {
    //cout << "Benchmarking " << oclDevice.getInfo<CL_DEVICE_NAME>() << "...\n";
    int nSM = 0;
    int nCores = 0;

    cl::Device dev = clDevices[iDev];

    cout << "Device info:\n";
    cout << "\tVendor: " << dev.getInfo<CL_DEVICE_VENDOR>() << endl;
    cout << "\tType: " << getDeviceTypeDescription(dev.getInfo<CL_DEVICE_TYPE>()) << endl;
    cout << "\tMemory - global: " << formatMemSizeInfo(dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()) << endl;
    cout << "\tMemory - global cache: " << formatMemSizeInfo(dev.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>()) << endl;
    cout << "\tMemory - local: " << formatMemSizeInfo(dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()) << endl;
    cout << "\tMax work group size: " << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
    nSM = dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    cout << "\tCompute units: " << nSM << endl;
    if (dev.getInfo<CL_DEVICE_VENDOR_ID>() == 0x10DE) {
        // NVidia specific info according to https://www.khronos.org/registry/OpenCL/extensions/nv/cl_nv_device_attribute_query.txt
        cl_uint majCompCapab = dev.getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV>();
        cl_uint minCompCapab = dev.getInfo<CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV>();
        cout << "\tCompute capability (NVidia): " << majCompCapab << '.' << minCompCapab << endl;
        // NVidia CUDA cores based on compute capability version (https://stackoverflow.com/questions/32530604/how-can-i-get-number-of-cores-in-cuda-device)
        nCores = getNvCudaCoresPerSm(majCompCapab, minCompCapab);
        cout << "\tCUDA cores per SM (NVidia): " << nCores << endl;
        cout << "\tCUDA cores total (NVidia): " << nCores * nSM << endl;
        //cout << "\tWarp size (NVidia): " << oclDevice.getInfo<CL_DEVICE_WARP_SIZE_NV>() << endl;
    } //> if
    //cout << "\tVendor ID: " << oclDevice.getInfo<CL_DEVICE_VENDOR_ID>() << endl;
    
    cout << endl;
} //> printClDeviceInfo()

/// <summary>
/// Get number of NVidia CUDA cores on specific GPU.
/// </summary>
/// <param name="verMajor">Major version.</param>
/// <param name="verMinor">Minor version.</param>
/// <returns></returns>
int getNvCudaCoresPerSm(cl_uint verMajor, cl_uint verMinor) {
    int nCores = 0;
    switch (verMajor) {
    case 2: // Fermi
        if (verMinor == 1) 
            nCores = 48;
        else 
            nCores = 32;
        break;
    case 3: // Kepler
        nCores = 192;
        break;
    case 5: // Maxwell
        nCores = 128;
        break;
    case 6: // Pascal
        if ((verMinor == 1) || (verMinor == 2))
            nCores = 128;
        else
            if (verMinor == 0)
                nCores = 64;
            else
                nCores = -1;
        break;
    case 7: // Volta and Turing
        if ((verMinor == 0) || (verMinor == 5)) 
            nCores = 64;
        else 
            nCores = -1;
        break;
    default:
        nCores = -1;
        break;
    }
    return nCores;
} //> getNvCudaCoresPerSm()

/// <summary>
/// Benchmark selected OpenCL device.
/// </summary>
/// <param name="iDev">Index of selected OpenCL device to benchmark.</param>
void benchmarkClDevice(int iDev) {
    unsigned nWorkgroupSize = clDevicesData[iDev].nMaxWorkgoupSize;
    cout << "\tRunning benchmark on workgroup size: " << nWorkgroupSize << "...";

    // prepare OpenCL devic for benchmarks
    //prepareClDeviceForBenchmark(oclDevice);

    // prepare to performance measurement
    vector<double> timeValues;          // time values for current bench
    timeValues.clear();

    // run the kernel on specific ND range
    qpc_timer.reset();
    for (int iBench = 0; iBench < KERNEL_BENCHMARK_LOOPS; iBench++) {
        clDevicesData[iDev].pCmdQueue->enqueueNDRangeKernel(*clDevicesData[iDev].pKernel, cl::NullRange, cl::NDRange(DATA_SIZE), cl::NDRange(nWorkgroupSize));
        clDevicesData[iDev].pCmdQueue->finish();

        double timeMs = qpc_timer.resetMs();
        timeValues.push_back(timeMs);

        // read output buffer into a host
        //pQueue->enqueueReadBuffer(oclOutputVector, CL_TRUE, 0, DATA_SIZE * sizeof(float), pOutputVector);
    } //> for

    // cleanup after benchmark
    //cleanupClDeviceAfterBenchmark();

    // print benchmark times
    cout << "done\n";
    double minTime, maxTime, avgTime;
    getBenchTimes(timeValues, minTime, maxTime, avgTime);
    cout << "\tTimes, ms (avg, min, max): " << avgTime << ", " << minTime << ", " << maxTime << endl;
} //> benchmarkClDevice()

/// <summary>
/// Free memory that was allocated for OpenCL devices data.
/// </summary>
void cleanupClDevices() {
    for (int i = 0; i < clDevicesData.size(); i++) {
        if (clDevicesData[i].pKernel)
            delete clDevicesData[i].pKernel, clDevicesData[i].pKernel = nullptr;
        if (clDevicesData[i].pOutputVec)
            delete clDevicesData[i].pOutputVec, clDevicesData[i].pOutputVec = nullptr;
        if (clDevicesData[i].pInputVec2)
            delete clDevicesData[i].pInputVec2, clDevicesData[i].pInputVec2 = nullptr;
        if (clDevicesData[i].pInputVec1)
            delete clDevicesData[i].pInputVec1, clDevicesData[i].pInputVec1 = nullptr;
        if (clDevicesData[i].pCmdQueue)
            delete clDevicesData[i].pCmdQueue, clDevicesData[i].pCmdQueue = nullptr;
        if (clDevicesData[i].pContext)
            delete clDevicesData[i].pContext, clDevicesData[i].pContext = nullptr;
    } //> for
} //> cleanupClDevices()

/// <summary>
/// Calculate time statistics for benchmark.
/// </summary>
/// <param name="timeValues">Input time values.</param>
/// <param name="minTime">Output minimum time.</param>
/// <param name="maxTime">Output maximum time.</param>
/// <param name="avgTime">Output average time.</param>
void getBenchTimes(vector<double> &timeValues, double &minTime, double &maxTime, double &avgTime) {
    sort(timeValues.begin(), timeValues.end());
    double totalTime = accumulate(timeValues.begin(), timeValues.end(), 0.0);
    avgTime = totalTime / timeValues.size();
    minTime = timeValues[0];
    maxTime = timeValues[timeValues.size() - 1];
    //double medianTime = timeValues[timeValues.size() / 2];
    //cout << "\t\tBench info: " << timeValues.size() << " runs, each on " << DATA_SIZE << " items" << endl;
    //cout << "\t\tAvg: " << averageTime << " ms" << endl;
    ////cout << "Avg: " << averageTime << " ms" << endl;
    //cout << "\t\tMin: " << minTime << " ms" << endl;
    //cout << "\t\tMax: " << maxTime << " ms" << endl << endl;
} //> printBenchTimes()

/// <summary>
/// Allocate memory and fill values for benchmark.
/// </summary>
void prepareTestData() {
    cout << "Preparing test data for benchmark...";
    // TODO: ����������� ������� ����� vector
    ::pInputVector1 = new float[DATA_SIZE];
    ::pInputVector2 = new float[DATA_SIZE];
    ::pOutputVector = new float[DATA_SIZE];

    // fill input vectors with random values
    srand(static_cast<unsigned int>(time(NULL)));
    for (int i = 0; i < DATA_SIZE; i++) {
        ::pInputVector1[i] = static_cast<float>(rand() * 1000.0 / RAND_MAX);
        ::pInputVector2[i] = static_cast<float>(rand() * 1000.0 / RAND_MAX);
    } //> for

    // clear output vector
    fill_n(pOutputVector, DATA_SIZE, static_cast<float>(0));
    cout << "done\n";
} //> prepareTestData()

/// <summary>
/// Prepare all OpenCL devices for benchmarks.
/// </summary>
void prepareClDevices() {
    cout << "Preparing OpenCL devices...";

    // read kernel source file
    ifstream sourceFile("oclFile.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, make_pair(sourceCode.c_str(), sourceCode.length() + 1));

    // iterate through each OpenCL device and prepare data for benchmark
    clDevicesData.clear();
    for (int i = 0; i < clDevices.size(); i++ ) {
        cl::Device dev = clDevices[i];
        ClDeviceBenchmarkData clDevData;
        clDevData.nMaxWorkgoupSize = static_cast<unsigned>(dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());

        // create context for device
        vector<cl::Device> contextDevices;
        contextDevices.push_back(dev);
        clDevData.pContext = new cl::Context(contextDevices);

        // create command queue for device
        clDevData.pCmdQueue = new cl::CommandQueue(*clDevData.pContext, dev);

        // create memory buffers for device
        clDevData.pInputVec1 = new cl::Buffer(*clDevData.pContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), ::pInputVector1);
        clDevData.pInputVec2 = new cl::Buffer(*clDevData.pContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), ::pInputVector2);
        clDevData.pOutputVec = new cl::Buffer(*clDevData.pContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), ::pOutputVector);

        // create kernel from source file
        cl::Program program = cl::Program(*clDevData.pContext, source);
        program.build(contextDevices);
        clDevData.pKernel = new cl::Kernel(program, "TestKernel");

        // set arguments to kernel
        int iArg = 0;
        clDevData.pKernel->setArg(iArg++, *clDevData.pInputVec1);
        clDevData.pKernel->setArg(iArg++, *clDevData.pInputVec2);
        clDevData.pKernel->setArg(iArg++, *clDevData.pOutputVec);
        clDevData.pKernel->setArg(iArg++, DATA_SIZE);

        // save data for current device
        clDevicesData.push_back(clDevData);
    } //> for

    cout << "done\n";
} //> prepareClDevices()

/// <summary>
/// Free test data memory.
/// </summary>
void freeTestData() {
    delete[] ::pInputVector1;
    delete[] ::pInputVector2;
    delete[] ::pOutputVector;
} //> freeTestData()

/// <summary>
/// Format text of memory size to user-friendly form.
/// </summary>
/// <param name="ms"></param>
/// <returns></returns>
string formatMemSizeInfo(cl_ulong ms) {
    int divides = 0;
    while (ms > 1024)
        ms /= 1024, ++divides;
    string retval = to_string(ms) + ' ';
    switch (divides) {
    case 1: retval += "KiB"; break;
    case 2: retval += "MiB"; break;
    case 3: retval += "GiB"; break;
    case 4: retval += "TiB"; break;
    case 5: retval += "PiB"; break;
    case 6: retval += "EiB"; break;
    case 7: retval += "EiB"; break;
    case 8: retval += "ZiB"; break;
    case 9: retval += "YiB"; break;
    default: retval += "???";  break;
    } //> switch
    return retval;
} //> formatMemSizeInfo()

/// <summary>
/// Remove multiple spaces from text.
/// </summary>
/// <param name="s">Input text to be processed. The value is being changed itself.</param>
/// <returns>Output processed text.</returns>
string removeMultiSpaces(string s) {
    size_t p = 0;
    while (p = s.find("  ", p), p != string::npos) {
        s.erase(p, 1);
    } //> while
    return s;
} //> removeMultiSpaces()

/// <summary>
/// Get text representation of OpenCL device type.
/// </summary>
/// <param name="dt">Input device type.</param>
/// <returns>Text representation of device type.</returns>
string getDeviceTypeDescription(cl_device_type dt) {
    switch (dt) {
    case CL_DEVICE_TYPE_CPU:
        return "CPU";
    case CL_DEVICE_TYPE_GPU:
        return "GPU";
    case CL_DEVICE_TYPE_ACCELERATOR:
        return "Accelerator";
    case CL_DEVICE_TYPE_DEFAULT:
        return "Default";
    } //> switch
    return "";
} //> getDeviceTypeDescription()
