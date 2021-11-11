#pragma once
#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <memory>
//#include <opencv\highgui.h>
#include <CL\cl.h>
#include <fstream>

using namespace std;
//using namespace cv;





struct FeatureMaps {
  vector<shared_ptr<cl_mem>> buffers;
  int width;
  int height;
};

class Neuron {
protected:
  //const stuff
  const cl_context      &context;
  const cl_command_queue &commandQueue;
  const cl_kernel &kernel_conv;
  const cl_kernel &kernel_pool;
  const cl_kernel &kernel_FC;
public:
  Neuron(const cl_context &context, const cl_command_queue &commandQueue, const cl_kernel &kernel_conv, const cl_kernel &kernel_pool, const cl_kernel &kernel_FC);
};

class CNeuron : public Neuron {
private:

  int            kernelWidth;
  int            nNeurons;
  int            nKernels;
  int            outImgsize;
  int            inImgsize;
  vector<float*> kernelsData;

  cl_mem kernelBuf;
  vector<float*> poolBias;
  cl_mem Bias;

public:
  CNeuron(vector<float*>kernelsdata, vector<float*> poolbias, int kernelWidth, const cl_context &context, const cl_command_queue &commandQueue, 
	  const cl_kernel &kernel_conv, const cl_kernel &kernel_pool, const cl_kernel &kernel_FC, int nNeurons, int nKernels, int inimgsize, int outimgsize);
  int kernelsize() { return kernelWidth; };

  void convolve(cl_mem & inFMaps);

  void setKernels(vector<float*>kernelsData, int width, int nNeurons, int nKernels);
  void setpoolBias(vector<float*>poolbias, int nNeurons);
  vector<float*> kernel() { return kernelsData; };
  cl_mem featureBuf;
};

/*class PNeuron : public Neuron {
private:

  vector<float*> poolBias;
  cl_mem Bias;
public:
  PNeuron(vector<float*> poolBias, const cl_context &context, const cl_command_queue &commandQueue, const cl_kernel &kernel_conv, const cl_kernel &kernel_pool, int nNeurons);

  shared_ptr<cl_mem> PNeuron::pool(const shared_ptr<cl_mem> buffer, int outWidth, int outHeight, float poolCoef);
  void setpoolBias(vector<float*>poolbias, int nNeurons);
};
*/


class Layer {
protected:
  FeatureMaps featureMaps;
public:
  ~Layer();
  Layer();
  FeatureMaps getFeatureMaps() { return featureMaps; };
};

class ILayer : public Layer {
private:
  cl_mem featureBuf;
public:
  ILayer(int imgsize, const cl_context &context);
  void activate(float* inImage, const cl_context &context, const cl_command_queue &commandQueue, int imagesize);
  cl_mem & getFeature() { return featureBuf; };
};

class OLayer : public Layer {
public:
  OLayer();
  void activate(cl_mem & getFeatureMaps);
};

class HiddenLayer : public Layer {
protected:
public:
  virtual ~HiddenLayer();
  HiddenLayer();
  virtual void activate(cl_mem & prevFeatureMaps);
};

class CLayer : public HiddenLayer {
protected:
  vector<shared_ptr<CNeuron>> neurons;
public:
  CLayer(vector<shared_ptr<CNeuron>> neurons);
  void activate(cl_mem  & prevFeatureMaps) override;
  cl_mem & getFeature() { return neurons[0].get()->featureBuf; };
};

class PLayer : public HiddenLayer {
private:
	const cl_context      &context;
	const cl_command_queue &commandQueue;
	const cl_kernel &kernel_pool;
	int nNeurons;
	int Imgsize;
	cl_mem featureBuf;
public:
	PLayer(const cl_context &context, const cl_command_queue &commandQueue, const cl_kernel &kernel_pool, int nNeurons, int imgsize);
	void activate(cl_mem & prevFeatureMaps) override;
	void pool(cl_mem & inFMaps);
	cl_mem & getFeature() { return featureBuf; };
};
/*
class CCLayer : public HiddenLayer {
protected:
	vector<shared_ptr<CNeuron>> neurons;
	float poolCoef;
public:
	CCLayer(vector<shared_ptr<CNeuron>> neurons);
	void activate(FeatureMaps prevFeatureMaps) override;
};
*/
void openclRetTackle(cl_int retValue, char* processInfo);
void printCLDeviceInfo();
int openclInit();
int openclCreateKernelFromFile(cl_program* cpProgram, cl_kernel* clKernel, const char* clFileName, const char* kernelName, int flag);
int init_cl();
int ReverseInt(int i);
void ReadMNIST(int NumberOfImages, int DataOfAnImage, float *arr, string name);
void ReadMNIST_Label(int NumberOfImages, float *arr, string name);
int findIndex(float* p);
void prepareCNeurons(int nNeurons, int nKernels, int kernelWidth, int outimgsize, int inimgsize, string filePath_conv, string filePath_bias, vector<shared_ptr<CNeuron>> &cns);
//void preparePNeurons(int nNeurons, string filePath, vector<shared_ptr<PNeuron>> &pns);

