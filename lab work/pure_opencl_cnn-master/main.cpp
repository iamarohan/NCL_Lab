#define __CL_ENABLE_EXCEPTIONS

#include "layer.h"

//#include <opencv\highgui.h>
//#include<opencv2/core/core.hpp>
//#include<opencv2/highgui/highgui.hpp>
#include <CL\cl.hpp>

#include <iostream>
#include <fstream>
#include <ctime>
#include <vector>

using namespace std;
//using namespace cv;

extern cl_context       cxGPUContext;
extern cl_command_queue cqCommandQueue;
extern cl_kernel kernel_pool;
/*
shared_ptr<cl_mem> create() {
	float * a = new float[10];
	for (int i = 0; i != 10; ++i) {
		a[i] = i;
	}
	cl_mem  p = clCreateBuffer(cxGPUContext, CL_MEM_COPY_HOST_PTR, sizeof(float) * 10, a, NULL);
	delete[] a;
	return(make_shared<cl_mem>(p));
}
*/
int main(int argc, char** argv)
{
	int test_image = 10000;
	float *inImage = new float[28 * 28];
	float *output = new float[10];
	float *true_out = new float[10];
	int inImgWidth = 28;
	int inImgHeight = 28;

	float *data = new float[10000 * 28 * 28];
	float *label = new float[10000 * 10];

	//input data
	ReadMNIST(test_image, inImgHeight, data, "t10k-images.idx3-ubyte");
	ReadMNIST_Label(test_image, label, "t10k-labels.idx1-ubyte");
/*	std::ofstream fout("mnist.txt");
	if (!fout)
	{
		std::cout << "文件不能打开" << std::endl;
	}
	else
	{
		for (int i = 0; i != 28; ++i) {
			for (int j = 0; j != 28; ++j) {
				fout << data[28*28*3 + i * 28 + j] << " ";
				}
			fout << std::endl;
		}
		fout.close();
	}
  */
  //initial opencl
   int ans = init_cl();
  //FeatureMaps P;

 // P.buffers.push_back(create());
 // float *b = new float[10];
 // clEnqueueReadBuffer(cqCommandQueue, *P.buffers[0].get(), CL_TRUE, 0, sizeof(float) * 10, b, 0, NULL, NULL);
 // for (int i = 0; i != 10; ++i) {
//	  cout << b[i] << " ";
 // }


  cout << "CNN layers are preparing...\n";
  //0 conv layer neurons
  int kernelWidth0 = 5;
  vector<shared_ptr<CNeuron>> cns0;
  //prepareCNeurons(2, 1, 5, 24, 28, "conv1.txt", "conv1_bias.txt", cns0);
  prepareCNeurons(20, 1, 5, 24, 28, "conv1.txt","conv1_bias.txt", cns0);
 
  //1 conv layer neurons
  int kernelWidth1 = 5; 
  vector<shared_ptr<CNeuron>> cns1;
 // prepareCNeurons(5, 2, 5, 8, 12, "conv2.txt", "conv2_bias.txt", cns1);
  prepareCNeurons(50, 20, 5, 8, 12, "conv2.txt", "conv2_bias.txt", cns1);

  //2 conv layer neurons (22 layer in matlab code)
  int kernelWidth2 = 1;
  vector<shared_ptr<CNeuron>> cns2;
 // prepareCNeurons(5, 80, 1, 1, 4, "ip1.txt","ip1_bias.txt",cns2);
  prepareCNeurons(500, 800, 1, 1, 4, "ip1.txt","ip1_bias.txt", cns2);

  //3 (Out) conv layer neurons
  int kernelWidth3 = 1;
  vector<shared_ptr<CNeuron>> cns3;
 // prepareCNeurons(10, 5, 1, 1, 1, "ip2.txt","ip2_bias.txt", cns3);
  prepareCNeurons(10, 500, 1, 1, 1, "ip2.txt","ip2_bias.txt", cns3);


  //init layers
  shared_ptr<ILayer> iLayer(make_shared<ILayer>(28, cxGPUContext));
  shared_ptr<CLayer> cLayer0(make_shared<CLayer>(cns0));
  shared_ptr<PLayer> pLayer0(make_shared<PLayer>(cxGPUContext, cqCommandQueue, kernel_pool, 20, 12));
  shared_ptr<CLayer> cLayer1(make_shared<CLayer>(cns1));
  shared_ptr<PLayer> pLayer1(make_shared<PLayer>(cxGPUContext, cqCommandQueue, kernel_pool, 50, 4));
  shared_ptr<CLayer> cLayer2(make_shared<CLayer>(cns2));
  shared_ptr<CLayer> outCLayer(make_shared<CLayer>(cns3));

  int flag = 0, total = 10000;
  cout << "Layers are ready. Let's run!\n";
  int ture_label = 0, test_label = 0;
  //cnn run
  double start, stop, durationTime, totaltime = 0;
  for (int i = 0; i != total; ++i) {
	  for (int j = 0; j != 28 * 28; ++j) {
		  inImage[j] = data[i * 28 * 28 + j];
		 // inImage[j] = 1;
		//  cout << inImage[j] << " ";
		 // if (j % 28 == 0) { cout << endl; }
	  }
	 // cout << endl;
	  for (int k = 0; k != 10; ++k) {
		  true_out[k] = label[i*10+k];
	  }

 
	  start = clock();
	  iLayer->activate(inImage, cxGPUContext, cqCommandQueue, inImgWidth);
	  cLayer0->activate(iLayer->getFeature());
	  pLayer0->activate(cLayer0->getFeature());

	  cLayer1->activate(pLayer0->getFeature());
	  pLayer1->activate(cLayer1->getFeature());

	  cLayer2->activate(pLayer1->getFeature());
	  outCLayer->activate(cLayer2->getFeature());
	  stop = clock();

	  durationTime = ((double)(stop - start)) / CLK_TCK;
	 // cout << "单次耗时：" << durationTime << " s" << endl;
	  totaltime += durationTime;
	  clEnqueueReadBuffer(cqCommandQueue, outCLayer->getFeature(), CL_TRUE, 0, sizeof(float) * 10, output, 0, NULL, NULL);
	  /*for (size_t w = 0; w != 10; w++) {
		  cout << output[w] << " ";
	  }
	  */
	  ture_label = findIndex(true_out);
	  test_label = findIndex(output);
	  if (ture_label == test_label) { flag++; }
	  if (i % 100 == 0) {
		  cout << "examples:" << i << "  true label: " << ture_label << " test label: " << test_label << " right num: " << flag << endl;
	  }
	    
  }
  cout << "程序耗时：" << totaltime << " s" << endl;
  cout << "平均每次耗时：" << totaltime/ total << " s" << endl;
  cout << "准确率:" << float(flag) / total << endl;
  /*
  char* x = new char[32];
  
  FeatureMaps out = outPLayer->getFeatureMaps();
  for (size_t i = 0; i < out.buffers.size(); i++) {
    cl::Buffer *o = out.buffers[i].get();
    Mat image = Mat::zeros(Size(out.width, out.height), CV_32FC3);
    commandQueue.enqueueReadBuffer(*o, CL_TRUE, 0, sizeof(cl_float) * 3 * out.width * out.height, image.data);
    sprintf(x, "output%d.png", i);
    image.convertTo(image, CV_8UC3);
    imwrite(x, image);
  }
  
  delete[] x;
  */
  cout << "Done!\n";
  delete[] data;
  delete[] label;
  delete[] inImage;
  delete[] output;
  delete[] true_out;
  system("pause");
  return 0;
}