__kernel void Convolution(__global float* inImage,
                          int kernelWidth,
                          __global float* kernelData,
                          __global float* outImage,
						  int imgsize,
						  int nKernels,
						  int inImagesize,
						  int outImagesize,
						   __global float* bias) {
    int c = get_global_id(0),
        r = get_global_id(1);
    int outputs   = get_global_size(0);
	int imgWidth  = get_global_size(1);
	int inimg, outimg, ker;
	outimg = c*imgWidth + (r/outImagesize) * outImagesize + r%outImagesize;
	outImage[outimg] = 0;
	for(int i = 0; i != nKernels; ++i){
		for (int m = 0; m < kernelWidth; m++) {
			for (int n = 0; n < kernelWidth; n++) {
			inimg = i * inImagesize * inImagesize + (r/outImagesize+m)*inImagesize+r%outImagesize+n;
			ker = (c * nKernels * kernelWidth * kernelWidth) + (i * kernelWidth * kernelWidth) + m * kernelWidth + n;
			outImage[outimg] += inImage[inimg] * kernelData[ker];
			//printf("%d %d %d %d %d     ",c,r,inimg,ker,outimg);
			 }
		 }
	 }
	 float ans = outImage[outimg]+bias[c];
	 outImage[outimg] = ans ;
	 //printf("%d %d %d %d %d\n",c,r,inimg,ker,outimg);
	 //printf("%d,%d outImage[%d]=%f\n",c,r,outimg,outImage[outimg]);
}
__kernel void Pooling(__global float* inImage,
	                    __global float* outImage,
						int inImgWidth,
						int outImgWidth) {

    int i = get_global_id(0),
    j = get_global_id(1);
	
	float a, b, c, d, fst_max, snd_max;

	int a1,b1,c1,d1;
	a1=i * inImgWidth * inImgWidth + 2 * (j / outImgWidth) * inImgWidth + 2 * (j%outImgWidth);
	b1=i * inImgWidth * inImgWidth + 2 * (j / outImgWidth) * inImgWidth + 2 * (j%outImgWidth) + 1;
	c1=i * inImgWidth * inImgWidth + (2 * (j / outImgWidth) + 1) * inImgWidth + 2 * (j%outImgWidth);
	d1=i * inImgWidth * inImgWidth + (2 * (j / outImgWidth) + 1) * inImgWidth + 2 * (j%outImgWidth) + 1;
	a = inImage[a1];
	b = inImage[b1];
	c = inImage[c1];
	d = inImage[d1];
	fst_max = ((a > b) ? a : b);
	snd_max = ((c > d) ? c : d);
	int outimg = i * outImgWidth * outImgWidth + (j / outImgWidth) * outImgWidth  + j%outImgWidth;
	outImage[outimg] = fst_max > snd_max ? fst_max : snd_max;
	//printf("%d %d %d %d %d %d\n",i,j,a1,b1,c1,d1);
	
}
__kernel void Fconnect(__global float* inImage,
	                    __global float* outImage,
						 __global float* kernelData,
						 __global float* bias,
						int inImgWidth) {


	int i = get_global_id(0);
	int size = get_global_size(0);
	outImage[i] = 0;
	float ans = 0;
	for(int a = 0; a != inImgWidth; ++a){
		outImage[i] += inImage[a] * kernelData[ i * inImgWidth + a];
		//printf("%d %f ",i,outImage[i]);
	}
	ans = outImage[i] + bias[i];
	outImage[i] = (size==10)? ans : (ans>0? ans : 0);
	//outImage[i] = ans;
	
}

__kernel void Pad(__global float* inImage,
                 __global float* outImage,
				 int inImagesize,
				 int outImagesize,
				 int pad) {
						   
	int c = get_global_id(0),
        r = get_global_id(1); 
	outImage[( c+ pad) * outImagesize * outImagesize + r + pad]=inImage[c * inImagesize * inImagesize + r];
	
	}