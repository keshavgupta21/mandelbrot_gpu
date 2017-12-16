#include <fstream>
#include <time.h>
using namespace std;

class params{
public:
  int width, height;
  double r_min, r_max, i_min, i_max;
  int max;
  params(){
    width = 3840, height = 2160;
    r_min = -2.86, r_max = 1.86, i_min = -1.32, i_max = 1.32;
    max = 50;
  }
};

void bmp_write(char * img_data, int width, int height, char * filename){
  int size = height*width*3 + 54;
  ofstream bmp(filename, ios::binary);

  char bmp_header[54] = {'B', 'M',
  (char)size, (char)(size >> 8), (char)(size >> 16), (char)(size >> 24),
  0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, (char)width, (char)(width >> 8),
  0, 0, (char)height,(char)(height >> 8), 0, 0, 1, 0, 24, 0, 0, 0, 0, 0,
  (char)(size - 54), (char)((size - 54) >> 8),(char)((size - 54) >> 16),
  (char)((size - 54) >> 24),0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0};

  bmp.write(bmp_header, sizeof bmp_header);
  bmp.write(img_data, size - 54);
  bmp.close();
}

__device__ double iter(double r_c, double i_c, int max){
  double r_t = 0.0, i_t = 0.0;
  double r_t_2, i_t_2;
  for (int i = 0; i < max; i++){
    r_t_2 = r_t * r_t - i_t * i_t;
    i_t_2 = 2 * r_t * i_t;
    r_t = r_t_2 + r_c;
    i_t = i_t_2 + i_c;
    if (r_t*r_t + i_t*i_t > 4) return (double)(i + 1 - log(log(r_t*r_t + i_t*i_t))/log(2.0))/(double)max;
  }
  return 0;
}

__global__ void populate(char * d_img_data, double r_min, double r_max, double i_min, double i_max, int width, int height, int max){
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x < width && y < height){
    double r = ((double)x/(double)width)*(r_max-r_min)+r_min;
    double i = ((double)y/(double)height)*(i_max-i_min)+i_min;
    double t = iter(r, i, max);
    d_img_data[y*3*width + x*3 + 0] = (char)(255*t);
    d_img_data[y*3*width + x*3 + 1] = (char)(255*t*t);
    d_img_data[y*3*width + x*3 + 2] = (char)(255*t*t*t);
  }
}

void plot_frame(params plot, char * filename){
  /*getting image data*/
  char * h_img_data = new char[plot.width*plot.height*3*sizeof(char)];
  char * d_img_data;
  cudaMalloc((void **) &d_img_data, plot.width*plot.height*3*sizeof(char));
  dim3 threads(32, 32, 1);
  dim3 grid(plot.width/threads.x, plot.height/threads.y, 1);
  populate<<<grid, threads>>>(d_img_data, plot.r_min, plot.r_max, plot.i_min, plot.i_max, plot.width, plot.height, plot.max);
  cudaMemcpy(h_img_data, d_img_data, plot.width*plot.height*3*sizeof(char), cudaMemcpyDeviceToHost);
  cudaFree(d_img_data);

  /*write to file*/
  bmp_write(h_img_data, plot.width, plot.height, filename);
  delete[] h_img_data;
}

int main(){
  char * filename = (char *)malloc(20*sizeof(char));
  int num_frames = 1;
  params plot;
  clock_t t1 = clock();
  for (int i = 0; i < num_frames; i++){
    sprintf(filename, "anim/%06d.bmp", i);
    plot_frame(plot, filename);
  }
  clock_t t2 = clock();
  printf("Took %1.6f seconds per frame.\n", (double)(t2-t1)/(CLOCKS_PER_SEC*num_frames));
  delete[] filename;
  return 0;
}
