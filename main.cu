#include <fstream>
using namespace std;

class params{
private:
  double r_min_ini, r_min_trg;
  double r_max_ini, r_max_trg;
  double i_min_ini, i_min_trg;
  double i_max_ini, i_max_trg;

public:
  int width, height;
  double r_min, r_max, i_min, i_max;
  int max;

  params(){
    r_min_ini = -2.86, r_min_trg = -0.1777;
    r_max_ini = 1.86, r_max_trg = -0.1194;
    i_min_ini = -1.33, i_min_trg = 1.0138;
    i_max_ini = 1.33, i_max_trg = 1.0472;

    width = 1920, height = 1080;
    r_min = r_min_ini, r_max = r_max_ini, i_min = i_min_ini, i_max = i_max_ini;
    max = 50;
  }

  void set_frame_number(int, int);
};

void params::set_frame_number(int n, int max){
  if (n < max){
    double t = 1 - pow(0.001, (double)n/(double)max);
    r_min = r_min_ini +  t*(r_min_trg - r_min_ini);
    r_max = r_max_ini +  t*(r_max_trg - r_max_ini);
    i_min = i_min_ini +  t*(i_min_trg - i_min_ini);
    i_max = i_max_ini +  t*(i_max_trg - i_max_ini);
  }
  else{
    r_min = r_min_trg, r_max = r_max_trg, i_min = i_min_trg, i_max = i_max_trg;
  }
}

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

__host__ __device__ double iter(double r_c, double i_c, int max){
  double r_t = 0.0, i_t = 0.0;
  double r_t_2, i_t_2, mag;
  for (int i = 0; i < max; i++){
    r_t_2 = r_t * r_t - i_t * i_t;
    i_t_2 = 2 * r_t * i_t;
    r_t = r_t_2 + r_c;
    i_t = i_t_2 + i_c;
    mag = r_t*r_t + i_t*i_t;
    if (mag > 4){
      return (double)(i + 1 - log(log(mag))/log(2.0))/(double)max;
    }
  }
  return 0;
}

__global__ void populate(char * d_img_data, double r_min, double r_max,
  double i_min, double i_max, int width, int height, int max){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < width && y < height){
      double r = ((double)x/(double)width)*(r_max-r_min)+r_min;
      double i = ((double)y/(double)height)*(i_max-i_min)+i_min;
      double t = iter(r, i, max);
      d_img_data[y*3*width + x*3 + 0] = (char)(0);
      d_img_data[y*3*width + x*3 + 1] = (char)(0);
      d_img_data[y*3*width + x*3 + 2] = (char)(255*t);
    }
  }

void plot_frame_gpu(params plot, char * filename){
  char * h_img_data = new char[plot.width*plot.height*3*sizeof(char)];
  char * d_img_data;
  cudaMalloc((void **) &d_img_data, plot.width*plot.height*3*sizeof(char));
  dim3 threads(32, 32, 1);
  dim3 grid(plot.width/32 + 1, plot.height/32 + 1, 1);
  populate<<<grid, threads>>>(d_img_data, plot.r_min, plot.r_max, plot.i_min,
    plot.i_max, plot.width, plot.height, plot.max);
  cudaMemcpy(h_img_data, d_img_data, plot.width*plot.height*3*sizeof(char),
    cudaMemcpyDeviceToHost);
  cudaFree(d_img_data);
  bmp_write(h_img_data, plot.width, plot.height, filename);
  delete[] h_img_data;
}

int main(int argc, char ** argv){
  params plot;
  plot_frame_gpu(plot, "test1.bmp");
  return 0;
}
