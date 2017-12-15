#include <fstream>
#include <iostream>
#include <time.h>
using namespace std;

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

double iter(double r_c, double i_c, int max){
  double r_t = 0.0, i_t = 0.0;
  double r_t_2, i_t_2;
  for (int i = 0; i < max; i++){
    r_t_2 = r_t * r_t - i_t * i_t;
    i_t_2 = 2 * r_t * i_t;
    r_t = r_t_2 + r_c;
    i_t = i_t_2 + i_c;
    if (r_t*r_t + i_t*i_t > 4) return (double)i/(double)max;
  }
  return 0;
}

int main(int argc, char ** argv){
  /*plotting parameters*/
  char filename[] = "test.bmp";
  int width = 3840, height = 2160;
  double r_min = -2.86, r_max = 1.86, i_min = -1.325, i_max = 1.325;
  int max = 100;

  /*temp vars*/
  double r, i, t;
  int c;

  /*image data generation*/
  char * img_data = new char[width*height*3];
  clock_t t1 = clock();
  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      r = ((double)x/(double)width)*(r_max-r_min)+r_min;
      i = ((double)y/(double)height)*(i_max-i_min)+i_min;
      t = iter(r, i, max);
      img_data[y*3*width + x*3 + 0] = (char)(0*t);
      img_data[y*3*width + x*3 + 1] = (char)(255*t);
      img_data[y*3*width + x*3 + 2] = (char)(0*t);
    }
  }
  clock_t t2 = clock();
  cout << "Took " << (double)(t2-t1)/CLOCKS_PER_SEC << " seconds.\n";

  /*writing to file*/
  bmp_write(img_data, width, height, filename);

  delete img_data;
  return 0;
}
