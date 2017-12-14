#include <fstream>

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
  int i = 0;
  while (i++ < max){
    r_t_2 = r_t * r_t - i_t * i_t;
    i_t_2 = 2 * r_t * i_t;
    r_t = r_t_2 + r_c;
    i_t = i_t_2 + i_c;
    if (r_t*r_t + i_t*i_t > 4) return (1-(double)i/(double)max);
  }
  return 0;
}

int main(int argc, char ** argv){
  char filename[] = "test.bmp";
  int width = 1920, height = 1080;
  double r_min = -3.2, r_max = 3.2, i_min = -1.8, i_max = 1.8;

  return 0;
}
