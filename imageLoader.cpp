
#include "lodepng.h"
#include <string>

typedef unsigned char byte; // most useful typedef ever
struct imgData {
    imgData(byte* pix = nullptr, unsigned int w = 0, unsigned int h = 0) : pixels(pix), width(w), height(h) {
    };
    byte* pixels;
    unsigned int width;
    unsigned int height;
};

imgData loadImage(char* filename) {
    unsigned int width, height;
    byte* rgb;
    unsigned error = lodepng_decode_file(&rgb, &width, &height, filename, LCT_RGBA, 8);
    if(error) {
        printf("LodePNG had an error during file processing. Exiting program.\n");
        printf("Error code: %u: %s\n", error, lodepng_error_text(error));
        exit(2);
    }
    byte* grayscale = new byte[width*height];
    byte* img = rgb;
    for(int i = 0; i < width*height; ++i) {
        int r = *img++;
        int g = *img++;
        int b = *img++;
        int a = *img++;
        grayscale[i] = 0.3*r + 0.6*g + 0.1*b+0.5;
    }
    free(rgb);
    return imgData(grayscale, width, height);
}


void writeImage(char* filename, std::string appendTxt, imgData img) {
  std::string newName = filename;
  newName = newName.substr(0, newName.rfind("."));
  newName.append("_").append(appendTxt).append(".png");
  unsigned error = lodepng_encode_file(newName.c_str(), img.pixels, img.width, img.height, LCT_GREY, 8);
    if(error) {
        printf("LodePNG had an error during file writing. Exiting program.\n");
        printf("Error code: %u: %s\n", error, lodepng_error_text(error));
        exit(3);
    }
    delete [] img.pixels;
}
