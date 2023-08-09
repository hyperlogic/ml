#include <stdint.h>

#ifndef __DATASET__
#define __DATASET__

typedef struct {
    uint32_t magic;
    uint32_t num_images;
    uint32_t rows;
    uint32_t cols;
} ImagesHeader;

typedef struct {
    uint32_t magic;
    uint32_t num_labels;
} LabelsHeader;

typedef struct {
    ImagesHeader i_header;
    uint8_t* images;
    LabelsHeader l_header;
    uint8_t* labels;
} DataSet;

DataSet* malloc_data_set(const ImagesHeader* i_header, const LabelsHeader* l_header);

void free_data_set(DataSet* data_set);

DataSet* load_data_set(const char* test_images_filename, const char* test_labels_filename);

#endif
