#include "dataset.h"

#include <stdlib.h>

DataSet* malloc_data_set(const ImagesHeader* i_header, const LabelsHeader* l_header) {
    DataSet* data_set = malloc(sizeof(DataSet));
    data_set->i_header = *i_header;
    data_set->images = malloc(i_header->num_images * i_header->rows * i_header->cols);
    data_set->l_header = *l_header;
    data_set->labels = malloc(l_header->num_labels);
    return data_set;
}

void free_data_set(DataSet* data_set) {
    free(data_set->images);
    free(data_set->labels);
    free(data_set);
}
