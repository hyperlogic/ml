#include "dataset.h"

#include <stdio.h>
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

// be = big endian
static uint32_t read_be_uint32(FILE* fp) {
    uint8_t bytes[4];
    size_t bytes_read = fread(&bytes, 1, 4, fp);
    if (bytes_read != 4) {
        printf("fread() error, bytes_read = %u expected = %u\n", (uint32_t)bytes_read, (uint32_t)4);
        return 0;
    }
    return (uint32_t)bytes[3] | (uint32_t)bytes[2] << 8 | (uint32_t)bytes[1] << 16 | (uint32_t)bytes[0] << 24;
}

static int read_bytes(FILE* fp, uint8_t* bytes, size_t num_bytes) {
    uint8_t* p = bytes;
    size_t bytes_left = num_bytes;
    size_t bytes_read = 0;
    do {
        bytes_read = fread(p, 1, bytes_left, fp);
        if (bytes_read == 0) {
            printf("read_bytes: error 0 bytes read\n");
            return 0;
        }
        p += bytes_read;
        bytes_left -= bytes_read;
    } while (bytes_read < bytes_left);
    return 1;
}

DataSet* load_data_set(const char* test_images_filename, const char* test_labels_filename)
{
    FILE* i_fp = fopen(test_images_filename, "rb");
    if (!i_fp) {
        printf("Error opening file \"%s\", errno = %d\n", test_images_filename, errno);
        return NULL;
    }

    FILE* l_fp = fopen(test_labels_filename, "rb");
    if (!l_fp) {
        printf("Error opening file \"%s\", errno = %d\n", test_images_filename, errno);
        return NULL;
    }

    ImagesHeader i_header;
    i_header.magic = read_be_uint32(i_fp);
    const uint32_t IMAGE_MAGIC = 2051;
    if (i_header.magic != IMAGE_MAGIC) {
        printf("Error loading \"%s\", bad magic number %d, expected %d\n", test_images_filename, i_header.magic, IMAGE_MAGIC);
        return NULL;
    }
    i_header.num_images = read_be_uint32(i_fp);
    i_header.rows = read_be_uint32(i_fp);
    i_header.cols = read_be_uint32(i_fp);

    LabelsHeader l_header;
    l_header.magic = read_be_uint32(l_fp);
    const uint32_t LABEL_MAGIC = 2049;
    if (l_header.magic != LABEL_MAGIC) {
        printf("Error loading \"%s\", bad magic number %d, expected %d\n", test_labels_filename, l_header.magic, LABEL_MAGIC);
        return NULL;
    }
    l_header.num_labels = read_be_uint32(l_fp);
    if (i_header.num_images != l_header.num_labels) {
        printf("Error image and label count mismatch, num_images = %d, num_labels = %d\n", i_header.num_images, l_header.num_labels);
        return NULL;
    }

    DataSet* data_set = malloc_data_set(&i_header, &l_header);

    size_t i_size = i_header.num_images * i_header.rows * i_header.cols;
    if (!read_bytes(i_fp, data_set->images, i_size)) {
        fclose(i_fp);
        fclose(l_fp);
        free_data_set(data_set);
        return NULL;
    }
    fclose(i_fp);

    if (!read_bytes(l_fp, data_set->labels, l_header.num_labels)) {
        fclose(l_fp);
        free_data_set(data_set);
        return NULL;
    }
    fclose(l_fp);

    return data_set;
}
