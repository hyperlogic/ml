#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <assert.h>

#include "math_util.h"

const char* PARAMS_FILENAME = "params.bin";

#define L0_SIZE 2
#define L1_SIZE 16
#define L2_SIZE 2

// structure of params.bin
typedef struct {
    float fc1_weight[L1_SIZE * L0_SIZE];  // 16 x 2
    float fc1_bias[L1_SIZE];
    float fc2_weight[L2_SIZE * L1_SIZE];  // 2 x 16
    float fc2_bias[L2_SIZE];
} Params;

typedef struct {
    int num_rows;
    int num_cols;
} Shape;

//#define INDEX(r, c) r * NUM_COLS + c

struct NodeStruct;
typedef void (*Operator)(struct NodeStruct* self, const struct NodeStruct* left, const struct NodeStruct* right);

#define MAX_NUM_CHILDREN 2
#define MAX_NUM_VALUES 32
typedef struct NodeStruct {
    const char* name;
    int parent;
    int child[MAX_NUM_CHILDREN];
    Shape shape;
    Operator op;
    float value[MAX_NUM_VALUES];
} Node;



// matrix multiplication
void mul_op(Node* self, const Node* left, const Node* right) {
    assert(left->shape.num_cols == right->shape.num_rows);
    assert(self->shape.num_rows == left->shape.num_rows && self->shape.num_cols == right->shape.num_cols);
    for (int r = 0; r < self->shape.num_rows; r++) {
        for (int c = 0; c < self->shape.num_cols; c++) {
            float accum = 0;
            for (int i = 0; i < left->shape.num_cols; i++) {
                accum += left->value[r * left->shape.num_cols + i] * right->value[i * right->shape.num_cols + c];
            }
            self->value[r * self->shape.num_cols + c] = accum;
        }
    }
}

// component-wise addition of two column vectors
void add_op(Node* self, const Node* left, const Node* right) {
    assert(left->shape.num_cols == 1 && right->shape.num_cols == 1);
    assert(left->shape.num_rows == right->shape.num_rows);
    printf("got here add!\n");
    for (int i = 0; i < left->shape.num_rows; i++) {
        self->value[i] = left->value[i] + right->value[i];
    }
}

// dot product of two column vectors.
void dot_op(Node* self, const Node* left, const Node* right) {
    assert(left->shape.num_cols == 1 && right->shape.num_cols == 1);
    assert(left->shape.num_rows == right->shape.num_rows);
    float accum = 0;
    for (int i = 0; i < left->shape.num_rows; i++) {
        accum += left->value[i] * right->value[i];
    }
    self->value[0] = accum;
}

// component-wise relu of a column vector
void relu_op(Node* self, const Node* left, const Node* right) {
    assert(left->shape.num_cols == 1);
    for (int i = 0; i < left->shape.num_rows; i++) {
        self->value[i] = relu(left->value[i]);
    }
}

#define SHAPE_R_2 {2, 1}
#define SHAPE_R_16 {16, 1}
#define SHAPE_SCALAR {1, 1}

// computational graph for loss of MLP
// L = 1/2 * dot(y, y_hat)
// where y = W_2 * relu(W_1 * x + b_1) + b_2
Node g_node[14] = {
    {"L", -1, {1, 2}, SHAPE_SCALAR, mul_op}, // 0
    {"1/2", 0, {-1, -1}, SHAPE_SCALAR, NULL}, // 1
    {"u_2", 0, {3, 13}, SHAPE_SCALAR, dot_op}, // 2
    {"y", 2, {4, 12}, SHAPE_R_2, add_op}, // 3
    {"u_4", 3, {5, 6}, SHAPE_R_2, mul_op}, // 4
    {"W_2", 4, {-1, -1}, {2, 16}, NULL}, // 5
    {"u_6", 4, {7, -1}, SHAPE_R_16, relu_op}, // 6
    {"u_7", 6, {8, 11}, SHAPE_R_16, add_op}, // 7
    {"u_8", 7, {9, 10}, SHAPE_R_16, mul_op}, // 8
    {"W_1", 8, {-1, -1}, {16, 2}, NULL}, // 9
    {"x", 8, {-1, -1}, SHAPE_R_2, NULL}, // 10
    {"b_1", 7, {-1, -1}, SHAPE_R_16, NULL}, // 11
    {"b_2", 3, {-1, -1}, SHAPE_R_2, NULL}, // 12
    {"y_hat", 2, {-1, -1}, SHAPE_R_2, NULL}, // 13
};

void print_graph(const Node* node, int indent) {
    for (int i = 0; i < indent; i++) {
        printf("    ");
    }

    // print name
    printf("%s = [ ", node->name);
    for (int i = 0; i < node->shape.num_cols * node->shape.num_rows; i++) {
        printf("%0.4f ", node->value[i]);
    }
    printf("]\n");

    if (node->child[0] >= 0) {
        print_graph(g_node + node->child[0], indent + 1);
    }
    if (node->child[1] >= 0) {
        print_graph(g_node + node->child[1], indent + 1);
    }
}

void forward(Node* node) {
    Node* left = node->child[0] >= 0 ? g_node + node->child[0] : NULL;
    Node* right = node->child[1] >= 0 ? g_node + node->child[1] : NULL;
    if (left) {
        forward(left);
    }
    if (right) {
        forward(right);
    }
    if (node->op) {
        printf("evaluating node \"%s\"\n", node->name);
        node->op(node, left, right);
    }
}

Params* load_model(const char* params_filename) {
    // open params file
    FILE* fp = fopen(params_filename, "rb");
    if (!fp) {
        printf("Error opening file \"%s\", errno = %d\n", params_filename, errno);
        return NULL;
    }

    // read data
    Params* p_params = malloc(sizeof(Params));
    size_t bytes_read = fread(p_params, 1, sizeof(Params), fp);
    if (bytes_read != sizeof(Params)) {
        printf("Error reading from file \"%s\", bytes_read = %u, expected = %u\n", params_filename, (uint32_t)bytes_read, (uint32_t)sizeof(Params));
        free(p_params);
        return NULL;
    }
    fclose(fp);

    return p_params;
}

void print_matrix(const float* mat, int num_rows, int num_cols) {
    for (int r = 0; r < num_rows; r++) {
        for (int c = 0; c < num_cols; c++) {
            printf("%8.4f", mat[r * num_cols + c]);
        }
        printf("\n");
    }
}

void print_params(const Params* p_params)
{
    printf("fc1_weight =\n");
    print_matrix(p_params->fc1_weight, L1_SIZE, L0_SIZE);
    printf("fc1_bias =\n");
    print_matrix(p_params->fc1_bias, 1, L1_SIZE);
    printf("fc2_weight =\n");
    print_matrix(p_params->fc2_weight, L2_SIZE, L1_SIZE);
    printf("fc2_bias =\n");
    print_matrix(p_params->fc2_bias, 1, L2_SIZE);
}


int main(int argc, const char* argv[]) {
    Params* p_params = load_model(PARAMS_FILENAME);

    print_params(p_params);

    // initialize the computation graph with the parameters from the model.
    // W_1
    memcpy(g_node[9].value, p_params->fc1_weight, L1_SIZE * L0_SIZE * sizeof(float));
    // b_1
    memcpy(g_node[11].value, p_params->fc1_bias, L1_SIZE * sizeof(float));
    // W_2
    memcpy(g_node[5].value, p_params->fc2_weight, L2_SIZE * L1_SIZE * sizeof(float));
    // b_2
    memcpy(g_node[12].value, p_params->fc2_bias, L2_SIZE * sizeof(float));
    // 1/2
    g_node[1].value[0] = 0.5f;
    // y_hat
    g_node[13].value[0] = 0.0f;
    g_node[13].value[1] = 1.0f;
    // x
    g_node[10].value[0] = 0.5f;
    g_node[10].value[1] = 0.5f;

    forward(g_node);

    print_graph(g_node, 0);

    free(p_params);

    return 0;
}
