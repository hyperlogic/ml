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

struct Node;
typedef void (*Operator)(struct Node* self);
typedef void (*GradientOperator)(const struct Node* self, struct Node* child);

#define MAX_NUM_CHILDREN 2
typedef struct Node {
    const char* name;
    int self_index;
    int parent_index;
    int child_index[MAX_NUM_CHILDREN];
    Tensor data;
    Operator op;
    GradientOperator grad_op;
    struct Node* parent;
    struct Node* child[MAX_NUM_CHILDREN];
} Node_t;

//
// ops - for evaluating evaluating graph "forward"
//

// matrix multiplication
void mul_op(Node_t* self) {
    assert(self);
    const Node_t* left = self->child[0];
    const Node_t* right = self->child[1];
    assert(left && right);
    tensor_mul(&self->data, &left->data, &right->data);
}

// component-wise addition of two column vectors
void add_op(Node_t* self) {
    assert(self);
    const Node_t* left = self->child[0];
    const Node_t* right = self->child[1];
    assert(left && right);
    tensor_add(&self->data, &left->data, &right->data);
}

// component-wise subtraction of two column vectors
void sub_op(Node_t* self) {
    assert(self);
    const Node_t* left = self->child[0];
    const Node_t* right = self->child[1];
    assert(left && right);
    tensor_sub(&self->data, &left->data, &right->data);
}

// dot product of a vector with itself.
void dot_self_op(Node_t* self) {
    assert(self);
    const Node_t* left = self->child[0];
    assert(left);

    assert(left->data.shape.num_cols == 1);

    Tensor temp = {{left->data.shape.num_cols, left->data.shape.num_rows}};
    tensor_transpose_xy(&temp, &left->data);
    tensor_mul(&self->data, &temp, &left->data);
}

// component-wise relu of a column vector
void relu_op(Node_t* self) {
    assert(self);
    const Node_t* left = self->child[0];
    assert(left);

    assert(left->data.shape.num_cols == 1);

    tensor_comp_func(&self->data, &left->data, relu);
}

//
// gradient ops - for evaluating gradients "backward"
// it returns the partial derivative of self w.r.t. the child
//

void mul_grad(const Node_t* self, Node_t* child) {
    // TODO:
}

void add_grad(const Node_t* self, Node_t* child) {
    // TODO:
}

void sub_grad(const Node_t* self, Node_t* child) {
    // TODO:
}

void dot_self_grad(const Node_t* self, Node_t* child) {
    // TODO:
}

void relu_grad(const Node_t* self, Node_t* child) {
    // TODO:
}

#define SHAPE_R_2 {2, 1}
#define SHAPE_R_16 {16, 1}
#define SHAPE_SCALAR {1, 1}

// computational graph for loss of MLP
// L = 1/2 * dot(y, y_hat)
// where y = W_2 * relu(W_1 * x + b_1) + b_2
#define NUM_NODES 15
Node_t g_node[NUM_NODES] = {
    {"L", 0, -1, {1, 2}, {SHAPE_SCALAR}, mul_op, mul_grad}, // L
    {"1/2", 1, 0, {-1, -1}, {SHAPE_SCALAR, {0.5f}}, NULL, NULL},
    {"u_2", 2, 0, {3, -1}, {SHAPE_SCALAR}, dot_self_op, dot_self_grad},
    {"u_3", 3, 2, {4, 14}, {SHAPE_R_2}, sub_op, sub_grad},
    {"y", 4, 3, {5, 13}, {SHAPE_R_2}, add_op, add_grad}, // y
    {"u_5", 5, 4, {6, 7}, {SHAPE_R_2}, mul_op, mul_grad},
    {"W_2", 6, 5, {-1, -1}, {{2, 16}}, NULL, NULL}, // W_2
    {"u_7", 7, 5, {8, -1}, {SHAPE_R_16}, relu_op, relu_grad},
    {"u_8", 8, 7, {9, 12}, {SHAPE_R_16}, add_op, add_grad},
    {"u_9", 9, 8, {10, 11}, {SHAPE_R_16}, mul_op, mul_grad},
    {"W_1", 10, 9, {-1, -1}, {{16, 2}}, NULL, NULL}, // W_1
    {"x", 11, 9, {-1, -1}, {SHAPE_R_2}, NULL, NULL}, // x
    {"b_1", 12, 8, {-1, -1}, {SHAPE_R_16}, NULL, NULL}, // b_1
    {"b_2", 13, 4, {-1, -1}, {SHAPE_R_2}, NULL, NULL}, // b_2
    {"y_hat", 14, 3, {-1, -1}, {SHAPE_R_2}, NULL, NULL} // y_hat
};

#define W_1_INDEX 10
#define B_1_INDEX 12
#define W_2_INDEX 6
#define B_2_INDEX 13
#define Y_HAT_INDEX 14
#define X_INDEX 11

float g_grad_table[NUM_NODES];

void print_graph(const Node_t* node, int indent) {
    for (int i = 0; i < indent; i++) {
        printf("    ");
    }

    // print name
    printf("%s = [ ", node->name);

    // print tensor elements...
    for (int i = 0; i < node->data.shape.num_cols * node->data.shape.num_rows; i++) {
        printf("%0.4f ", node->data.value[i]);
    }
    printf("]\n");

    if (node->child[0]) {
        print_graph(node->child[0], indent + 1);
    }
    if (node->child[1]) {
        print_graph(node->child[1], indent + 1);
    }
}

void forward(Node_t* node) {
    Node_t* left = node->child[0] ? node->child[0] : NULL;
    Node_t* right = node->child[1] ? node->child[1] : NULL;
    if (left) {
        forward(left);
    }
    if (right) {
        forward(right);
    }
    if (node->op) {
        node->op(node);
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

void print_params(const Params* p_params) {
    printf("fc1_weight =\n");
    print_matrix(p_params->fc1_weight, L1_SIZE, L0_SIZE);
    printf("fc1_bias =\n");
    print_matrix(p_params->fc1_bias, 1, L1_SIZE);
    printf("fc2_weight =\n");
    print_matrix(p_params->fc2_weight, L2_SIZE, L1_SIZE);
    printf("fc2_bias =\n");
    print_matrix(p_params->fc2_bias, 1, L2_SIZE);
}

void backward() {
    /*
    g_grad_table[0] = 1.0f;
    for (int j = 1; j >= NUM_NODES; j++) {
        Node_t* node = g_node + j;
        g_grad_table[j] = g_grad_table[node->parent_index] * parent->grad_op(parent, node);
    }
    */
}

void init_graph(const Params* params)
{
    // update pointers from indices
    for (int i = 0; i < NUM_NODES; i++) {
        g_node[i].parent = g_node[i].parent_index >= 0 ? g_node + g_node[i].parent_index : NULL;
        g_node[i].child[0] = g_node[i].child_index[0] >= 0 ? g_node + g_node[i].child_index[0] : NULL;
        g_node[i].child[1] = g_node[i].child_index[1] >= 0 ? g_node + g_node[i].child_index[1] : NULL;
    }

    // W_1
    memcpy(g_node[W_1_INDEX].data.value, params->fc1_weight, L1_SIZE * L0_SIZE * sizeof(float));
    // b_1
    memcpy(g_node[B_1_INDEX].data.value, params->fc1_bias, L1_SIZE * sizeof(float));
    // W_2
    memcpy(g_node[W_2_INDEX].data.value, params->fc2_weight, L2_SIZE * L1_SIZE * sizeof(float));
    // b_2
    memcpy(g_node[B_2_INDEX].data.value, params->fc2_bias, L2_SIZE * sizeof(float));
}

int main(int argc, const char* argv[]) {
    Params* params = load_model(PARAMS_FILENAME);

    print_params(params);

    // initialize the computation graph with the parameters from the model.
    init_graph(params);

    // y_hat
    g_node[Y_HAT_INDEX].data.value[0] = 0.0f;
    g_node[Y_HAT_INDEX].data.value[1] = 1.0f;
    // x
    g_node[X_INDEX].data.value[0] = 0.5f;
    g_node[X_INDEX].data.value[1] = 0.5f;

    forward(g_node);

    print_graph(g_node, 0);

    backward(g_node);

    free(params);

    return 0;
}
