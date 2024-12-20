#include <assert.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>

#include <limits>
#include <random>

#include "board/board.hpp"
#include "math_lib/maths.hpp"
#include "pcg_random.hpp"
#define MAX_NODES 120000
#define MAX_SIMULATION_COUNT 393216000
#define BATCH_SIZE 32768
#define SEARCH_DEPTH 64
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error:\n");                                 \
            fprintf(stderr, "    File:       %s\n", __FILE__);                \
            fprintf(stderr, "    Line:       %d\n", __LINE__);                \
            fprintf(stderr, "    Error code: %d\n", err);                     \
            fprintf(stderr, "    Error text: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)
pcg_extras::seed_seq_from<std::random_device> seed_source;  // Create a seed source from random_device
pcg32 random_num(seed_source);                              // Pass the seed source to initialize the RNG

static const int x_coords[25] = {
    0, 1, 2, 3, 4,
    0, 1, 2, 3, 4,
    0, 1, 2, 3, 4,
    0, 1, 2, 3, 4,
    0, 1, 2, 3, 4};

static const int y_coords[25] = {
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4};

inline Direction get_direction(int start, int destination) {
    if (x_coords[start] == x_coords[destination]) {
        return VERTICAL;
    } else if (y_coords[start] == y_coords[destination]) {
        return HORIZONTAL;
    } else {
        return DIAGONAL;
    }
}

struct Node {
    Board board;
    Node* parent;
    Node* children[64];
    int num_children;
    int num_untried_moves;
    int move_from_parent;
    int visits;
    int wins;
    bool is_terminal;
};

Node node_pool[MAX_NODES];
int node_pool_size = 0;

Node* allocate_node() {
    Node* node;

    if (node_pool_size >= MAX_NODES) {
        assert(false);
        return nullptr;
    }
    node = &node_pool[node_pool_size++];

    return node;
}

int total_pruned_num = 0;
#define RATIO_PARAM 0.125
#define MIN_VISITS_FOR_PRUNING 163840

Node* select_child(Node* node) {
    const int num_children = node->num_children;
    if (num_children == 1) {
        return node->children[0];
    }
    float averages[64];
    float std_devs[64];
    float left_expected_outcomes;
    int pruned[64] = {0};

    float max_left_expected_outcome = -std::numeric_limits<float>::infinity();
    int max_left_expected_outcome_index = -1;

    // Compute average, standard deviation, and left_expected_outcome
    for (int i = 0; i < num_children; i++) {
        Node* child = node->children[i];
        averages[i] = (float)child->wins / child->visits;
        if (child->visits > MIN_VISITS_FOR_PRUNING) {
            std_devs[i] = sqrtf(averages[i] * (1.0f - averages[i]));
            left_expected_outcomes = averages[i] - RATIO_PARAM * std_devs[i];
            if (left_expected_outcomes > max_left_expected_outcome) {
                max_left_expected_outcome = left_expected_outcomes;
                max_left_expected_outcome_index = i;
            }
        }
    }

    for (int i = 0; i < num_children; i++) {
        if ((node->children[i]->visits > MIN_VISITS_FOR_PRUNING &&
             averages[i] + RATIO_PARAM * std_devs[i] < max_left_expected_outcome &&
             i != max_left_expected_outcome_index)) {
            pruned[i] = 1;
        }
    }

    // Remove pruned children
    int new_num_children = 0;
    for (int i = 0; i < num_children; i++) {
        if (!pruned[i]) {
            node->children[new_num_children++] = node->children[i];
        } else {
            // Optionally, free the pruned child node
            total_pruned_num++;
        }
    }
    if (new_num_children == 0) {
        node->children[new_num_children++] = node->children[0];
    }
    node->num_children = new_num_children;
    if (new_num_children == 1) {
        return node->children[0];
    }

    Node* best_child = nullptr;
    float best_value = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < node->num_children; i++) {
        Node* child = node->children[i];
        int child_visits = child->visits;
        int parent_visits = node->visits;
        float standard_value;
        // Standard UCB value
        if (child_visits > 0) {
            standard_value = fast_UCB(child->wins, child_visits, parent_visits);
        } else {
            // Encourage exploration of unvisited nodes
            standard_value = std::numeric_limits<float>::infinity();
        }

        if (standard_value > best_value) {
            best_value = standard_value;
            best_child = child;
        }
    }
    return best_child;
}

__global__ void setup_kernel(curandState* state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void simulate_kernel(curandState* state, Board* board, int* result, int batch_size) {
    extern __shared__ int local_win[];  // Use dynamic shared memory for results

    int thread_id = threadIdx.x;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= BATCH_SIZE) return;

    local_win[thread_id] = 0;
    __syncthreads();

    // Simulation for each thread
    Board board_copy = *board;
    bool done = false;

    while (!board_copy.check_winner() && !done) {
        board_copy.generate_moves();
        int valid_pieces[6];
        int valid_pieces_count = 0;

        // Determine valid pieces
        for (int valid_piece = 0; valid_piece < 6; ++valid_piece) {
            int pos = board_copy.piece_position[board_copy.moving_color ^ 1][valid_piece];
            if (pos == -1) continue;
            if (board_copy.moving_color == BLUE) {
                pos = 24 - pos;
            }
            if (pos != 1 && pos != 5 && pos != 6) {
                valid_pieces[valid_pieces_count++] = valid_piece;
            }
        }

        if (valid_pieces_count == 0) {
            local_win[thread_id] = (board_copy.moving_color == (*board).moving_color) ? 1 : 0;
            done = true;
            break;
        }

        // Copy state to local memory for efficiency
        curandState localState = state[id];
        // Pick a random valid piece and move
        int random_index = curand(&localState) % valid_pieces_count;
        int move = valid_pieces[random_index];
        move += PIECE_NUM * (curand(&localState) % board_copy.move_multiplier);

        board_copy.move(move);
    }

    // Final winner check if simulation was incomplete
    if (!done) {
        local_win[thread_id] = (board_copy.moving_color == (*board).moving_color) ? 1 : 0;
    }

    // Reduction to aggregate results in shared memory
    __syncthreads();
    for (int stride = batch_size / 2; stride > 0; stride >>= 1) {
        if (thread_id + stride < batch_size) {
            local_win[thread_id] += local_win[thread_id + stride];
        }
        __syncthreads();
    }

    // Write the final result from thread 0 of each block
    if (thread_id == 0) {
        atomicAdd(result, local_win[0]);
    }
}
int call_times = 1;

int MCTS(Board& root_board) {
    node_pool_size = 0;
    Node* root_node = allocate_node();
    if (!root_node) {
        return -1;
    }
    root_node->board = root_board;

    root_node->parent = nullptr;
    root_node->visits = 0;
    root_node->wins = 0;
    root_node->num_children = 0;
    root_node->is_terminal = root_board.check_winner();
    root_node->move_from_parent = -1;

    if (root_board.dice == -1) {
        for (int i = 0; i < 6; i++) {
            root_board.moves[i][0] = -1;
            root_board.moves[i][1] = -1;
            root_board.move_count = 6;
        }
    } else {
        root_board.generate_moves();
        for (int j = 0; j < root_board.move_count; j += PIECE_NUM) {
            int move_id = j / PIECE_NUM;
            if (root_board.moves[move_id][1] == 24 || root_board.moves[move_id][1] == 0) {
                return j;
            }
        }
    }

    root_node->num_untried_moves = root_board.move_count;

    int total_simulation_count = 0;

    Board* d_board;
    int* d_result;
    cudaMalloc(&d_board, sizeof(Board));
    cudaMalloc(&d_result, sizeof(int));

    int batch_size = 512;  // Number of threads per block
    int num_blocks = (BATCH_SIZE + batch_size - 1) / batch_size;
    int shared_mem_size = batch_size * sizeof(int);

    curandState* devStates;
    cudaMalloc((void**)&devStates, BATCH_SIZE * sizeof(curandState));
    setup_kernel<<<num_blocks, batch_size>>>(devStates, time(NULL));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    while (total_simulation_count < MAX_SIMULATION_COUNT) {
        Node* node = root_node;
        Board board = root_board;

        while (!node->is_terminal && node->num_untried_moves == 0) {
            if (root_node->num_children == 1) {
                return root_node->children[0]->move_from_parent;
            }
            node = select_child(node);
            assert(node);
            board = node->board;
        }

        if (!node->is_terminal) {
            if (node->num_untried_moves > 0) {
                int move = node->num_untried_moves - 1;
                node->num_untried_moves--;
                
                if (board.piece_position[board.moving_color ^ 1][(node->num_untried_moves) % 6] == -1) {
                    continue;
                }
                
                if (board.dice == -1) {
                    board.moving_color ^= 1;
                } else {
                    board.move(move);
                }

                Node* child_node = allocate_node();
                if (!child_node) {
                    return -1;
                }

                child_node->parent = node;
                child_node->visits = 0;
                child_node->wins = 0;
                child_node->num_children = 0;
                child_node->is_terminal = board.check_winner();
                child_node->move_from_parent = move;

                if (!child_node->is_terminal) {
                    board.generate_moves();
                    child_node->num_untried_moves = board.move_count;
                } else {
                    child_node->num_untried_moves = 0;
                }
                child_node->board = board;

                node->children[node->num_children++] = child_node;

                node = child_node;
            }
        }

        cudaMemcpy(d_board, &(node->board), sizeof(Board), cudaMemcpyHostToDevice);
        cudaMemset(d_result, 0, sizeof(int));

        simulate_kernel<<<num_blocks, batch_size, shared_mem_size>>>(devStates, d_board, d_result, batch_size);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy data back to host
        int total_simulation_result;
        cudaMemcpy(&total_simulation_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        Node* temp_node = node;
        while (temp_node != nullptr) {
            temp_node->visits += BATCH_SIZE;
            temp_node->wins += total_simulation_result;
            total_simulation_result = BATCH_SIZE - total_simulation_result;

            temp_node = temp_node->parent;
        }

        total_simulation_count += BATCH_SIZE;
    }

    int best_move = -1;
    float best_win_rate = -1;
    for (int i = 0; i < root_node->num_children; i++) {
        Node* child = root_node->children[i];
        float win_rate = (float)child->wins / (float)child->visits;

        int step_id = child->move_from_parent / PIECE_NUM;
        int step_start_position = root_board.moves[step_id][0], step_destination = root_board.moves[step_id][1];
        int moving_piece = root_board.board[step_start_position] - root_board.moving_color * PIECE_NUM;
        printf("child %d move %d to %d: %d/%d, winrate:%f\n", child->move_from_parent, moving_piece + 1, step_destination, child->wins, child->visits, win_rate);

        if (win_rate > best_win_rate) {
            best_win_rate = win_rate;
            best_move = child->move_from_parent;
        }
    }
    printf("total_pruned_num: %d\n", total_pruned_num);

    cudaFree(d_board);
    cudaFree(d_result);
    return best_move;
}

int Board::decide() {
    return MCTS(*this);
}

int Board::first_move_decide_dice() {
    return MCTS(*this);
}
