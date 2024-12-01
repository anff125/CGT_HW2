#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <limits>
#include <random>

#include "board/board.hpp"
#include "math_lib/maths.hpp"
#include "pcg_random.hpp"
#define MAX_NODES 6002
#define MAX_SIMULATION_COUNT 6000000
#define BATCH_SIZE 1000
#define MIN_VISITS_FOR_PRUNING 2000
#define SEARCH_DEPTH 64

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

    // Additional fields for RAVE
    int rave_wins[25][3];
    int rave_visits[25][3];
};

Node node_pool[MAX_NODES];
int node_pool_size = 0;

Node* allocate_node() {
    Node* node;

    if (node_pool_size >= MAX_NODES) {
        // Handle error
        assert(false);
        return nullptr;
    }
    node = &node_pool[node_pool_size++];

    // Initialize the node
    node->visits = 0;
    node->wins = 0;
    node->num_children = 0;
    node->is_terminal = false;
    node->move_from_parent = -1;
    node->parent = nullptr;
    node->num_untried_moves = 0;
    // Initialize RAVE statistics
    for (int i = 0; i < 25; i++) {
        for (int j = 0; j < 3; j++) {
            node->rave_wins[i][j] = 0;
            node->rave_visits[i][j] = 0;
        }
    }
    return node;
}

const float C = sqrt(2.0);
typedef struct {
    int index;
    float LCB;
} LCBStat;
// Comparison function for qsort
int compare_LCB(const void* a, const void* b) {
    float diff = ((LCBStat*)a)->LCB - ((LCBStat*)b)->LCB;
    if (diff < 0)
        return -1;
    else if (diff > 0)
        return 1;
    else
        return 0;
}

// Binary search function to find the smallest LCB ≥ UCB_i
int binary_search_LCB(LCBStat lcb_stats[], int n, float UCB_i, int index_i) {
    int left = 0, right = n - 1;
    int pos = n;  // Default position if no LCB ≥ UCB_i is found

    while (left <= right) {
        int mid = (left + right) / 2;
        float LCB_mid = lcb_stats[mid].LCB;

        if (LCB_mid >= UCB_i) {
            if (lcb_stats[mid].index != index_i) {
                pos = mid;
            }
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return pos;
}

Node* select_child(Node* node) {
    const int num_children = node->num_children;
    int valid_children = 0;

    float UCB_values[64];
    float LCB_values[64];
    int pruned[64] = {0};

    LCBStat lcb_stats[64];

    // Compute UCB and LCB for each child
    for (int i = 0; i < num_children; i++) {
        Node* child = node->children[i];
        if (child->visits > MIN_VISITS_FOR_PRUNING) {
            UCB_values[i] = fast_UCB(child->wins, child->visits, node->visits);
            LCB_values[i] = fast_LCB(child->wins, child->visits, node->visits);
            lcb_stats[valid_children].index = i;
            lcb_stats[valid_children].LCB = LCB_values[i];
            valid_children++;
        } else {
            pruned[i] = 0;
        }
    }

    // Sort lcb_stats[] based on LCB in ascending order
    qsort(lcb_stats, valid_children, sizeof(LCBStat), compare_LCB);

    // For each child, check if it can be pruned
    for (int idx = 0; idx < valid_children; idx++) {
        int i = lcb_stats[idx].index;
        if (pruned[i]) continue;

        float UCB_i = UCB_values[i];

        // Perform binary search to find the smallest LCB ≥ UCB_i
        int pos = binary_search_LCB(lcb_stats, valid_children, UCB_i, i);

        bool can_prune = false;
        for (int j = pos; j < valid_children; j++) {
            if (lcb_stats[j].index != i) {
                can_prune = true;
                break;
            }
        }

        if (can_prune) {
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
        }
    }
    node->num_children = new_num_children;

    Node* best_child = nullptr;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < node->num_children; i++) {
        Node* child = node->children[i];
        int move = child->move_from_parent;

        // Decode move to get position and direction
        int move_id = move / PIECE_NUM;
        int start = node->board.moves[move_id][0];
        int destination = node->board.moves[move_id][1];
        Direction direction = get_direction(start, destination);

        int child_visits = child->visits;
        int parent_visits = node->visits;
        float standard_value, rave_value, combined_value, beta;

        // Standard UCB value using fast_UCB
        if (child_visits > 0) {
            standard_value = fast_UCB(child->wins, child_visits, parent_visits);
        } else {
            // Encourage exploration of unvisited nodes
            standard_value = std::numeric_limits<float>::infinity();
        }

        // RAVE value
        int rave_visits = node->rave_visits[start][direction];
        if (rave_visits > 0) {
            float rave_win_rate = (float)node->rave_wins[start][direction] / (float)rave_visits;
            rave_value = rave_win_rate;
        } else {
            rave_value = 0.5f;  // Default value
        }

        // Compute beta
        int k = 3000;  // RAVE constant
        beta = sqrtf((float)k / (3 * parent_visits + k));

        combined_value = beta * rave_value + (1 - beta) * standard_value;

        if (combined_value > best_value) {
            best_value = combined_value;
            best_child = child;
        }
    }
    return best_child;
}

bool Board::simulate(int* positions, Direction* directions, int* move_color, int& moves_count) {
    Board board_copy = *this;
    moves_count = 0;
    // Run until the game ends
    while (!board_copy.check_winner()) {
        board_copy.generate_moves();

        int move = random_num() % board_copy.move_count;
        int move_id = move / PIECE_NUM;
        int start = board_copy.moves[move_id][0];
        int destination = board_copy.moves[move_id][1];
        Direction direction = get_direction(start, destination);

        positions[moves_count] = start;
        directions[moves_count] = direction;
        move_color[moves_count] = board_copy.moving_color;
        moves_count++;

        board_copy.move(move);
    }
    // Determine the winner
    return board_copy.moving_color == moving_color;
}

int MCTS(Board& root_board) {
    node_pool_size = 0;  // Reset node pool
    Node* root_node = allocate_node();
    if (!root_node) {
        // Handle error
        return -1;
    }

    root_node->board = root_board;
    root_node->parent = nullptr;
    root_node->visits = 0;
    root_node->wins = 0;
    root_node->num_children = 0;
    root_node->is_terminal = root_board.check_winner();
    root_node->move_from_parent = -1;  // No move leads to the root

    if (root_board.dice == -1) {
        for (int i = 0; i < 6; i++) {
            root_board.moves[i][0] = -1;
            root_board.moves[i][1] = -1;
            root_board.move_count = 6;
        }
    } else {
        root_board.generate_moves();
    }

    root_node->num_untried_moves = root_board.move_count;

    int total_simulation_count = 0;
    while (total_simulation_count < MAX_SIMULATION_COUNT) {
        Node* node = root_node;
        Board board = root_board;

        // Selection
        Node* path_nodes[SEARCH_DEPTH];
        int path_length = 0;
        while (!node->is_terminal && node->num_untried_moves == 0) {
            node = select_child(node);
            if (node == nullptr) {
                break;  // No valid child
            }
            board = node->board;  // Use the board from the child node directly
            path_nodes[path_length++] = node;
        }

        // Expansion
        if (!node->is_terminal) {
            if (node->num_untried_moves > 0) {
                // Expand one of the untried moves
                int move = node->num_untried_moves - 1;
                node->num_untried_moves--;
                if (board.dice == -1) {
                    board.moving_color ^= 1;
                } else {
                    board.move(move);
                }

                Node* child_node = allocate_node();
                if (!child_node) {
                    // Handle error
                    return -1;
                }

                child_node->parent = node;
                child_node->visits = 0;
                child_node->wins = 0;
                child_node->num_children = 0;
                child_node->is_terminal = board.check_winner();
                child_node->move_from_parent = move;

                // Initialize child_node's untried moves
                if (!child_node->is_terminal) {
                    board.generate_moves();
                    child_node->num_untried_moves = board.move_count;
                } else {
                    child_node->num_untried_moves = 0;
                }
                child_node->board = board;

                node->children[node->num_children++] = child_node;

                node = child_node;
                path_nodes[path_length++] = node;
            }
        }
        if (node->num_children == 1) {
            return node->children[0]->move_from_parent;
        }
        // Simulation
        int total_simulation_result = 0;
        int simulation_results[BATCH_SIZE];
        int positions[BATCH_SIZE][64];
        Direction directions[BATCH_SIZE][64];
        int moves_counts[BATCH_SIZE];
        int move_color[BATCH_SIZE][64];

        for (int sim = 0; sim < BATCH_SIZE; sim++) {
            int moves_count = 0;
            simulation_results[sim] = node->board.simulate(positions[sim], directions[sim], move_color[sim], moves_count);
            total_simulation_result += simulation_results[sim];
            moves_counts[sim] = moves_count;
        }

        // Backpropagation
        // Update standard statistics
        Node* temp_node = node;
        while (temp_node != nullptr) {
            temp_node->visits += BATCH_SIZE;
            temp_node->wins += total_simulation_result;
            total_simulation_result = BATCH_SIZE - total_simulation_result;
            temp_node = temp_node->parent;
        }

        // Update RAVE statistics
        for (int i = 0; i < path_length; i++) {
            Node* temp_node = path_nodes[i];
            for (int sim = 0; sim < BATCH_SIZE; sim++) {
                int moves_count = moves_counts[sim];
                for (int j = 0; j < moves_count; j++) {
                    int pos = positions[sim][j];
                    Direction dir = directions[sim][j];

                    int rave_pos = pos;
                    if (temp_node->board.moving_color == BLUE) {
                        rave_pos = 24 - pos;  // Adjust for BLUE player's perspective
                    }

                    temp_node->rave_visits[rave_pos][dir]++;
                    if ((move_color[sim][j] == node->board.moving_color) == simulation_results[sim]) {
                        temp_node->rave_wins[rave_pos][dir]++;
                    }
                }
            }
        }

        total_simulation_count += BATCH_SIZE;
    }

    // Select the best move
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
    return best_move;
}

int Board::decide() {
    return MCTS(*this);
}

int Board::first_move_decide_dice() {
    return MCTS(*this);
}
