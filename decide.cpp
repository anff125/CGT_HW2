#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <limits>
#include <random>

#include "board/board.hpp"
#include "math_lib/maths.hpp"
#include "pcg_random.hpp"
#define MAX_NODES 3002
#define MAX_SIMULATION_COUNT 3000000
#define BATCH_SIZE 1000
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
    int rave_wins;
    int rave_visits;
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

    node->rave_wins = 0;
    node->rave_visits = 0;

    return node;
}

int total_pruned_num = 0;
#define RATIO_PARAM 0.125
#define MIN_VISITS_FOR_PRUNING 5000

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
        float standard_value, rave_value, combined_value, beta;

        // Standard UCB value using fast_UCB
        if (child_visits > 0) {
            standard_value = fast_UCB(child->wins, child_visits, parent_visits);
        } else {
            // Encourage exploration of unvisited nodes
            standard_value = std::numeric_limits<float>::infinity();
        }

        // RAVE value
        int rave_visits = node->rave_visits;
        if (rave_visits > 0) {
            float rave_win_rate = (float)node->rave_wins / (float)rave_visits;
            rave_value = rave_win_rate;
        } else {
            rave_value = 0.5f;  // Default value
        }

        // Compute beta
        float k = 0.00001;  // RAVE constant
        beta = (float)rave_visits / (float)(rave_visits + parent_visits + k * rave_visits * parent_visits);

        combined_value = beta * rave_value + (1 - beta) * standard_value;
        //combined_value = standard_value;

        if (combined_value > best_value) {
            best_value = combined_value;
            best_child = child;
        }
    }
    return best_child;
}

// Use a fixed-size array for valid moves
int valid_pieces[6];

Board path_boards[SEARCH_DEPTH];

bool Board::simulate(int* rave_visits, int* rave_wins) {
    Board board_copy = *this;
    int search_depth = 0;
    bool rave[25][3][2] = {0};  // pos, direction, color

    // Run until the game ends
    while (!board_copy.check_winner()) {
        board_copy.generate_moves();
        int valid_pieces_count = 0;

        // Don't pick dice that the opponent can win.
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
        // If no valid moves are available, break the loop
        if (valid_pieces_count == 0) {
            return board_copy.moving_color == moving_color;
        }

        int random_index = random_num() % valid_pieces_count;
        int move = valid_pieces[random_index];
        move += PIECE_NUM * (random_num() % board_copy.move_mutiplier);

        // rave visited
        int move_id = move / PIECE_NUM;
        int start = board_copy.moves[move_id][0];
        int destination = board_copy.moves[move_id][1];
        Direction direction = get_direction(start, destination);
        rave[start][direction][board_copy.moving_color] = true;
        path_boards[search_depth++] = board_copy;

        board_copy.move(move);
    }

    // Determine the winner
    if (board_copy.moving_color == moving_color) {
        // rave update
        for (int i = 0; i < search_depth; i++) {
            for (int j = 0; j < path_boards[i].move_count; j += PIECE_NUM) {
                int move_id = j / PIECE_NUM;
                int start = path_boards[i].moves[move_id][0];
                int destination = path_boards[i].moves[move_id][1];
                Direction direction = get_direction(start, destination);
                if (rave[start][direction][path_boards[i].moving_color]) {
                    (*rave_visits)++;
                    (*rave_wins)++;
                }
            }
        }
        return true;
    } else {
        // rave update
        for (int i = 0; i < search_depth; i++) {
            for (int j = 0; j < path_boards[i].move_count; j += PIECE_NUM) {
                int move_id = j / PIECE_NUM;
                int start = path_boards[i].moves[move_id][0];
                int destination = path_boards[i].moves[move_id][1];
                Direction direction = get_direction(start, destination);
                if (rave[start][direction][path_boards[i].moving_color]) {
                    (*rave_visits)++;
                }
            }
        }
        return false;
    }
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
        for (int j = 0; j < root_board.move_count; j += PIECE_NUM) {
            int move_id = j / PIECE_NUM;
            if (root_board.moves[move_id][1] == 24 || root_board.moves[move_id][1] == 0) {
                return j;
            }
        }
    }

    root_node->num_untried_moves = root_board.move_count;

    int total_simulation_count = 0;

    while (total_simulation_count < MAX_SIMULATION_COUNT) {
        Node* node = root_node;
        Board board = root_board;

        // Selection
        while (!node->is_terminal && node->num_untried_moves == 0) {
            if (root_node->num_children == 1) {
                return root_node->children[0]->move_from_parent;
            }
            node = select_child(node);
            board = node->board;  // Use the board from the child node directly
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
            }
        }

        // Simulation
        int total_simulation_result = 0;
        int rave_wins = 0;
        int rave_visits = 0;
        for (int sim = 0; sim < BATCH_SIZE; sim++) {
            total_simulation_result += node->board.simulate(&rave_visits, &rave_wins);
        }

        // Backpropagation
        Node* temp_node = node;
        while (temp_node != nullptr) {
            // Update standard statistics
            temp_node->visits += BATCH_SIZE;
            temp_node->wins += total_simulation_result;
            total_simulation_result = BATCH_SIZE - total_simulation_result;

            // Update RAVE statistics
            temp_node->rave_visits += rave_visits;
            temp_node->rave_wins += rave_wins;
            rave_wins = rave_visits - rave_wins;

            temp_node = temp_node->parent;
        }

        total_simulation_count += BATCH_SIZE;
    }

    // Select the best move
    int best_move = -1;
    float best_win_rate = -1;
    for (int i = 0; i < root_node->num_children; i++) {
        Node* child = root_node->children[i];
        float win_rate = (float)child->wins / (float)child->visits;

        // print child info for analysis
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
    return best_move;
}

int Board::decide() {
    return MCTS(*this);
}

int Board::first_move_decide_dice() {
    return MCTS(*this);
}
