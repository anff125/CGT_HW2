#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "pcg_random.hpp"

#include <limits>
#include <random>

#include "board/board.hpp"
#include "math_lib/maths.hpp"
#define MAX_NODES 1000000
#define MAX_SIMULATION_COUNT 6000000
#define BATCH_SIZE 1000

pcg_extras::seed_seq_from<std::random_device> seed_source; // Create a seed source from random_device
pcg32 random_num(seed_source); // Pass the seed source to initialize the RNG


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
    if (node_pool_size >= MAX_NODES) {
        // Handle error
        assert(false);
        return nullptr;
    }
    return &node_pool[node_pool_size++];
}

Node* select_child(Node* node) {
    Node* best_child = nullptr;
    float best_ucb = -1.0f;
    for (int i = 0; i < node->num_children; i++) {
        Node* child = node->children[i];
        float ucb = fast_UCB(child->wins, child->visits, node->visits);
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_child = child;
        }
    }
    return best_child;
}

bool Board::simulate() {
    Board board_copy = *(this);
    // run until game ends.
    while (!board_copy.check_winner()) {
        board_copy.generate_moves();

        for (int i = 0; i < board_copy.move_count / PIECE_NUM; i++) {
            if (board_copy.moves[i][1] == 0 || board_copy.moves[i][1] == 24) {
                return board_copy.moving_color != moving_color;
            }
        }

        board_copy.move(random_num() % board_copy.move_count);
    }
    // game ends! find winner.
    // Win!
    if (board_copy.moving_color == moving_color)
        return true;
    // Lose!
    else
        return false;
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
        // if (total_simulation_count % 100000 == 0) {
        //     fprintf(stderr, "total_simulation_count: %d\n", total_simulation_count);
        // }

        // Selection
        while (!node->is_terminal && node->num_untried_moves == 0) {
            node = select_child(node);
            board = node->board;  // Use the board from the child node directly
        }

        // Expansion
        if (!node->is_terminal) {
            if (node->num_untried_moves > 0) {
                // Expand one of the untried moves
                node->num_untried_moves--;
                int move = node->num_untried_moves;

                board.move(move);

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
        if (node->is_terminal) {
            // Backpropagation
            while (node != nullptr) {
                node->visits += BATCH_SIZE;

                // Determine wins for the current node based on the player
                int winsFornode;
                if (node->board.moving_color == root_board.moving_color) {
                    // It's your turn at this node
                    winsFornode = BATCH_SIZE;
                } else {
                    // It's your opponent's turn at this node
                    winsFornode = BATCH_SIZE - BATCH_SIZE;
                }

                node->wins += winsFornode;
                node = node->parent;
            }
        }

        int result = 0;
        // Simulation
        for (int i = 0; i < BATCH_SIZE; i++) {
            result += board.simulate();
        }

        // Backpropagation
        while (node != nullptr) {
            node->visits += BATCH_SIZE;

            // Determine wins for the current node based on the player
            int winsFornode;
            if (node->board.moving_color != root_board.moving_color) {
                // It's your turn at this node
                winsFornode = result;
            } else {
                // It's your opponent's turn at this node
                winsFornode = BATCH_SIZE - result;
            }

            node->wins += winsFornode;
            node = node->parent;
        }

        total_simulation_count += BATCH_SIZE;
    }

    // At the end, select the child of root_node with the highest win rate
    int best_move = -1;
    float best_win_rate = -1;
    for (int i = 0; i < root_node->num_children; i++) {
        Node* child = root_node->children[i];
        float win_rate = (float)child->wins / (float)child->visits;

        int step_id = child->move_from_parent / PIECE_NUM;
        int step_start_position = root_board.moves[step_id][0], step_destination = root_board.moves[step_id][1];
        int moving_piece = root_board.board[step_start_position] - root_board.moving_color * PIECE_NUM;

        printf("child %d move %d to %d: %d/%d, winrate:%f\n", child->move_from_parent, moving_piece + 1, step_destination, root_node->children[i]->wins, root_node->children[i]->visits, win_rate);
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