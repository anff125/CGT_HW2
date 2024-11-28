#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <limits>
#include <random>

#include "board/board.hpp"
#include "math_lib/maths.hpp"
#define MAX_NODES 1000000
#define MAX_SIMULATION_COUNT 600000

std::mt19937 random_num(std::random_device{}());

struct Node {
    Board board;
    Node* parent;
    Node* children[64];
    int num_children;
    int num_untried_moves;
    int move_from_parent;
    int visits;
    int wins;
    bool is_fully_expanded;
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
    root_node->is_fully_expanded = false;
    root_node->move_from_parent = -1;  // No move leads to the root

    // Initialize untried moves
    root_board.generate_moves();
    root_node->num_untried_moves = root_board.move_count;

    int total_simulation_count = 0;
    while (total_simulation_count < MAX_SIMULATION_COUNT) {
        Node* node = root_node;
        Board board = root_board;
        if (total_simulation_count % 1000 == 0) {
            fprintf(stderr, "total_simulation_count: %d\n", total_simulation_count);
        }

        // Selection
        while (!node->is_terminal && node->num_untried_moves == 0) {
            node = select_child(node);
            if (!node) {
                printf("select_child failed\n");
                assert(false);
                break;
            }  // No valid child
            board = node->board;  // Use the board from the child node directly
        }

        // Expansion
        if (!node->is_terminal) {
            if (node->num_untried_moves > 0) {
                // Expand one of the untried moves
                node->num_untried_moves--;
                int move = node->num_untried_moves;

                Board tempBoard = board;

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
                child_node->is_fully_expanded = false;
                child_node->move_from_parent = move;

                // Initialize child_node's untried moves
                if (!child_node->is_terminal) {
                    board.generate_moves();
                    child_node->num_untried_moves = board.move_count;
                } else {
                    child_node->num_untried_moves = 0;
                    child_node->is_fully_expanded = true;
                }
                child_node->board = board;

                node->children[node->num_children++] = child_node;

                if (node->num_untried_moves == 0) {
                    node->is_fully_expanded = true;
                }

                node = child_node;
            }
        }

        // Simulation
        int result = board.simulate();

        // Backpropagation
        while (node != nullptr) {
            node->visits++;
            if (result) {
                node->wins++;
            }
            node = node->parent;
        }

        total_simulation_count++;
    }

    // At the end, select the child of root_node with the highest visit count
    int best_move = -1;
    int best_visit_count = -1;
    for (int i = 0; i < root_node->num_children; i++) {
        Node* child = root_node->children[i];
        if (child->visits > best_visit_count) {
            best_visit_count = child->visits;
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