// These are not basic function for board.
// write them in separate file makes code cleaner.
// I recommend write all your complicated algorithms in this file.
// Or sort them into even more files to make code even cleaner!

#include <stdio.h>

#include <random>

#include "board/board.hpp"
#include "math_lib/maths.hpp"
#define SIMULATION_BATCH_NUM 1000
#define MAX_SIMULATION_COUNT 600000

// MCS UCB version
// I use max sample number as the simulation limit.
// Recommend: Using wall clock time to precisely limit the time.
// Advanced: Using both max sample number and wall clock time to limit the simiulation time.

int MCS_UCB_argmax(Board child_nodes[], int move_num) {
    // you can change the parameter C of the UCB here
    // if you comment out this line, the default value will be 1.41421
    // ucb_param_C = 1.41421

    unsigned int node_sample_num[64] = {};
    unsigned int win_num[64] = {};
    unsigned int total_sample_num = 0;
    // start simulating
    // Prevent 0 appears in Denominator, do a even sampling first
    for (int i = 0; i < move_num; i++) {
        for (int j = 0; j < SIMULATION_BATCH_NUM; j++) {
            if (child_nodes[i].simulate() == true) {
                win_num[i] += 1;
            }
            node_sample_num[i] += 1;
            total_sample_num += 1;
        }
    }
    // Then do MCS UCB
    while (total_sample_num < MAX_SIMULATION_COUNT) {
        // find the child which has the highest UCB
        int argmax = 0;
        float max_UCB = -1;
        for (int i = 0; i < move_num; i++) {
            float child_UCB = fast_UCB(win_num[i], node_sample_num[i], total_sample_num);
            if (child_UCB > max_UCB) {
                max_UCB = child_UCB;
                argmax = i;
            }
        }
        // do simulation on child[argmax]
        for (int j = 0; j < SIMULATION_BATCH_NUM; j++) {
            if (child_nodes[argmax].simulate() == true) {
                win_num[argmax] += 1;
            }
            node_sample_num[argmax] += 1;
            total_sample_num += 1;
        }
    }
    // Then return best step according to the win rate
    // NOT UCB! NOT UCB! NOT UCB!
    int return_argmax = 0;
    float max_WR = -1;
    for (int i = 0; i < move_num; i++) {
        float child_WR = (float)win_num[i] / (float)node_sample_num[i];
        if (child_WR > max_WR) {
            max_WR = child_WR;
            return_argmax = i;
        }
    }
    return return_argmax;
}

// Except first move, call decide to decide which move to perform
int Board::decide() {
    generate_moves();
    // A nice example for expansion
    // quick and elegant!
    Board child_nodes[64];
    for (int i = 0; i < move_count; i++) {
        Board child = *(this);
        child.move(i);
        child_nodes[i] = child;
    }
    return MCS_UCB_argmax(child_nodes, move_count);
}

// Only used in first move
int Board::first_move_decide_dice() {
    // A nice example for expansion
    // quick and elegant!
    Board child_nodes[PIECE_NUM];
    for (int i = 0; i < PIECE_NUM; i++) {
        Board board_copy = *(this);
        board_copy.dice = i;
        board_copy.moving_color ^= 1;
        child_nodes[i] = board_copy;
    }
    return MCS_UCB_argmax(child_nodes, PIECE_NUM);
}
// You should use mersenne twister and a random_device seed for the simulation
// But no worries, I've done it for you. Hope it can save you some time!
// Call random_num()%num to randomly pick number from 0 to num-1
std::mt19937 random_num(std::random_device{}());

// Very fast and clean simulation!
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