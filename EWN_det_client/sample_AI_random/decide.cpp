#include "board/board.hpp"
#include <stdio.h>
#include <random>
// You should use mersenne twister and a random_device seed for the pseudo-random generator
// Call random_num()%num to randomly pick number from 0 to num-1
std::mt19937 random_num(std::random_device{}());
// Except first move, call decide to decide which move to perform
int Board::decide()
{
    generate_moves();
    return random_num() % move_count;
}

// Only used in first move
int Board::first_move_decide_dice()
{
    return random_num() % PIECE_NUM;
}
