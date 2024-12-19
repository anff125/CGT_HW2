#ifndef __BOARD__
#define __BOARD__ 1

#include <cuda_runtime.h>

#define RED 0
#define BLUE 1
#define NOT_YET 0
#define RED_WIN 1
#define BLUE_WIN 2

#define PIECE_NUM 6

extern double remain_time;

void initialize_device_constants();

typedef struct _board
{
    // all captured: piece_bits becomes 0
    unsigned char piece_bits[2];
    int piece_position[2][PIECE_NUM];
    // blank is -1
    int board[25];
    char moves[PIECE_NUM][2];
    int move_count;
    char moving_color;
    char dice;
    void init_with_piecepos(int input_piecepos[2][6], char input_color);
    __host__ __device__ void move(int id_with_dice);
    __host__ __device__ void generate_moves();
    __host__ __device__ char check_winner();
    void print_board();

    // not basic functions, written in decide.cpp
    bool simulate(char root_color);
    int decide();
    int first_move_decide_dice();
} Board;
#endif