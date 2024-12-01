#ifndef __BOARD__
#define __BOARD__ 1

#include <vector>

#define RED 0
#define BLUE 1

#define PIECE_NUM 6

enum Direction { DIAGONAL,
                 VERTICAL,
                 HORIZONTAL };

typedef struct _board {
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
    void move(int id_with_dice);
    void generate_moves();
    bool check_winner();
    void print_board();

    // not basic functions, written in decide.cpp
    bool simulate();
    bool simulate(int* positions, Direction* directions, int* move_color, int& moves_count);
    int decide();
    int first_move_decide_dice();
} Board;
#endif