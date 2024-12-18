#include "board.hpp"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "tables.hpp"

void initialize_device_constants() {
    cudaMemcpyToSymbol(d_movable_piece_table, h_movable_piece_table, sizeof(h_movable_piece_table));
    cudaMemcpyToSymbol(d_step_table, h_step_table, sizeof(h_step_table));
}
void Board::init_with_piecepos(int input_piecepos[2][6], char input_color) {
    moving_color = input_color;
    memcpy(piece_position, input_piecepos, 12 * sizeof(int));
    memset(board, -1, 25 * sizeof(int));
    piece_bits[RED] = 0;
    piece_bits[BLUE] = 0;
    for (int _color = 0; _color < 2; _color++) {
        for (int _piece = 0; _piece < 6; _piece++) {
            int position = input_piecepos[_color][_piece];
            if (position >= 0) {
                board[position] = _piece + 6 * _color;
                piece_bits[_color] += bit_mask(_piece);
            }
        }
    }
}
__host__ __device__ void Board::move(int id_with_dice) {
    int move_id = id_with_dice / PIECE_NUM;
    dice = id_with_dice % PIECE_NUM;
    int start = moves[move_id][0], destination = moves[move_id][1];
    int moving_piece = board[start], dest_piece = board[destination];
    board[start] = -1;
    if (dest_piece >= 0) {
        if (dest_piece < PIECE_NUM) {
            piece_bits[RED] &= ~bit_mask(dest_piece);
            piece_position[RED][dest_piece] = -1;
        } else {
            piece_bits[BLUE] &= ~bit_mask(dest_piece - PIECE_NUM);
            piece_position[BLUE][dest_piece - PIECE_NUM] = -1;
        }
    }
    piece_position[moving_color][moving_piece % PIECE_NUM] = destination;
    board[destination] = moving_piece;
    moving_color ^= 1;
}
__host__ __device__ void Board::generate_moves() {
    int movable_piece1, movable_piece2;
    int *piece1_steps, *piece2_steps;
    int piece1_pos, piece2_pos;
#ifdef __CUDA_ARCH__
    movable_piece1 = d_movable_piece_table[piece_bits[moving_color]][dice][0];
#else
    movable_piece1 = h_movable_piece_table[piece_bits[moving_color]][dice][0];
#endif
    // movable_piece1 will always != -1
    assert(movable_piece1 != -1);

#ifdef __CUDA_ARCH__
    movable_piece2 = d_movable_piece_table[piece_bits[moving_color]][dice][1];
#else
    movable_piece2 = h_movable_piece_table[piece_bits[moving_color]][dice][1];
#endif

    if (movable_piece2 == -1) {
        piece1_pos = piece_position[moving_color][movable_piece1];
#ifdef __CUDA_ARCH__
        piece1_steps = (int *)d_step_table[moving_color][piece1_pos];
#else
        piece1_steps = (int *)h_step_table[moving_color][piece1_pos];
#endif
        move_count = piece1_steps[3];
    } else {
        piece1_pos = piece_position[moving_color][movable_piece1];
        piece2_pos = piece_position[moving_color][movable_piece2];

#ifdef __CUDA_ARCH__
        piece1_steps = (int *)d_step_table[moving_color][piece1_pos];
        piece2_steps = (int *)d_step_table[moving_color][piece2_pos];
#else
        piece1_steps = (int *)h_step_table[moving_color][piece1_pos];
        piece2_steps = (int *)h_step_table[moving_color][piece2_pos];
#endif

        move_count = piece1_steps[3] + piece2_steps[3];
    }
    // combine piece1_steps and piece2_steps
    int i = 0;
    for (; i < piece1_steps[3]; i++) {
        moves[i][0] = piece1_pos;
        moves[i][1] = piece1_steps[i];
    }
    for (; i < move_count; i++) {
        moves[i][0] = piece2_pos;
        moves[i][1] = piece2_steps[i - piece1_steps[3]];
    }
    move_multiplier = move_count;
    move_count *= PIECE_NUM;
    return;
}
__host__ __device__ bool Board::check_winner() {
    if (moving_color == RED) {
        if (piece_bits[RED] == 0 || (board[0] >= PIECE_NUM))
            return true;
        return false;
    }
    // blue
    else {
        if (piece_bits[BLUE] == 0 || (0 <= board[24] && board[24] < PIECE_NUM))
            return true;
        return false;
    }
}
void Board::print_board() {
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            if (board[5 * y + x] == -1) {
                printf("_ ");
            } else if (board[5 * y + x] < PIECE_NUM) {
                printf("%c ", 'A' + board[5 * y + x]);
            } else {
                printf("%d ", board[5 * y + x] - PIECE_NUM + 1);
            }
        }
        printf("\n");
    }
    printf("dice: %d\n", dice + 1);
    printf("================\n");
}
