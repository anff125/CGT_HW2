#include <stdio.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <random>

#include "board/board.hpp"

int main() {
    // Disable buffering on stdout and stderr
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

#define COMMAND_UPDATE 0
#define COMMAND_EXIT 1

    struct timespec start_time, end_time;  // To track wall clock time

    while (true) {
        double remain_time;
        int command, moving_color, dice;
        int piece_position[2][6];
        scanf("%d", &command);
        switch (command) {
            case COMMAND_UPDATE: {
                clock_gettime(CLOCK_MONOTONIC, &start_time);  // Start timing

                scanf("%d", &moving_color);
                scanf("%lf", &remain_time);
                for (int _color_ = 0; _color_ < 2; _color_++) {
                    for (int piece_num = 0; piece_num < 6; piece_num++) {
                        scanf("%d", &piece_position[_color_][piece_num]);
                    }
                }
                scanf("%d", &dice);
                Board current_board;
                current_board.init_with_piecepos(piece_position, moving_color);
                current_board.dice = dice;
                current_board.print_board();
                int step;

                if (dice == -1) {
                    step = current_board.first_move_decide_dice();
                    fprintf(stderr, "====>%d %d %d\n", step, 0, 0);
                } else {
                    step = current_board.decide();
                    int step_id = step / PIECE_NUM;
                    int step_dice = step % PIECE_NUM;
                    int step_start_position = current_board.moves[step_id][0], step_destination = current_board.moves[step_id][1];
                    int moving_piece = current_board.board[step_start_position] - moving_color * PIECE_NUM;
                    fprintf(stderr, "====>%d %d %d\n", step_dice, moving_piece, step_destination);
                }

                clock_gettime(CLOCK_MONOTONIC, &end_time);  // End timing

                // Calculate elapsed time in seconds
                double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                                      (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
                printf("Processing time: %.9f seconds\n", elapsed_time);

                break;
            }
            case COMMAND_EXIT: {
                printf("exit normally...");
                return 0;
            }
            default:
                return 0;
        }
    }
    return 0;
}
