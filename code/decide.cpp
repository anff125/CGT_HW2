// These are not basic function for board.
// write them in separate file makes code cleaner.
// I recommend write all your complicated algorithms in this file.
// Or sort them into even more files to make code even cleaner! 
#include <omp.h>
#include "board/board.hpp"
#include "math_lib/maths.hpp"
#include <stdio.h>
#include <random>
#include <assert.h>
#include <cstring>
#include <unordered_set>
#include <time.h>

#define SIMULATION_BATCH_NUM 1024
#define MAX_SIMULATION_COUNT 200000000
#define RAVE_MAX_PLAYOUT 40000
// beta becomes small after N > 1/6 MAX_SIMULATION_COUNT
#define RAVE_4B_SQUARE 0.0005
#define MAX_CHILD 36
#define MAX_NODE 220000
#define MAX_DEPTH 32
#define MINIMAL_SIMULATION_COUNT 150
#define PRUNING_THRESHOLD 0.2
// #define TT_TABLE_SIZE (1<<26)

typedef struct _node {
    Board board;
    bool has_win;
    bool has_lose;
    int move; // the ply from parent to here, for move()
    int p_idx; // parent id, root’s parent is the root
    int c_idx[MAX_CHILD]; // children id
    int depth; // depth, 0 for the root
    int Nchild; // number of children
    int Ntotal; // total # of simulations
    float CsqrtlogN; // c * sqrt(log(Ntotal))
    float sqrtN; // sqrt(Ntotal)
    int Wins;   // number of wins
    float WR; // win rate
// for AMAF
#ifdef USE_AMAF
    int AMAF_Ntotal; // total # of simulations
    float AMAF_CsqrtlogN; // c * sqrt(log(Ntotal))
    float AMAF_sqrtN; // sqrt(Ntotal)
    int AMAF_Wins;   // number of wins
    float AMAF_WR; // win rate
#endif
}Node;

typedef struct _entry {
    int samples;
    int wins;
}Entry;

double remain_time;
int cur_round = 0;
int max_round = 13;

bool winner_table[2][3] =
{
    {false, true, false},
    {false, false, true}
};

// Entry TT[TT_TABLE_SIZE];
Node nodes[MAX_NODE];
Board child_boards[MAX_CHILD];
float child_probability[MAX_CHILD];
int num_of_nodes = 0;

std::mt19937 random_num(std::random_device{}());

// compute the UCB score of nodes[idx]
float get_UCB_by_idx(int idx)
{
    return ((nodes[idx].depth%2) ? (nodes[idx].WR) : (1.0 - nodes[idx].WR)) + 
            (nodes[nodes[idx].p_idx].CsqrtlogN) / (nodes[idx].sqrtN);
}
#ifdef USE_AMAF
float get_RAVE_UCB_by_idx(int idx)
{
    float v1 = ((nodes[idx].depth%2) ? (nodes[idx].WR) : (1.0 - nodes[idx].WR)) + 
            (nodes[nodes[idx].p_idx].CsqrtlogN) / (nodes[idx].sqrtN);
    float v2 = ((nodes[idx].depth%2) ? (nodes[idx].AMAF_WR) : (1.0 - nodes[idx].AMAF_WR)) + 
            (nodes[nodes[idx].p_idx].AMAF_CsqrtlogN) / (nodes[idx].AMAF_sqrtN);
    // float alpha = std::min((float)1.0, (float)nodes[idx].Ntotal / RAVE_MAX_PLAYOUT);
    float beta = 1.0 / ((float)nodes[idx].Ntotal / (float)nodes[idx].AMAF_Ntotal + 1.0 
                            + RAVE_4B_SQUARE * (float)nodes[idx].Ntotal);

    return (1.0 - beta) * v1 + beta * v2;
}
#endif
int encode_ply(int idx, int move)
{
    int move_id = move / PIECE_NUM;
    int dice = move % PIECE_NUM;
    int color = nodes[idx].board.moving_color;
    int start = nodes[idx].board.moves[move_id][0];
    int dest  = nodes[idx].board.moves[move_id][1];
    int piece = nodes[idx].board.board[start];

    return (color << 16) | (dice << 13) | (piece << 10) | (start << 5) | dest;
}
// Zobrist’s hash function given a board
// __uint128_t hash_board(Board& board)
// {
//     __uint128_t key = z_color[board.moving_color];
//     for (int i = 0; i < PIECE_NUM; i++) {
//         int pos = board.piece_position[RED][i];
//         if (pos != -1){
//             key ^= zobrist[i][pos];
//         }
//     }
//     for (int i = 0; i < PIECE_NUM; i++) {
//         int pos = board.piece_position[BLUE][i];
//         if (pos != -1){
//             key ^= zobrist[i + PIECE_NUM][pos];
//         }
//     }

//     return key;
// }
// // given the key of the board and the next move
// __uint128_t hash_move(__uint128_t& key, Board& board, int move)
// {
//     // nomatter which color first
//     __uint128_t res_key = (key ^ z_color[RED]) ^ z_color[BLUE];
//     int move_id = move / PIECE_NUM;
//     int start = board.moves[move_id][0], dest = board.moves[move_id][1];
//     int moving_piece = board.board[start], dest_piece = board.board[dest];
//     if (dest_piece >= 0)
//     {
//         res_key ^= zobrist[moving_piece][start] ^ zobrist[moving_piece][dest]
//                     ^ zobrist[dest_piece][dest];
//     }else
//     {
//         res_key ^= zobrist[moving_piece][start] ^ zobrist[moving_piece][dest];
//     }

//     return res_key;
// }

void init_node(int idx, Board& board) {
    nodes[idx].board = board;

    nodes[idx].has_win = false;
    nodes[idx].has_lose = false;
    nodes[idx].p_idx = idx;
    nodes[idx].depth = 0;
    nodes[idx].Nchild = 0;
    nodes[idx].Ntotal = 0;
    nodes[idx].CsqrtlogN = 0.0;
    nodes[idx].sqrtN = 0.0;
    nodes[idx].Wins = 0;
    nodes[idx].WR = 0.0;
#ifdef USE_AMAF
    nodes[idx].AMAF_Ntotal = 0;
    nodes[idx].AMAF_CsqrtlogN = 0.0;
    nodes[idx].AMAF_sqrtN = 0.0;
    nodes[idx].AMAF_Wins = 0;
    nodes[idx].AMAF_WR = 0.0;
#endif
}
void update_node(int idx, int wins, int samples) {
    nodes[idx].Ntotal += samples;
    nodes[idx].Wins += wins;
    nodes[idx].WR = (float)(nodes[idx].Wins) / (float)(nodes[idx].Ntotal);
    nodes[idx].sqrtN = (float)sqrt((double)(nodes[idx].Ntotal));
    nodes[idx].CsqrtlogN = ucb_param_C * (float)sqrt((double)fast_log2(nodes[idx].Ntotal));
}
#ifdef USE_AMAF
void AMAF_update_node(int idx, int wins, int samples) {
    nodes[idx].AMAF_Ntotal += samples;
    nodes[idx].AMAF_Wins += wins;
    nodes[idx].AMAF_WR = (float)(nodes[idx].AMAF_Wins) / (float)(nodes[idx].AMAF_Ntotal);
    nodes[idx].AMAF_sqrtN = (float)sqrt((double)(nodes[idx].AMAF_Ntotal));
    nodes[idx].AMAF_CsqrtlogN = ucb_param_C * (float)sqrt((double)fast_log2(nodes[idx].AMAF_Ntotal));
}
#endif

// Selection: Traverses the tree using UCB1 to find the PV path.
int select(int root_idx) {
    int current_idx = root_idx;
    while (nodes[current_idx].Nchild > 0) {
        int best_child_idx = nodes[current_idx].c_idx[0];
    #ifdef USE_RAVE
        float max_UCB = get_RAVE_UCB_by_idx(best_child_idx);
    #else
        float max_UCB = get_UCB_by_idx(best_child_idx);
    #endif
        for (int i = 1; i < nodes[current_idx].Nchild; ++i) {
            int child_idx = nodes[current_idx].c_idx[i];
        #ifdef USE_RAVE
            float child_UCB = get_RAVE_UCB_by_idx(child_idx);
        #else
            float child_UCB = get_UCB_by_idx(child_idx);
        #endif
            if (child_UCB > max_UCB) {
                max_UCB = child_UCB;
                best_child_idx = child_idx;
            }
        }

        current_idx = best_child_idx;
    }
    return current_idx;
}
// Expansion: Expands a leaf node by generating all possible children.
void expand(int idx, Board& board) {
    nodes[idx].board.generate_moves();
    board = nodes[idx].board;

    nodes[idx].Nchild = board.move_count;
    for (int i = 0; i < board.move_count; ++i)
    {
        Board board_copy = board;
        board_copy.move(i);
        init_node(num_of_nodes, board_copy);

        char result;
        if ((result = board_copy.check_winner()) != NOT_YET)
        {
            // lose terminal node
            if (winner_table[nodes[0].board.moving_color][result] == false)
            {
                // min node
                if (nodes[idx].depth % 2 == 1)
                {
                    num_of_nodes -= i;
                    nodes[idx].Nchild = 1;
                    init_node(num_of_nodes, board_copy);
                    nodes[idx].c_idx[0] = num_of_nodes;
                    nodes[num_of_nodes].has_lose = true;
                    nodes[num_of_nodes].move = i;
                    nodes[num_of_nodes].p_idx = idx;
                    nodes[num_of_nodes].depth = nodes[idx].depth + 1;
                    num_of_nodes++;
                    break;
                }
                nodes[num_of_nodes].has_lose = true;
            }
            // win terminal node
            else{
                // max node
                if (nodes[idx].depth % 2 == 0)
                {
                    num_of_nodes -= i;
                    nodes[idx].Nchild = 1;
                    init_node(num_of_nodes, board_copy);
                    nodes[idx].c_idx[0] = num_of_nodes;
                    nodes[num_of_nodes].has_win = true;
                    nodes[num_of_nodes].move = i;
                    nodes[num_of_nodes].p_idx = idx;
                    nodes[num_of_nodes].depth = nodes[idx].depth + 1;
                    num_of_nodes++;
                    break;
                }
                nodes[num_of_nodes].has_win = true;
            }
        }
        
        nodes[idx].c_idx[i] = num_of_nodes;
        nodes[num_of_nodes].move = i;
        nodes[num_of_nodes].p_idx = idx;
        nodes[num_of_nodes].depth = nodes[idx].depth + 1;
        
        num_of_nodes++;
    }
}
// Backpropagation: Update the nodes on the PV path.
void backpropagate(int idx, int wins, int samples) {
    while(idx != nodes[idx].p_idx){
        update_node(idx, wins, samples);
        idx = nodes[idx].p_idx;
    }
    // root
    update_node(idx, wins, samples);
}
// not use in the end
// it updates TT, if node's samples is more

// Backpropagation with AMAF
#ifdef USE_AMAF
void AMAF_backpropagate(int idx, int wins, int samples, int move)
{
    int total_count = 1;
    int encoded_ply = 0;
    std::unordered_set<int> played_plies;
    while(idx != nodes[idx].p_idx){
        // current_idx == idx
        for (int i = 0; i < nodes[idx].Nchild; i++)
        {
            // should not count the real ply
            if (i == move)  continue;
            encoded_ply = encode_ply(idx, i);
            if (played_plies.count(encoded_ply))
            {
                int child_idx = nodes[idx].c_idx[i];
                AMAF_update_node(child_idx, wins, samples);
                total_count++;
            }
        }
        AMAF_update_node(idx, wins*total_count, samples*total_count);

        // current_idx != idx
        int p_idx = nodes[idx].p_idx;
        int start_parent_move = (nodes[idx].move / PIECE_NUM) * PIECE_NUM;
        int end_parent_move = start_parent_move + PIECE_NUM;
        for (int m = start_parent_move; m < end_parent_move; m++)
        {
            int current_idx = nodes[p_idx].c_idx[m];
            if (nodes[current_idx].Nchild == 0 || current_idx == idx)
                continue;

            int local_count = 0;
            for (int i = 0; i < nodes[current_idx].Nchild; i++)
            {
                encoded_ply = encode_ply(current_idx, i);
                if (played_plies.count(encoded_ply))
                {
                    int child_idx = nodes[idx].c_idx[i];
                    AMAF_update_node(child_idx, wins, samples);
                    local_count++;
                    total_count++;
                }
            }
            AMAF_update_node(current_idx, wins*local_count, samples*local_count);
        }
        

        encoded_ply = encode_ply(idx, move);
        played_plies.insert(encoded_ply);
        move = nodes[idx].move;
        idx = nodes[idx].p_idx;
    }
    // root
    for (int i = 0; i < nodes[idx].Nchild; i++)
    {
        // should not count the real ply
        if (i == move)  continue;
        encoded_ply = encode_ply(idx, i);
        if (played_plies.count(encoded_ply))
        {
            int child_idx = nodes[idx].c_idx[i];
            // update node
            AMAF_update_node(child_idx, wins, samples);
            total_count++;
        }
    }
    AMAF_update_node(idx, wins*total_count, samples*total_count);

    // fprintf(stderr, "count: %d\n", count);
}
#endif

// Monte Carlo Tree Search algorithm
int MCTS(Board& root_board) {
    // if (max_round - cur_round <= 3)
    //     max_round += 5;
    // double availible_time = remain_time / (max_round - cur_round);
    // fprintf(stderr, "availible time: %lf, max round: %d, cur round: %d\n", availible_time, max_round, cur_round);
    cur_round += 1;
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    int root_idx = 0;
    int simulation_num = 0;
    int real_simulation_num = 0;
    //  __uint128_t root_key = hash_board(root_board);

    init_node(root_idx, root_board);
    num_of_nodes = 1;

    int max_depth = 1;
    while (simulation_num < MAX_SIMULATION_COUNT)
    {
        // 0: Time control
        clock_gettime(CLOCK_REALTIME, &end);
        double wall_clock_in_seconds = (double)((end.tv_sec+end.tv_nsec*1e-9) -
                                       (double)(start.tv_sec+start.tv_nsec*1e-9));
        if (wall_clock_in_seconds > 2.0)
            break;

        // 1: Selection
        int selected_node = select(root_idx);
        // Debug
        if (nodes[selected_node].depth > max_depth)
            max_depth = nodes[selected_node].depth;

        // 1.5: check if it's a terminal node 
        if (nodes[selected_node].has_win)  // leaf lose
        {   
            simulation_num += SIMULATION_BATCH_NUM * 3;
            backpropagate(selected_node, SIMULATION_BATCH_NUM * 3, SIMULATION_BATCH_NUM * 3);
            #ifdef USE_AMAF
            AMAF_update_node(selected_node, SIMULATION_BATCH_NUM * 3, SIMULATION_BATCH_NUM * 3);
            AMAF_backpropagate(nodes[selected_node].p_idx, SIMULATION_BATCH_NUM * 3, 
                                SIMULATION_BATCH_NUM * 3, nodes[selected_node].move);
            #endif
            continue;
        }
        // root lose
        else if (nodes[selected_node].has_lose)
        {
            simulation_num += SIMULATION_BATCH_NUM * 3;
            backpropagate(selected_node, 0, SIMULATION_BATCH_NUM * 3);
            #ifdef USE_AMAF
            AMAF_update_node(selected_node, 0, SIMULATION_BATCH_NUM * 3);
            AMAF_backpropagate(nodes[selected_node].p_idx, 0, 
                                SIMULATION_BATCH_NUM * 3, nodes[selected_node].move);
            #endif
            continue;
        }

        // 2: Expansion    
        Board leaf_board;
        expand(selected_node, leaf_board);

        // 3: Simulation
        // __uint128_t hash_key = hash_board(leaf_board);
        int total_sample_num = 0;
        int total_win_num = 0;
        // use node.Nchild here, not board.move_count!
        // Parallelize the for loop using OpenMP
        // #pragma omp parallel for reduction(+:total_sample_num, total_win_num)
        for (int i = 0; i < nodes[selected_node].Nchild; i++)
        {
            int child_idx = nodes[selected_node].c_idx[i];
            int node_sample_num = 0;
            int node_win_num = 0;
            if (nodes[child_idx].has_win)
            {
                node_sample_num += SIMULATION_BATCH_NUM;
                node_win_num += SIMULATION_BATCH_NUM;
            }
            else if (nodes[child_idx].has_lose)
            {
                node_sample_num += SIMULATION_BATCH_NUM;
                // node_win_num += 0;
            }
            else{
                for (int j = 0; j < SIMULATION_BATCH_NUM; j++)
                {
                    child_boards[i] = nodes[child_idx].board;
                    if (child_boards[i].simulate(root_board.moving_color) == true)
                    {
                        node_win_num += 1;
                    }
                }
                node_sample_num += SIMULATION_BATCH_NUM;
                real_simulation_num += SIMULATION_BATCH_NUM;
            }

            // Check if the state(board) is explored before
            // int hash_index = hash_move(hash_key, leaf_board, i) % TT_TABLE_SIZE;
            // if (TT[hash_index].samples) 
            // {
            //     if (TT[hash_index].samples >= 18 * SIMULATION_BATCH_NUM)
            //     {
            //         int multiple = TT[hash_index].samples / SIMULATION_BATCH_NUM;
            //         node_win_num = TT[hash_index].wins / multiple;
            //     }else
            //     {
            //         int multiple = TT[hash_index].samples / SIMULATION_BATCH_NUM;
            //         TT[hash_index].wins += node_win_num;
            //         TT[hash_index].samples += node_sample_num;
            //         node_win_num = (TT[hash_index].wins) / (multiple + 1);
            //     }
            // }
            // else{
            //     TT[hash_index].samples = node_sample_num;
            //     TT[hash_index].wins = node_win_num;
            // }
            
            update_node(child_idx, node_win_num, node_sample_num);

            // 4.0: Backpropagation with AMAF
            #ifdef USE_AMAF
            #pragma omp critical
            {
                AMAF_update_node(child_idx, node_win_num, node_sample_num);
                AMAF_backpropagate(selected_node, node_win_num, node_sample_num, i);
            }
            #endif

            // Aggregate results
            total_sample_num += node_sample_num;
            total_win_num += node_win_num;
        }

        // 4: Backpropagation (all children at once)
        backpropagate(selected_node, total_win_num, total_sample_num);
        simulation_num += total_sample_num;
    } 

    // Choose the best move based on win rate
    int best_move = 0;
    float best_WR = 0.0;
    #ifdef USE_AMAF
    float beta = 1.0 / ((float)nodes[0].Ntotal / (float)nodes[0].AMAF_Ntotal + 1.0 
                            + RAVE_4B_SQUARE * (float)nodes[0].Ntotal);
    for (int i = 0; i < nodes[0].Nchild; ++i) {
        int child_idx = nodes[0].c_idx[i];
        float WR = (1.0 - beta) * nodes[child_idx].WR + beta * nodes[child_idx].AMAF_WR;
        if (WR > best_WR) {
            best_WR = WR;
            best_move = nodes[child_idx].move;
        }
    }
    #else
    for (int i = 0; i < nodes[0].Nchild; ++i) {
        int child_idx = nodes[0].c_idx[i];
        if (nodes[child_idx].WR > best_WR) {
            best_WR = nodes[child_idx].WR;
            best_move = nodes[child_idx].move;
        }
    }
    #endif

    // For debug
    fprintf(stderr, "best WR: %f\n", best_WR);
    fprintf(stderr, "max depth: %d\n", max_depth);
    fprintf(stderr, "expand nodes: %d\n", num_of_nodes);
    fprintf(stderr, "real total samples: %d\n", real_simulation_num);
    #ifdef USE_AMAF
        fprintf(stderr, "amaf total samples: %d\n", nodes[0].AMAF_Ntotal);
    #endif

    return best_move;
}
// not use in the end
float depth_1_MCTS(Board& board, int dice)
{
    int root_idx = 0;
    init_node(root_idx, board);
    num_of_nodes = 1;

    expand(root_idx, board);

    int simulation = 0;
    // start simulating
    // Prevent 0 appears in Denominator, do a even sampling first
    for (int i = 0; i < board.move_count; i++)
    {
        int child_idx = nodes[root_idx].c_idx[i];
        int win_num = 0;
        int node_sample_num = 0;
        for (int j = 0; j < SIMULATION_BATCH_NUM; j++)
        {
            child_boards[i] = nodes[child_idx].board;
            if (child_boards[i].simulate(board.moving_color) == true)
            {
                win_num += 1;
            }
            node_sample_num += 1;
            simulation += 1;
        }

        update_node(child_idx, win_num, node_sample_num);
    }
    // Then do MCS UCB
    while (simulation < MAX_SIMULATION_COUNT / 3)
    {
        // find the child which has the highest UCB
        int best_child = 0;
        float max_UCB = -1;
        for (int i = 0; i < board.move_count; i++)
        {
            int child_idx = nodes[root_idx].c_idx[i];
            float child_UCB = get_UCB_by_idx(child_idx);
            if (child_UCB > max_UCB)
            {
                max_UCB = child_UCB;
                best_child = child_idx;
            }
        }
        // do simulation on best child
        int win_num = 0;
        int node_sample_num = 0;
        for (int j = 0; j < SIMULATION_BATCH_NUM; j++)
        {
            child_boards[best_child] = nodes[best_child].board;
            if (child_boards[best_child].simulate(board.moving_color) == true)
            {
                win_num += 1;
            }
            node_sample_num += 1;
            simulation += 1;
        }
        update_node(best_child, win_num, node_sample_num);
    }
    // Then return best step according to the win rate
    // NOT UCB! NOT UCB! NOT UCB!
    float max_WR = -1;
    for (int i = 0; i < board.move_count; i++)
    {
        int child_idx = nodes[root_idx].c_idx[i];
        float child_WR = nodes[child_idx].WR;
        if (child_WR > max_WR)
        {
            max_WR = child_WR;
        }
    }
    return max_WR;
}

int Board::decide()
{
    generate_moves();
    Board board_copy = *(this);

    return MCTS(board_copy);
}
// Only used in first move
int Board::first_move_decide_dice()
{
    // Always pick the positions 14 or 2
    int target_pos1 = piece_position[moving_color][0] < 12 ? 14 : 2;
    // int target_pos2 = piece_position[moving_color][0] < 12 ? 18 : 6;
    // int target_pos3 = piece_position[moving_color][0] < 12 ? 22 : 10;
    int dice1;

    for (int i = 0; i < PIECE_NUM; i++)
    {
        if (target_pos1 == piece_position[moving_color^1][i])
            dice1 = i;
        // if (target_pos2 == piece_position[moving_color^1][i])
        //     dice2 = i;
        // if (target_pos3 == piece_position[moving_color^1][i])
        //     dice3 = i;
    }
    
    return dice1;
}
// You should use mersenne twister and a random_device seed for the simulation
// But no worries, I've done it for you. Hope it can save you some time!
// Call random_num()%num to randomly pick number from 0 to num-1

// Very fast and clean simulation!
bool Board::simulate(char root_color)
{
    int valid_pieces[PIECE_NUM];
    char result;
    // run until game ends.
    while ((result = this->check_winner()) == NOT_YET)
    {
        this->generate_moves();
        int valid_pieces_count = 0;

        for (int valid_piece = 0; valid_piece < PIECE_NUM; ++valid_piece)
        {
            int pos = this->piece_position[this->moving_color ^ 1][valid_piece];
            if (pos == -1) continue;
            if (this->moving_color == BLUE) {
                pos = 24 - pos;
            }
            if (pos != 1 && pos != 5 && pos != 6) {
                valid_pieces[valid_pieces_count++] = valid_piece;
            }
        }

        // If no valid moves are available, break the loop
        if (valid_pieces_count == 0) {
            if (this->moving_color == RED)
                return winner_table[root_color][BLUE_WIN];
            else
                return winner_table[root_color][RED_WIN];
        }

        int selected_piece = valid_pieces[random_num() % valid_pieces_count];
        int move_id = this->move_count / PIECE_NUM;
        int ply = (random_num() % move_id) * PIECE_NUM + selected_piece;

        // fprintf(stderr, "piece: %d, move_id: %d, ply: %d\n", selected_piece,
        //                             move_id, ply);

        this->move(ply);
    }

    // game ends! find winner.
    return winner_table[root_color][result];
    
}
