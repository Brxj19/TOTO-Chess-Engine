/*****************************************************\
 =====================================================

 * =============*<TOTO CHESS ENGINE(TCE)>*============*
 * ==================AUTHOR: Brxj19===================*
 
 =====================================================
\*****************************************************/




//*****************************************************
//====================================================
//                    headers
//====================================================
//*****************************************************


#include <stdio.h>
#include <string.h>


//defines bit board data type
#define U64 unsigned long long

//FEN debug position

#define empty_board "8/8/8/8/8/8/8/8 w - - "
#define start_position "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 "
#define tricky_position "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 "
#define killer_position "rnbqkb1r/pp1p1pPp/8/2p1pP2/1P1P4/3P3P/P1P1P3/RNBQKBNR w KQkq e6 0 1 "
#define cmk_position "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9 "
/*
   tricky_position
 8  ♜  .  .  .  ♚  .  .  ♜
 7  ♟  .  ♟  ♟  ♛  ♟  ♝  .
 6  ♝  ♞  .  .  ♟  ♞  ♟  .
 5  .  .  .  ♙  ♘  .  .  .
 4  .  ♟  .  .  ♙  .  .  .
 3  .  .  ♘  .  .  ♕  .  ♟
 2  ♙  ♙  ♙  ♗  ♗  ♙  ♙  ♙
 1  ♖  .  .  .  ♔  .  .  ♖

    a  b  c  d  e  f  g  h

  Side to move: WHITE
  Enpassant Square: NO
  Castle Rights: KQkk
*/

//enum board square
enum {
    a8, b8, c8, d8, e8, f8, g8, h8,
    a7, b7, c7, d7, e7, f7, g7, h7,
    a6, b6, c6, d6, e6, f6, g6, h6,
    a5, b5, c5, d5, e5, f5, g5, h5,
    a4, b4, c4, d4, e4, f4, g4, h4,
    a3, b3, c3, d3, e3, f3, g3, h3,
    a2, b2, c2, d2, e2, f2, g2, h2,
    a1, b1, c1, d1, e1, f1, g1, h1,no_sq
   
};

//side to move (colors)
enum { WHITE, BLACK, BOTH };

enum {rook,bishop};

enum {wk = 1, wq = 2, bk = 4, bq = 8};

//enum piece code 
enum { P, N, B, R, Q, K, p, n, b, r, q, k };
//square to coordinates
const char *square_to_coordinates[] = {
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "a0", "b0", "c0", "d0", "e0", "f0", "g0", "h0",
};

//ASCII pieces
char ascii_pieces[12] = "PNBRQKpnbrqk";

//UNICODE pieces
char *unicode_pieces[12] = {"\u2659",
                            "\u2658",
                            "\u2657",
                            "\u2656",
                            "\u2655",
                            "\u2654",
                            "\u265F",
                            "\u265E",
                            "\u265D",
                            "\u265C",
                            "\u265B",
                            "\u265A"
};

//convert ASCII character pieces to encoded constant
int char_pieces[] = {
    ['P'] = P, ['N'] = N, ['B'] = B, ['R'] = R, ['Q'] = Q, ['K'] = K,
    ['p'] = p, ['n'] = n, ['b'] = b, ['r'] = r, ['q'] = q, ['k'] = k
};
//*****************************************************
//====================================================
//                    CHESS BOARD
//====================================================
//*****************************************************

//piece bitboards
U64 bitboards[12];

//occupancy bitboards
U64 occupancies[3];

//side to move
int side = -1;

//enpassant square
int enpassant = no_sq;

//castle rights
int castle;

//*****************************************************
//====================================================
//                    Random Numbers
//====================================================
//*****************************************************


//pseudo random number state
unsigned int random_state = 1804289383;

//generate 32-bit pseudo legal number
unsigned int get_random_U32_number(){
    //xorshift32 algorithm
    unsigned int x = random_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    random_state = x;
    return x;
}

//generate 64-bit pseudo legal number
U64 get_random_U64_number(){
    U64 n1,n2,n3,n4;
    //init 4 random number
    n1 = (U64)(get_random_U32_number() & 0xFFFF);
    n2 = (U64)(get_random_U32_number() & 0xFFFF);
    n3 = (U64)(get_random_U32_number() & 0xFFFF);
    n4 = (U64)(get_random_U32_number() & 0xFFFF);
    
    return n1 | (n2 << 16) | (n3 << 32) | (n4 << 48);
}

//generate magic number candidate
U64 generate_magic_number(){
    return get_random_U64_number() & get_random_U64_number() & get_random_U64_number();
}


//*****************************************************
//====================================================
//                    Bit Manipulation
//====================================================
//*****************************************************


//set/get/pop macros
#define set_bit(bitboard,square) ((bitboard) |= (1ULL<<(square)))
#define get_bit(bitboard,square) ((bitboard) & (1ULL<<(square)))
#define pop_bit(bitboard,square) ((bitboard) &= ~(1ULL<<(square)))

// count the bits in the given bitboard
static inline int count_bits(U64 bitboard){
    int bit_count = 0;

    //consecutively reset least significant 1st bit
    while(bitboard){
        //increment bit_count
        bit_count++;
        //reset least significant 1st bit
        bitboard &= bitboard-1;
    }

    return bit_count;
}

// get the index of the least significant bit
static inline int get_ls1b_index(U64 bitboard){
    //make sure the bitboard is not 0
    if(bitboard){
        //count the trailig bits  befor LS1B that will be only the index of LS1B
        return count_bits((bitboard & -bitboard) - 1);
    }
    //otherwise return -1 i.e illegal index
    else return -1;

}


//*****************************************************
//====================================================
//                   Input and Output
//====================================================
//*****************************************************
//print bitboard
void print_bitboard(U64 bitboard){
    //loop over board ranks
    printf("\n");
    for (int rank = 0 ; rank < 8 ; rank++){
        //loop over board files
        for(int file = 0 ; file < 8 ; file++){
            //convert rank and file to square index 
            int square = rank * 8 + file;
            //print ranks
            if(!file){
                printf(" %d ",8 - rank);
            }
            //print bit state (either 1 or 0)
            printf(" %d",get_bit(bitboard,square) ? 1 : 0);
        }
        //print new line after every rank 
        printf("\n");
    }

    //print files
    printf("\n    a b c d e f g h\n\n");

    //print bitboard in decimal format
    printf("    Bitboard: %llud\n",bitboard);
}

void print_unicode_pieces() {
    for (int i = 0; i < 12; i++) {
        printf("%s ", unicode_pieces[i]);
    }
    printf("\n");
}

//print board function
void print_board(){
    //loop over board ranks 
    for (int rank = 0 ; rank < 8 ; rank++){
        //loop over board files
        for(int file = 0 ; file < 8 ; file++){
            //convert rank and file to square index
            int square = rank * 8 + file;
            //print ranks
            if(!file){
                printf(" %d ",8 - rank);
            }
            //define piece variable
            int piece = -1;

            //loop over 12 pieces bitboards
            for (int bb_piece = P; bb_piece <= k; bb_piece++){
                if(get_bit(bitboards[bb_piece],square)){
                    piece = bb_piece;
                }
            }
            printf(" %s",(piece == -1) ? "." : unicode_pieces[piece]);
        }
        printf("\n");
    }
    //print files
    printf("\n    a b c d e f g h\n\n");
    //print side to move
    printf("  Side to move: %s\n",!side ? "WHITE" : "BLACK");
    //print enpassant square
    printf("  Enpassant Square: %s\n",enpassant == no_sq ? "NO" : square_to_coordinates[enpassant]);
    //print castle rights
    printf("  Castle Rights: %s%s%s%s\n",castle & wk ? "K" : "-",castle & wq ? "Q" : "-",castle & bk ? "k" : "-",castle & bq ? "q" : "-");
}

//parse FEN function
void parse_fen(char *fen ){
    //reset board position (bitboards)
    memset(bitboards,0ULL,sizeof(bitboards));
    //reset occupancies (bitboard)
    memset(occupancies,0ULL,sizeof(occupancies));
    //reset gamestate variables
    side = 0;
    enpassant = no_sq;
    castle = 0;

    //loop over board ranks
    for (int rank = 0 ; rank < 8 ; rank++){
        //loop over board files
        for(int file = 0 ; file < 8 ; file++){
            int square = 8*rank +file;

            //match the ascii piece 
            if((*fen >= 'a' && *fen <= 'z')||(*fen >= 'A' && *fen <= 'Z')){
                //init piece type 
                int piece = char_pieces[*fen];

                //set the piece on the corresponding board
                set_bit(bitboards[piece],square);

                fen++;
            }
            //match the  empty squares
            if(*fen >= '0' && *fen <= '9'){
                //init offset 
                int offset = *fen - '0';
                
                
                int piece = -1;
                //loop over 12 pieces bitboards
                for (int bb_piece = P; bb_piece <= k; bb_piece++){
                    if(get_bit(bitboards[bb_piece],square)){
                        piece = bb_piece;
                    }
                }
                //if no piece on current square
                if(piece == -1)
                    file--;
                file += offset;

                fen++;
            }
            //match the rank seerator
            if(*fen == '/')
                fen++;
        }
    }
    //got to parsing the side to move
    fen++;
    //parse the side to move
    side = (*fen == 'w') ? WHITE : BLACK;
    //got to parsing the castle rights
    fen += 2;
    //parse the castle rights
    while (*fen && *fen != ' ') {
        switch (*fen) {
            case 'K': castle |= wk; break;
            case 'Q': castle |= wq; break;
            case 'k': castle |= bk; break;
            case 'q': castle |= bq; break;
            case '-': break;
        }
        fen++;
    }
    //got to parsing the enpassant square
    fen++;
    //parse the enpassant square
    if(*fen != '-'){
        int file = fen[0] - 'a';
        int rank = 8 - (fen[1] - '0');
        enpassant = 8 * rank + file;
    }
    else{
        enpassant = no_sq;
    }

    //init white occupancy
    for(int piece = P ; piece <= K ; piece++){
        occupancies[WHITE] |= bitboards[piece];
    }
    //init black occupancy
    for(int piece = p ; piece <= k ; piece++){
        occupancies[BLACK] |= bitboards[piece];
    }
    //init both occupancy
    occupancies[BOTH] = occupancies[WHITE] | occupancies[BLACK];

    printf("FEN:'%s'\n",fen);
}


//*****************************************************
//====================================================
//                       Attaks
//====================================================
//*****************************************************

/*

    not A file
 8  0 1 1 1 1 1 1 1
 7  0 1 1 1 1 1 1 1
 6  0 1 1 1 1 1 1 1
 5  0 1 1 1 1 1 1 1
 4  0 1 1 1 1 1 1 1
 3  0 1 1 1 1 1 1 1
 2  0 1 1 1 1 1 1 1
 1  0 1 1 1 1 1 1 1

    a b c d e f g h

    not H file
 8  1 1 1 1 1 1 1 0
 7  1 1 1 1 1 1 1 0
 6  1 1 1 1 1 1 1 0
 5  1 1 1 1 1 1 1 0
 4  1 1 1 1 1 1 1 0
 3  1 1 1 1 1 1 1 0
 2  1 1 1 1 1 1 1 0
 1  1 1 1 1 1 1 1 0

    a b c d e f g h

    not HG file
 8  1 1 1 1 1 1 0 0
 7  1 1 1 1 1 1 0 0
 6  1 1 1 1 1 1 0 0
 5  1 1 1 1 1 1 0 0
 4  1 1 1 1 1 1 0 0
 3  1 1 1 1 1 1 0 0
 2  1 1 1 1 1 1 0 0
 1  1 1 1 1 1 1 0 0

    a b c d e f g h

    not AB file
 8  0 0 1 1 1 1 1 1
 7  0 0 1 1 1 1 1 1
 6  0 0 1 1 1 1 1 1
 5  0 0 1 1 1 1 1 1
 4  0 0 1 1 1 1 1 1
 3  0 0 1 1 1 1 1 1
 2  0 0 1 1 1 1 1 1
 1  0 0 1 1 1 1 1 1

    a b c d e f g h
*/


//not A file constant 
const U64 not_a_file = 18374403900871474942ULL;

//not H file constant
const U64 not_h_file = 9187201950435737471ULL;

// not HG file constant
const U64 not_hg_file = 4557430888798830399ULL;

//not AB file constant
const U64 not_ab_file = 18229723555195321596ULL;

//pawn attacks table [side][square]
U64 pawn_attacks[2][64];

//knight attacks table [square]
U64 knight_attacks[64];

//king attacks table [square]
U64 king_attacks[64];

//bishop attack table[square][occupancy] and mask
U64 bishop_mask[64];
U64 bishop_attacks[64][512];

//rook attack table[square][occupancy] and mask
U64 rook_mask[64];
U64 rook_attacks[64][4096];


//relevant occupancy bit count for every square for bishop,rook & queen on board
const int bishop_relevant_bits[64] = {
                                    6, 5, 5, 5, 5, 5, 5, 6,
                                    5, 5, 5, 5, 5, 5, 5, 5,
                                    5, 5, 7, 7, 7, 7, 5, 5,
                                    5, 5, 7, 9, 9, 7, 5, 5,
                                    5, 5, 7, 9, 9, 7, 5, 5,
                                    5, 5, 7, 7, 7, 7, 5, 5,
                                    5, 5, 5, 5, 5, 5, 5, 5,
                                    6, 5, 5, 5, 5, 5, 5, 6,
};

const int rook_relevant_bits[64] = {
                                12, 11, 11, 11, 11, 11, 11, 12,
                                11, 10, 10, 10, 10, 10, 10, 11,
                                11, 10, 10, 10, 10, 10, 10, 11,
                                11, 10, 10, 10, 10, 10, 10, 11,
                                11, 10, 10, 10, 10, 10, 10, 11,
                                11, 10, 10, 10, 10, 10, 10, 11,
                                11, 10, 10, 10, 10, 10, 10, 11,
                                12, 11, 11, 11, 11, 11, 11, 12,
};


//rook magic numbers 
U64 rook_magic_numbers[64] = {
    0x8a80104000800020ULL,
    0x140002000100040ULL,
    0x2801880a0017001ULL,
    0x100081001000420ULL,
    0x200020010080420ULL,
    0x3001c0002010008ULL,
    0x8480008002000100ULL,
    0x2080088004402900ULL,
    0x800098204000ULL,
    0x2024401000200040ULL,
    0x100802000801000ULL,
    0x120800800801000ULL,
    0x208808088000400ULL,
    0x2802200800400ULL,
    0x2200800100020080ULL,
    0x801000060821100ULL,
    0x80044006422000ULL,
    0x100808020004000ULL,
    0x12108a0010204200ULL,
    0x140848010000802ULL,
    0x481828014002800ULL,
    0x8094004002004100ULL,
    0x4010040010010802ULL,
    0x20008806104ULL,
    0x100400080208000ULL,
    0x2040002120081000ULL,
    0x21200680100081ULL,
    0x20100080080080ULL,
    0x2000a00200410ULL,
    0x20080800400ULL,
    0x80088400100102ULL,
    0x80004600042881ULL,
    0x4040008040800020ULL,
    0x440003000200801ULL,
    0x4200011004500ULL,
    0x188020010100100ULL,
    0x14800401802800ULL,
    0x2080040080800200ULL,
    0x124080204001001ULL,
    0x200046502000484ULL,
    0x480400080088020ULL,
    0x1000422010034000ULL,
    0x30200100110040ULL,
    0x100021010009ULL,
    0x2002080100110004ULL,
    0x202008004008002ULL,
    0x20020004010100ULL,
    0x2048440040820001ULL,
    0x101002200408200ULL,
    0x40802000401080ULL,
    0x4008142004410100ULL,
    0x2060820c0120200ULL,
    0x1001004080100ULL,
    0x20c020080040080ULL,
    0x2935610830022400ULL,
    0x44440041009200ULL,
    0x280001040802101ULL,
    0x2100190040002085ULL,
    0x80c0084100102001ULL,
    0x4024081001000421ULL,
    0x20030a0244872ULL,
    0x12001008414402ULL,
    0x2006104900a0804ULL,
    0x1004081002402ULL
};

//bishop magic numbers
U64 bishop_magic_numbers[64] = {
    0x40040844404084ULL,
    0x2004208a004208ULL,
    0x10190041080202ULL,
    0x108060845042010ULL,
    0x581104180800210ULL,
    0x2112080446200010ULL,
    0x1080820820060210ULL,
    0x3c0808410220200ULL,
    0x4050404440404ULL,
    0x21001420088ULL,
    0x24d0080801082102ULL,
    0x1020a0a020400ULL,
    0x40308200402ULL,
    0x4011002100800ULL,
    0x401484104104005ULL,
    0x801010402020200ULL,
    0x400210c3880100ULL,
    0x404022024108200ULL,
    0x810018200204102ULL,
    0x4002801a02003ULL,
    0x85040820080400ULL,
    0x810102c808880400ULL,
    0xe900410884800ULL,
    0x8002020480840102ULL,
    0x220200865090201ULL,
    0x2010100a02021202ULL,
    0x152048408022401ULL,
    0x20080002081110ULL,
    0x4001001021004000ULL,
    0x800040400a011002ULL,
    0xe4004081011002ULL,
    0x1c004001012080ULL,
    0x8004200962a00220ULL,
    0x8422100208500202ULL,
    0x2000402200300c08ULL,
    0x8646020080080080ULL,
    0x80020a0200100808ULL,
    0x2010004880111000ULL,
    0x623000a080011400ULL,
    0x42008c0340209202ULL,
    0x209188240001000ULL,
    0x400408a884001800ULL,
    0x110400a6080400ULL,
    0x1840060a44020800ULL,
    0x90080104000041ULL,
    0x201011000808101ULL,
    0x1a2208080504f080ULL,
    0x8012020600211212ULL,
    0x500861011240000ULL,
    0x180806108200800ULL,
    0x4000020e01040044ULL,
    0x300000261044000aULL,
    0x802241102020002ULL,
    0x20906061210001ULL,
    0x5a84841004010310ULL,
    0x4010801011c04ULL,
    0xa010109502200ULL,
    0x4a02012000ULL,
    0x500201010098b028ULL,
    0x8040002811040900ULL,
    0x28000010020204ULL,
    0x6000020202d0240ULL,
    0x8918844842082200ULL,
    0x4010011029020020ULL,
};

//generate pawn attacks
U64 mask_pawn_attacks(int side,int square){
    //result attcack bitboard
    U64 attacks = 0ULL;

    //piece bitboard
    U64 bitboard = 0ULL;

    //set piece on board
    set_bit(bitboard,square); 


    //white pawns
    if(!side){
        //generate pawn attacks
        if((bitboard >> 7) & not_a_file) attacks |= (bitboard >> 7);
        if((bitboard >> 9) & not_h_file) attacks |= (bitboard >> 9);
    }
    //black pawns
    else{
        //generate pawn attacks
        if((bitboard << 7) & not_h_file) attacks |= (bitboard << 7);
        if((bitboard << 9) & not_a_file) attacks |= (bitboard << 9);
    }

    return attacks;
    
}

//generate knight attacks
U64 mask_knight_attacks(int square){
    //result attack bitboard
    U64 attacks = 0ULL;

    //piece bitboard
    U64 bitboard = 0ULL;

    //set piece on board
    set_bit(bitboard,square);

    //generate knight attacks
    if((bitboard >> 17) & not_h_file) attacks |= (bitboard >> 17);
    if((bitboard >> 15) & not_a_file) attacks |= (bitboard >> 15);
    if((bitboard >> 10) & not_hg_file) attacks |= (bitboard >> 10);
    if((bitboard >> 6)  & not_ab_file) attacks |= (bitboard >> 6);

    if((bitboard << 17) & not_a_file) attacks |= (bitboard << 17);
    if((bitboard << 15) & not_h_file) attacks |= (bitboard << 15);
    if((bitboard << 10) & not_ab_file) attacks |= (bitboard << 10);
    if((bitboard << 6)  & not_hg_file) attacks |= (bitboard << 6);
    return attacks;
}

//generate king attacks
U64 mask_king_attacks(int square){
    //result attack bitboard
    U64 attacks =0ULL;

    //piece bitboard
    U64 bitboard = 0ULL;

    //set piece on board
    set_bit(bitboard,square);

    //generate king attacks
    if(bitboard >> 8) attacks |= (bitboard >> 8);
    if((bitboard >> 9) & not_h_file) attacks |= (bitboard >> 9);
    if((bitboard >> 7) & not_a_file) attacks |= (bitboard >> 7);
    if((bitboard >> 1) & not_h_file) attacks |= (bitboard >> 1);

    if( bitboard << 8) attacks |= (bitboard << 8);
    if((bitboard << 9) & not_a_file) attacks |= (bitboard << 9);
    if((bitboard << 7) & not_h_file) attacks |= (bitboard << 7);
    if((bitboard << 1) & not_a_file) attacks |= (bitboard << 1);
    return attacks;
}

//mask bishop attacks 
U64 mask_bishop_attacks(int square){
    //result attack bitboard
    U64 attacks = 0ULL;

    //init ranks and files 
    int r,f;

    //init target ranks and files
    int tr = square / 8;
    int tf = square % 8;

    //mask relevant bishop occupancy bits
    for(r = tr + 1 , f = tf + 1 ; r <= 6 && f <= 6 ; r++ , f++) set_bit(attacks,(r * 8 + f));
    for(r = tr - 1 , f = tf + 1 ; r >= 1 && f <= 6 ; r-- , f++) set_bit(attacks,(r * 8 + f));
    for(r = tr + 1 , f = tf - 1 ; r <= 6 && f >= 1 ; r++ , f--) set_bit(attacks,(r * 8 + f));
    for(r = tr - 1 , f = tf - 1 ; r >= 1 && f >= 1 ; r-- , f--) set_bit(attacks,(r * 8 + f));
    
    return attacks;
}

//mask rook attacks
U64 mask_rook_attacks(int square){
    //result attack bitboard
    U64 attacks = 0ULL;

    //init ranks and files 
    int r,f;

    //init target ranks and files
    int tr = square / 8;
    int tf = square % 8;

    //mask relevant rook occupancy bits
    for(r = tr + 1 ; r <= 6 ; r++) set_bit(attacks,(r * 8 + tf));
    for(r = tr - 1 ; r >= 1 ; r--) set_bit(attacks,(r * 8 + tf));
    for(f = tf + 1 ; f <= 6 ; f++) set_bit(attacks,(tr * 8 + f));
    for(f = tf - 1 ; f >= 1 ; f--) set_bit(attacks,(tr * 8 + f));

    return attacks;
}

//mask queen attacks
U64 mask_queen_attacks(int square){
    return mask_bishop_attacks(square) | mask_rook_attacks(square);
}

//generate  bishop attacks on the fly
U64 bishop_attacks_on_the_fly(int square, U64 block){
    //result attack bitboard
    U64 attacks = 0ULL;

    //init ranks and files 
    int r,f;

    //init target ranks and files
    int tr = square / 8;
    int tf = square % 8;

    //Generate bishop attacks
    for(r = tr + 1 , f = tf + 1 ; r <= 7 && f <= 7 ; r++ , f++) {
        set_bit(attacks,(r * 8 + f));
        if(1ULL << (r * 8 + f) & block) break;
    }
    for(r = tr - 1 , f = tf + 1 ; r >= 0 && f <= 7 ; r-- , f++) {
        set_bit(attacks,(r * 8 + f));
        if(1ULL << (r * 8 + f) & block) break;
    }
    for(r = tr + 1 , f = tf - 1 ; r <= 7 && f >= 0 ; r++ , f--) {
        set_bit(attacks,(r * 8 + f));
        if(1ULL << (r * 8 + f) & block) break;
    }
    for(r = tr - 1 , f = tf - 1 ; r >= 0 && f >= 0 ; r-- , f--) {
        set_bit(attacks,(r * 8 + f));
        if(1ULL << (r * 8 + f) & block) break;
    }
    
    return attacks;
}

//rook attacks on the fly
U64 rook_attacks_on_the_fly(int square, U64 block){
    //result attack bitboard
    U64 attacks = 0ULL;

    //init ranks and files 
    int r,f;

    //init target ranks and files
    int tr = square / 8;
    int tf = square % 8;

    //mask relevant rook occupancy bits
    for(r = tr + 1 ; r <= 7 ; r++) {
        set_bit(attacks,(r * 8 + tf));
        if(1ULL << (r * 8 + tf) & block) break;
    }
    for(r = tr - 1 ; r >= 0 ; r--) {
        set_bit(attacks,(r * 8 + tf));
        if(1ULL << (r * 8 + tf) & block) break;
    }
    for(f = tf + 1 ; f <= 7 ; f++) {
        set_bit(attacks,(tr * 8 + f));
        if(1ULL << (tr * 8 + f) & block) break;
    }
    for(f = tf - 1 ; f >= 0 ; f--) {
        set_bit(attacks,(tr * 8 + f));
        if(1ULL << (tr * 8 + f) & block) break;
    }
    
    return attacks;
}

// init leaper pieces attacks
void init_leapers_attack(){
    //loop over 64 board squares
    for(int square = 0 ; square < 64 ; square++){
        //init pawn attacks
        pawn_attacks[WHITE][square] = mask_pawn_attacks(WHITE,square);
        pawn_attacks[BLACK][square] = mask_pawn_attacks(BLACK,square);

        //init knight attacks
        knight_attacks[square] = mask_knight_attacks(square);

        //init king attacks
        king_attacks[square] = mask_king_attacks(square);
    }
}

//set occupancy function 
U64 set_occupancy(int index, int bits_in_mask, U64 attack_mask){
    //init occupancy bitboard
    U64 occupancy = 0ULL;

    //loop over the attack mask bits

    for(int count = 0 ; count < bits_in_mask ; count++){
        //get LS1B index of attack mask
        int square = get_ls1b_index(attack_mask); 
        //pop LS1B from attack mask
        pop_bit(attack_mask,square);
        //make sure occupancy is on the board
        if(index & (1 << count)){
            //set occupancy bit
            set_bit(occupancy,square);
        } 
    }

    //return occupancy bitboard
    return occupancy;
}

//init slider piece attacks
void init_sliders_attack(int bishop){
    //loop over 64 board squares
    for(int square = 0 ; square < 64 ; square++){
        //init bishop and rook masks
        bishop_mask[square] = mask_bishop_attacks(square);
        rook_mask[square] = mask_rook_attacks(square);

        //init current mask
        U64 attack_mask = bishop ? bishop_mask[square] : rook_mask[square];

        //init relevant occupancy bit count
        int relevant_bits_count = count_bits(attack_mask);

        //init occupancy indicies
        int occupancy_indicies = 1 << relevant_bits_count;

        //loop over occupancy indicies
        for(int i = 0 ; i < occupancy_indicies ; i++){
            //bishop
            if(bishop){
                //init current occupancy
                U64 occupancy = set_occupancy(i,relevant_bits_count,attack_mask);

                //init magic index
                int magic_index = (occupancy * bishop_magic_numbers[square]) >> (64 - bishop_relevant_bits[square]);

                //init bishop attacks
                bishop_attacks[square][magic_index] = bishop_attacks_on_the_fly(square,occupancy);
            }
            //rook
            else{
                //init current occupancy
                U64 occupancy = set_occupancy(i,relevant_bits_count,attack_mask);

                //init magic index
                int magic_index = (occupancy * rook_magic_numbers[square]) >> (64 - rook_relevant_bits[square]);

                //init bishop attacks
                rook_attacks[square][magic_index] = rook_attacks_on_the_fly(square,occupancy);
            }
        }
    }
}

//get bishop attacks
static inline U64 get_bishop_attacks(int square, U64 occupancy){
    //get bishop attacks assuming current board occupancy
    occupancy &=bishop_mask[square];
    occupancy *= bishop_magic_numbers[square];
    occupancy >>= 64 - bishop_relevant_bits[square];

    return bishop_attacks[square][occupancy];
}

//get rook attacks
static inline U64 get_rook_attacks(int square, U64 occupancy){
    //get bishop attacks assuming current board occupancy
    occupancy &=rook_mask[square];
    occupancy *= rook_magic_numbers[square];
    occupancy >>= 64 - rook_relevant_bits[square];

    return rook_attacks[square][occupancy];
}

//get queen attacks
static inline U64 get_queen_attacks(int square, U64 occupancy){
    return get_bishop_attacks(square,occupancy) | get_rook_attacks(square,occupancy);
}   


//*****************************************************
//====================================================
//                        Magics
//====================================================
//*****************************************************


//find appropriate magic number
U64 find_magic_number(int square , int relevant_bits, int bishop){
    //init occupancies
    U64 occupancies[4096];

    //init attack tables
    U64 attacks[4096];

    //init used attacks
    U64 used_attacks[4096];

    //init attack mask for current piece;
    U64 attack_mask = bishop ? mask_bishop_attacks(square) : mask_rook_attacks(square);

    //init occupancy indicies
    int occupancy_indicies = 1 << relevant_bits;

    //loop over occupancy indicies

    for(int index = 0 ; index < occupancy_indicies ; index++){
        //init occupancies
        occupancies[index] = set_occupancy(index, relevant_bits,attack_mask);

        //init attacks
        attacks[index] = bishop ? bishop_attacks_on_the_fly(square,occupancies[index]) : rook_attacks_on_the_fly(square, occupancies[index]);
    }

    //test magic number loops
    for(int random_count = 0 ; random_count < 100000000 ; random_count++){
        //generate magin number candidate
        U64 magic_number = generate_magic_number();
        //skip inappropriate magic numbers
        if(count_bits((attack_mask * magic_number) & 0xFF00000000000000) < 6) continue;
        //init used attacks
        memset(used_attacks,0ULL,sizeof(used_attacks));

        //init index and failed flag
        int index , fail;
        //test magic index loop
        for(index = 0, fail = 0 ; !fail && index < occupancy_indicies ; index++){
            int magic_index = (int)((occupancies[index] * magic_number ) >> (64 - relevant_bits));

            //if magic index works
            if(used_attacks[magic_index] == 0ULL)
                //init used attacks
                used_attacks[magic_index] = attacks[index];
            else if(used_attacks[magic_index] != attacks[index])
                //magic index doesnt works
                fail = 1;
        }

        if(!fail)
            //return magic number 
            return magic_number;
    }

    //magic number doesnt work 
    printf("    magic number fails!");
    return 0ULL;
}

//init magic numbers
void init_magic_number(){
    //loop for 64 board square
    printf("rook magic numbers\n");
    for(int square = 0 ; square < 64 ; square++){
        //init rook magic numbers
        rook_magic_numbers[square] = find_magic_number(square,rook_relevant_bits[square], rook);
    }
    printf("\n\n");
    printf("bishop magic numbers\n");
    for(int square = 0 ; square < 64 ; square++){
        //init rook magic numbers
        bishop_magic_numbers[square] = find_magic_number(square,bishop_relevant_bits[square], bishop);
    }
}


//*****************************************************
//====================================================
//                    INIT ALL
//====================================================
//*****************************************************


//init all variables
void init_all(){
    //init leapers attacks
    init_leapers_attack();

    //init sliders attacks
    init_sliders_attack(bishop);
    init_sliders_attack(rook);
     
}

//is the current given square attacjed by the current given side
static inline int is_square_attacked(int square, int side){
    //if attacked by white pawns
    if((side == WHITE) && (pawn_attacks[BLACK][square] & bitboards[P])) return 1; 
    //if attacked by black pawns
    if((side == BLACK) && (pawn_attacks[WHITE][square] & bitboards[p])) return 1;
    //if attacked by knights
    if((knight_attacks[square] & (side == WHITE ? bitboards[N] : bitboards[n]))) return 1;
    //if attacked by kings
    if((king_attacks[square] & (side == WHITE ? bitboards[K] : bitboards[k]))) return 1;
    //if attacked by bishops
    if(get_bishop_attacks(square,occupancies[BOTH]) & ((side == WHITE ? bitboards[B] : bitboards[b]))) return 1;
    
    //if attacked by rooks
    if(get_rook_attacks(square,occupancies[BOTH]) & ((side == WHITE ? bitboards[R] : bitboards[r]))) return 1;

    //if attacked by queens
    if(get_queen_attacks(square,occupancies[BOTH]) & ((side == WHITE ? bitboards[Q] : bitboards[q]))) return 1;


    //by default not attacked
    return 0;
}

//print attacked squares
void print_attacked_squares(int side){
    //loop over board ranks
    printf("\n");
    for(int rank = 0 ; rank < 8 ; rank++){
        //loop over board files
        for(int file = 0 ; file < 8 ; file++){
            //get square
            int square = rank * 8 + file;
            //print ranks
            if(!file){
                printf(" %d ",8 - rank);
            }
            //print square
            printf(" %d",is_square_attacked(square,side) ? 1 : 0);
        }
        //print rank
        printf("\n");
    }
    //print files
    printf("\n    a b c d e f g h\n\n");
}

//*****************************************************
//====================================================
//                    Main Driver Function
//====================================================
//*****************************************************


int main(){
    //init all

    init_all();
    parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 ");
    print_board();
    print_attacked_squares(WHITE);
    print_attacked_squares(BLACK);
    print_bitboard(occupancies[BOTH]);

    return 0;
}