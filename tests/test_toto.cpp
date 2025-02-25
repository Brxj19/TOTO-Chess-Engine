#include <gtest/gtest.h>
extern "C" {
    #include "../src/toto.c"
}

TEST(TotoTest, MaskPawnAttacksWhite) {
    U64 expected = 0ULL;
    set_bit(expected, g3);
    set_bit(expected, e3);
    EXPECT_EQ(expected, mask_pawn_attacks(f2, WHITE));
}

TEST(TotoTest, MaskPawnAttacksBlack) {
    U64 expected = 0ULL;
    set_bit(expected, g6);
    set_bit(expected, e6);
    EXPECT_EQ(expected, mask_pawn_attacks(f7, BLACK));
}

TEST(TotoTest, PrintBitboard) {
    U64 bitboard = 0ULL;
    set_bit(bitboard, a1);
    set_bit(bitboard, h8);
    print_bitboard(bitboard);
    // Manually verify the output
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}