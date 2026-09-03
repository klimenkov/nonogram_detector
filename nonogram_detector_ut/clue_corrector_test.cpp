#include <iostream>
#include <vector>

#include "clue_corrector.hpp"


namespace
{

ng::ClueCell cell(int digit, double conf = 0.99)
{
    return ng::ClueCell{digit, conf};
}

bool eq_line(std::vector<int> const& a, std::vector<int> const& b)
{
    return a == b;
}

std::vector<std::string> failures;

void check(bool cond, std::string const& name)
{
    if (!cond)
    {
        std::cerr << "  [FAIL] " << name << "\n";
        failures.push_back(name);
    }
}

// ---------- group_clue_line ----------
void test_group()
{
    std::cout << "case: group_clue_line\n";

    // The decoder emits one clue per cell (a single digit 1..9, or a two-digit
    // value 10..99 for a cell the counter model flagged as holding two digits),
    // so adjacent non-empty cells are adjacent *distinct* clues, never digits
    // of a single multi-digit number.
    check(eq_line(ng::group_clue_line({}), {}), "empty line -> empty");
    check(eq_line(ng::group_clue_line({cell(-1), cell(-1)}), {}), "all-empty -> empty");
    check(eq_line(ng::group_clue_line({cell(2), cell(-1), cell(4)}), {2, 4}),
          "gap separates single-digit clues");
    check(eq_line(ng::group_clue_line({cell(1), cell(4), cell(-1), cell(9)}), {1, 4, 9}),
          "adjacent non-empty cells are distinct clues (1, 4, 9)");
    check(eq_line(ng::group_clue_line({cell(2), cell(-1), cell(-1), cell(3)}), {2, 3}),
          "consecutive gaps still separate");
    check(eq_line(ng::group_clue_line({cell(0)}), {}), "lone zero dropped");
    check(eq_line(ng::group_clue_line({cell(1), cell(0)}), {1}),
          "bogus zero cell dropped");
    check(eq_line(ng::group_clue_line({cell(10)}), {10}),
          "two-digit cell is one clue value");
    check(eq_line(ng::group_clue_line({cell(2), cell(10)}), {2, 10}),
          "2-digit cell separated from its neighbour clue");
    check(eq_line(ng::group_clue_line({cell(1), cell(-1), cell(0), cell(-1), cell(2)}), {1, 2}),
          "zero as its own number dropped");
}

// ---------- clue_line_min_size ----------
void test_min_size()
{
    std::cout << "case: clue_line_min_size\n";

    check(ng::clue_line_min_size({}) == 0, "empty -> 0");
    check(ng::clue_line_min_size({3}) == 3, "{3} -> 3");
    check(ng::clue_line_min_size({1, 1, 1}) == 5, "{1,1,1} -> 5");
    check(ng::clue_line_min_size({14, 9}) == 24, "{14,9} -> 24");
}

// ---------- analyze_consistency ----------
void test_consistency()
{
    std::cout << "case: analyze_consistency\n";

    // 2x2 all-single: balanced
    ng::DecodedCells cells;
    cells.rows = {
        { cell(1), cell(-1), },
        { cell(-1), cell(1), },
    };
    cells.cols = {
        { cell(1) }, { cell(1) },
    };
    auto r = ng::analyze_consistency(cells, 2, 2);
    check(r.consistent, "balanced 2x2 reported consistent");
    check(r.row_tiles == 2 && r.col_tiles == 2, "tile totals equal (2)");
    check(r.row_overflow.empty() && r.col_overflow.empty(), "no overflow lines");

    // Row/col tile mismatch
    ng::DecodedCells bad;
    bad.rows = { { cell(1), cell(-1), cell(1) } };  // "1" and "1" -> 2 tiles
    bad.cols = { { cell(1) } };                      // 1 tile
    auto r2 = ng::analyze_consistency(bad, 2, 1);
    check(!r2.consistent, "tile-total mismatch reported inconsistent");
    check(r2.row_tiles == 2 && r2.col_tiles == 1, "tile totals recorded (2 vs 1)");

    // Overflow: a row clue that can't fit the width
    ng::DecodedCells over;
    over.rows = { { cell(5) } };  // min_size 5 > width 3
    over.cols = { { cell(1) }, { cell(1) }, { cell(1) } };
    auto r3 = ng::analyze_consistency(over, 3, 1);
    check(!r3.consistent, "overflow line reported inconsistent");
    check(r3.row_overflow == std::vector<int>{0}, "overflowing row identified");
}

// ---------- correct_clues ----------
void test_correct()
{
    std::cout << "case: correct_clues\n";

    // A bogus standalone '0' cell (border/stray read) is dropped -> empty.
    ng::DecodedCells c;
    c.rows = { { cell(2), cell(-1), cell(0), cell(-1) } };   // "2" then bogus 0
    c.cols = { { cell(2) }, { cell(-1) }, { cell(-1) }, { cell(-1) } };
    ng::ConsistencyReport rep;
    auto out = ng::correct_clues(c, 3, 1, rep);
    check(out.rows[0][2].digit == -1, "bogus 0-cell dropped to empty");
    check(rep.row_tiles == 2 && rep.col_tiles == 2, "tile totals balanced after 0-drop");

    // A zero inside a two-digit clue is preserved (a single cell reading "10").
    ng::DecodedCells ten;
    ten.rows = { { cell(10) } };
    ten.cols = { { cell(10) } };
    ng::ConsistencyReport rep_ten;
    auto out_ten = ng::correct_clues(ten, 10, 1, rep_ten);
    check(eq_line(ng::group_clue_line(out_ten.rows[0]), std::vector<int>{10}),
          "two-digit clue 10 preserved as one clue");

    // Two adjacent single-digit cells are two distinct clues that fit a
    // width-10 grid (sum 9, min size 10); no overflow, no re-grouping.
    ng::DecodedCells adj;
    adj.rows = { { cell(2), cell(7) } };
    adj.cols = { { cell(2) }, { cell(7) } };
    ng::ConsistencyReport rep_adj;
    auto out_adj = ng::correct_clues(adj, 10, 1, rep_adj);
    check(eq_line(ng::group_clue_line(out_adj.rows[0]), std::vector<int>{2, 7}),
          "adjacent single-digit cells stay two distinct clues (2, 7)");
    check(rep_adj.row_overflow.empty(), "adjacent clues fit width 10");

    // Realistic photo row: 7 adjacent cells read "4 2 1 1 1 1 2" on a width-30
    // grid. Each cell is its own clue and fits (min size 18), so the line is
    // consistent and preserved verbatim once the bogus reads are absent.
    ng::DecodedCells row14;
    row14.rows = { { cell(4), cell(2), cell(1), cell(1), cell(1), cell(1), cell(2) } };
    row14.cols = {};
    ng::ConsistencyReport rep14;
    auto out14 = ng::correct_clues(row14, 30, 7, rep14);
    check(rep14.row_overflow.empty(),
          "row-14-style adjacent clues reported as fitting width 30");
    check(eq_line(ng::group_clue_line(out14.rows[0]),
                  std::vector<int>{4, 2, 1, 1, 1, 1, 2}),
          "row-14-style adjacent clues preserved as seven clues");
}

}

int run_clue_corrector_tests()
{
    failures.clear();
    test_group();
    test_min_size();
    test_consistency();
    test_correct();

    for (auto const& f : failures)
        std::cerr << "clue_corrector: " << f << "\n";
    if (!failures.empty())
    {
        std::cerr << failures.size() << " clue_corrector test(s) FAILED\n";
        return static_cast<int>(failures.size());
    }
    std::cout << "all clue_corrector tests passed\n";
    return 0;
}
