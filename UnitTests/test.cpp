#include "pch.h"
#include "Game.h"

namespace {
bool withinTolerance(double d1, double d2, double tolerance) {
  return std::abs(d1 - d2) <= tolerance;
}
}; // namespace

// Test that the CFRM algorithm with two-player Kuhn Poker produces the correct
// results
TEST(Game, CFRM_with_two_players_produces_correct_results) {
  Game game(2);
  game.init();
  game.train(10000000);
  game.calcProperties(1000);
  const auto &betProbs = game.getAvgBetProbabilities();
  const auto &gameValues = game.getAvgGameValues();

  ASSERT_EQ(betProbs.size(), 12);

  double tolerance = 1e-2;
  double alpha = betProbs[0].back();                     // J
  EXPECT_NEAR(betProbs[1].back(), 1.0 / 3.0, tolerance); // Jp
  EXPECT_NEAR(betProbs[2].back(), 0.0, tolerance);       // Jb
  EXPECT_NEAR(betProbs[3].back(), 0.0, tolerance);       // Jpb

  EXPECT_NEAR(betProbs[4].back(), 0.0, tolerance);               // Q
  EXPECT_NEAR(betProbs[5].back(), 0.0, tolerance);               // Qp
  EXPECT_NEAR(betProbs[6].back(), 1.0 / 3.0, tolerance);         // Qb
  EXPECT_NEAR(betProbs[7].back(), alpha + 1.0 / 3.0, tolerance); // Qpb

  EXPECT_NEAR(betProbs[8].back(), 3.0 * alpha, tolerance); // K
  EXPECT_NEAR(betProbs[9].back(), 1.0, tolerance);         // Kp
  EXPECT_NEAR(betProbs[10].back(), 1.0, tolerance);        // Kb
  EXPECT_NEAR(betProbs[11].back(), 1.0, tolerance);        // Kbp

  EXPECT_NEAR(gameValues[0].back(), -1.0 / 18.0, tolerance); // player 1
  EXPECT_NEAR(gameValues[1].back(), 1.0 / 18.0, tolerance);  // player 2

  std::cout << "\nResults:\n--------\n\n" << game;
}

// Test that the CFRM algorithm with three-player Kuhn Poker produces the
// correct results
TEST(Game, CFRM_with_three_players_produces_correct_results) {
  Game game(3);
  game.init();
  game.train(10000000);
  game.calcProperties(1000);
  const auto &betProbs = game.getAvgBetProbabilities();
  const auto &gameValues = game.getAvgGameValues();

  ASSERT_EQ(betProbs.size(), 48);

  double tolerance = 3e-2;
  double jp = betProbs[1].back();
  double jpp = betProbs[3].back();
  double qp = betProbs[13].back();
  double qppbp = betProbs[22].back();
  double kb = betProbs[26].back();
  double alpha = std::max(jp, qp);
  double handProb = 1.0 / 24.0; // probability of each hand

  EXPECT_NEAR(betProbs[0].back(), 0.0, tolerance); // J
  if (withinTolerance(jpp, 0.0, tolerance))
    EXPECT_LE(jp, qp);
  else
    EXPECT_LE(jp, 0.25);                                             // Jp
  EXPECT_NEAR(betProbs[2].back(), 0.0, tolerance);                   // Jb
  EXPECT_LE(jpp, std::min(0.5, (2.0 - jp) / (3 + 2.0 * (jp + qp)))); // Jpp
  EXPECT_NEAR(betProbs[4].back(), 0.0, tolerance);                   // Jpb
  EXPECT_NEAR(betProbs[5].back(), 0.0, tolerance);                   // Jbp
  EXPECT_NEAR(betProbs[6].back(), 0.0, tolerance);                   // Jbb
  EXPECT_NEAR(betProbs[7].back(), 0.0, tolerance);                   // Jppb
  EXPECT_NEAR(betProbs[8].back(), 0.0, tolerance);                   // Jpbp
  EXPECT_NEAR(betProbs[9].back(), 0.0, tolerance);                   // Jpbb
  EXPECT_NEAR(betProbs[10].back(), 0.0, tolerance);                  // Jppbp
  EXPECT_NEAR(betProbs[11].back(), 0.0, tolerance);                  // Jppbb

  EXPECT_NEAR(betProbs[12].back(), 0.0, tolerance); // Q
  if (withinTolerance(jpp, 0.0, tolerance)) {
    EXPECT_LE(qp, 0.25);
  } else if (jpp < 0.5) {
    if (!withinTolerance(jpp, 0.5, tolerance))
      EXPECT_NEAR(qp, jp, tolerance);
    else
      EXPECT_LE(qp, std::min(jp, 0.5 - 2 * jp));
  }                                                              // Qp
  EXPECT_NEAR(betProbs[14].back(), 0.0, tolerance);              // Qb
  EXPECT_NEAR(betProbs[15].back(), 0.5 - jpp, tolerance);        // Qpp
  EXPECT_NEAR(betProbs[16].back(), 0.0, tolerance);              // Qpb
  EXPECT_NEAR(betProbs[17].back(), 0.0, tolerance);              // Qbp
  EXPECT_NEAR(betProbs[18].back(), 0.0, tolerance);              // Qbb
  EXPECT_NEAR(betProbs[19].back(), 0.0, tolerance);              // Qppb
  EXPECT_NEAR(betProbs[20].back(), 0.0, tolerance);              // Qpbp
  EXPECT_NEAR(betProbs[21].back(), 0.0, tolerance);              // Qpbb
  EXPECT_LE(qppbp, std::max(0.0, (jp - qp) / 2.0 * (1.0 - qp))); // Qppbp
  EXPECT_NEAR(betProbs[23].back(), 0.0, tolerance);              // Qppbb

  EXPECT_NEAR(betProbs[24].back(), 0.0, tolerance);      // K
  EXPECT_NEAR(betProbs[25].back(), 0.0, tolerance);      // Kp
  EXPECT_LE(kb, 0.25 * (2.0 + 3.0 * (jp + qp) + alpha)); // Kb
  EXPECT_NEAR(betProbs[27].back(), 0.0, tolerance);      // Kpp
  EXPECT_NEAR(betProbs[28].back(), 0.0, tolerance);      // Kpb
  EXPECT_GE(betProbs[29].back(), 0.5 - kb);
  EXPECT_LE(betProbs[29].back(),
            0.25 * (2.0 + 3.0 * (jp + qp) + alpha) - kb); // Kbp
  EXPECT_GE(betProbs[30].back(), 0.0);
  EXPECT_LE(betProbs[30].back(), 1.0);              // Kbb
  EXPECT_NEAR(betProbs[31].back(), 0.0, tolerance); // Kppb
  EXPECT_NEAR(betProbs[32].back(), 0.5, tolerance); // Kpbp
  EXPECT_NEAR(betProbs[33].back(), 0.0, tolerance); // Kpbb
  EXPECT_NEAR(betProbs[34].back(),
              0.5 * (1.0 + jp + qp + alpha) + qppbp * (qp - 1.0),
              tolerance);                           // Kppbp
  EXPECT_NEAR(betProbs[35].back(), 0.0, tolerance); // Kppbb

  EXPECT_NEAR(betProbs[36].back(), 0.0, tolerance);             // A
  EXPECT_NEAR(betProbs[37].back(), 2.0 * (jp + qp), tolerance); // Ap
  EXPECT_NEAR(betProbs[38].back(), 1.0, tolerance);             // Ab
  EXPECT_NEAR(betProbs[39].back(), 1.0, tolerance);             // App
  EXPECT_NEAR(betProbs[40].back(), 1.0, tolerance);             // Apb
  EXPECT_NEAR(betProbs[41].back(), 1.0, tolerance);             // Abp
  EXPECT_NEAR(betProbs[42].back(), 1.0, tolerance);             // Abb
  EXPECT_NEAR(betProbs[43].back(), 1.0, tolerance);             // Appb
  EXPECT_NEAR(betProbs[44].back(), 1.0, tolerance);             // Apbp
  EXPECT_NEAR(betProbs[45].back(), 1.0, tolerance);             // Apbb
  EXPECT_NEAR(betProbs[46].back(), 1.0, tolerance);             // Appbp
  EXPECT_NEAR(betProbs[47].back(), 1.0, tolerance);             // Appbb

  EXPECT_NEAR(gameValues[0].back(), -handProb * (0.5 + alpha),
              tolerance);                                        // player 1
  EXPECT_NEAR(gameValues[1].back(), -handProb * 0.5, tolerance); // player 2
  EXPECT_NEAR(gameValues[2].back(), handProb * (1.0 + alpha),
              tolerance); // player 3

  std::cout << "\nResults:\n--------\n\n" << game;
}