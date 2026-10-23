#include <iostream>

#include "lqr_controller_math.hpp"

int main() {
  using namespace swift_pico::lqr;
  StateMatrix a;
  InputMatrix b;
  make_discrete_model(0.02, 1.5, 9.81, &a, &b);
  CostMatrix q = CostMatrix::Zero();
  q.diagonal() << 10.0, 10.0, 20.0, 8.0, 8.0, 6.0;
  InputCostMatrix r = InputCostMatrix::Zero();
  r.diagonal() << 10.0, 10.0, 0.8;
  const GainMatrix k = solve_discrete_lqr(a, b, q, r);
  if (!k.allFinite() || !is_stable(a, b, k)) {
    std::cerr << "LQR gain is non-finite or does not stabilize the discrete model\n";
    return 1;
  }
  State error = State::Zero();
  error(0) = 1.0;
  if ((-k * error)(0) >= 0.0) { std::cerr << "x sign failure\n"; return 1; }
  error.setZero(); error(1) = 1.0;
  if ((-k * error)(1) >= 0.0) { std::cerr << "y sign failure\n"; return 1; }
  error.setZero(); error(2) = 1.0;
  if ((-k * error)(2) <= 0.0) { std::cerr << "z sign failure\n"; return 1; }
  return 0;
}
