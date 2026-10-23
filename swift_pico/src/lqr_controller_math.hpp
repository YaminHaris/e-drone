#pragma once

#include <Eigen/Dense>

#include <stdexcept>

// The simulator's attitude controller is the inner loop. This controller is
// an outer-loop small-angle model with inputs [pitch, roll, dT]. State axes
// are WhyCode's camera-centric x/y/depth coordinates. The camera orientation
// makes WhyCode y opposite world y and depth opposite world-up.
namespace swift_pico::lqr {

constexpr int kStateSize = 6;
constexpr int kInputSize = 3;
using State = Eigen::Matrix<double, kStateSize, 1>;
using Input = Eigen::Matrix<double, kInputSize, 1>;
using StateMatrix = Eigen::Matrix<double, kStateSize, kStateSize>;
using InputMatrix = Eigen::Matrix<double, kStateSize, kInputSize>;
using GainMatrix = Eigen::Matrix<double, kInputSize, kStateSize>;
using CostMatrix = Eigen::Matrix<double, kStateSize, kStateSize>;
using InputCostMatrix = Eigen::Matrix<double, kInputSize, kInputSize>;

inline void make_discrete_model(const double dt, const double mass,
                                const double gravity, StateMatrix *a,
                                InputMatrix *b) {
  if (dt <= 0.0 || mass <= 0.0 || gravity <= 0.0) {
    throw std::invalid_argument("LQR model requires positive dt, mass and gravity");
  }
  *a = StateMatrix::Identity();
  a->block<3, 3>(0, 3) = dt * Eigen::Matrix3d::Identity();
  *b = InputMatrix::Zero();
  (*b)(0, 0) = 0.5 * gravity * dt * dt;  // pitch -> x acceleration
  (*b)(3, 0) = gravity * dt;
  (*b)(1, 1) = 0.5 * gravity * dt * dt;  // roll -> WhyCode y acceleration
  (*b)(4, 1) = gravity * dt;
  (*b)(2, 2) = -0.5 * dt * dt / mass;    // dT reduces camera depth
  (*b)(5, 2) = -dt / mass;
}

// Discrete algebraic Riccati fixed-point iteration. Eigen is the only
// dependency, so no unverified hard-coded gain or external control library.
inline GainMatrix solve_discrete_lqr(const StateMatrix &a, const InputMatrix &b,
                                     const CostMatrix &q, const InputCostMatrix &r,
                                     const int max_iterations = 20000,
                                     const double tolerance = 1.0e-9) {
  if (!a.allFinite() || !b.allFinite() || !q.allFinite() || !r.allFinite()) {
    throw std::invalid_argument("LQR matrices contain a non-finite value");
  }
  CostMatrix p = q;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    const InputCostMatrix s = r + b.transpose() * p * b;
    Eigen::LDLT<InputCostMatrix> solver(s);
    if (solver.info() != Eigen::Success || !solver.isPositive()) {
      throw std::runtime_error("LQR R + B'PB is not positive definite");
    }
    const CostMatrix next = q + a.transpose() * p * a -
                            a.transpose() * p * b * solver.solve(b.transpose() * p * a);
    if (!next.allFinite()) throw std::runtime_error("LQR Riccati iteration became non-finite");
    if ((next - p).cwiseAbs().maxCoeff() < tolerance) {
      const InputCostMatrix final_s = r + b.transpose() * next * b;
      Eigen::LDLT<InputCostMatrix> final_solver(final_s);
      if (final_solver.info() != Eigen::Success || !final_solver.isPositive()) {
        throw std::runtime_error("LQR final R + B'PB is not positive definite");
      }
      const GainMatrix gain = final_solver.solve(b.transpose() * next * a);
      if (!gain.allFinite()) throw std::runtime_error("LQR gain is non-finite");
      return gain;
    }
    p = next;
  }
  throw std::runtime_error("LQR Riccati iteration did not converge");
}

inline bool is_stable(const StateMatrix &a, const InputMatrix &b, const GainMatrix &gain) {
  const Eigen::EigenSolver<StateMatrix> solver(a - b * gain, false);
  return solver.info() == Eigen::Success && solver.eigenvalues().array().abs().maxCoeff() < 1.0;
}

}  // namespace swift_pico::lqr
