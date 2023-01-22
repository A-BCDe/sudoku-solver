#include "ortools/linear_solver/linear_solver.h"

#include <cassert>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

class Sheet {
public:
	bool solve();
	virtual bool isValid() const = 0;
	virtual bool hasUniqueSolution() const = 0;

	friend std::ostream &operator<<(std::ostream &os, Sheet const &sheet) {
		sheet.print(os);
		return os;
	}

private:
	virtual bool solveSheet() = 0;
	virtual void print(std::ostream &os) const = 0;
};

bool Sheet::solve() {
	if(!isValid()) {
		LOG(WARNING) << "The sheet is not valid.";
		return false;
	}
	return solveSheet();
}

class SudokuSheet : public Sheet {
public:
	SudokuSheet(uint32_t, std::vector<uint32_t>);

	bool isValid() const override;
	bool hasUniqueSolution() const override;

private:
	uint32_t size;
	std::vector<uint32_t> puzzle;
	
	bool solveSheet() override;
	void print(std::ostream &os) const override;
};

SudokuSheet::SudokuSheet(uint32_t size, std::vector<uint32_t> puzzle)
	: size(size), puzzle(std::move(puzzle)), Sheet() {
	assert(puzzle.size() == size * size * size * size);
}

bool SudokuSheet::solveSheet() {
	using namespace operations_research;

	std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
	if(!solver) {
		LOG(WARNING) << "SCIP solver unavailable.";
		return false;
	}
	
	double const infinity = solver->infinity();

	uint32_t const size2 = size * size;
	uint32_t const size4 = size2 * size2;
	uint32_t const size6 = size4 * size2;
	std::vector<MPVariable*> variables;
	variables.reserve(size6);
	for(uint32_t i = 0; i < size2; i++) {
		for(uint32_t j = 0; j < size2; j++) {
			for(uint32_t k = 0; k < size2; k++) {
				variables.push_back(solver->MakeIntVar(0, 1, ""));
			}
		}
	}

	LOG(INFO) << "Number of variables = " << solver->NumVariables();

	std::vector<MPConstraint*> constraints;
	for(uint32_t j = 0; j < size2; j++) {
		for(uint32_t k = 0; k < size2; k++) {
			MPConstraint *const c = solver->MakeRowConstraint(1, 1);
			for(uint32_t i = 0; i < size2; i++) {
				c->SetCoefficient(variables[size4 * i + size2 * j + k], 1);
			}
			constraints.push_back(c);
		}
	}
	for(uint32_t i = 0; i < size2; i++) {
		for(uint32_t k = 0; k < size2; k++) {
			MPConstraint *const c = solver->MakeRowConstraint(1, 1);
			for(uint32_t j = 0; j < size2; j++) {
				c->SetCoefficient(variables[size4 * i + size2 * j + k], 1);
			}
			constraints.push_back(c);
		}
	}
	for(uint32_t i = 0; i < size2; i++) {
		for(uint32_t j = 0; j < size2; j++) {
			MPConstraint *const c = solver->MakeRowConstraint(1, 1);
			for(uint32_t k = 0; k < size2; k++) {
				c->SetCoefficient(variables[size4 * i + size2 * j + k], 1);
			}
			constraints.push_back(c);
		}
	}
	for(uint32_t p = 0; p < size; p++) {
		for(uint32_t q = 0; q < size; q++) {
			for(uint32_t k = 0; k < size2; k++) {
				MPConstraint *const c = solver->MakeRowConstraint(1, 1);
				for(uint32_t i = p * size; i < p * size + size; i++) {
					for(uint32_t j = q * size; j < q * size + size; j++) {
						c->SetCoefficient(variables[size4 * i + size2 * j + k], 1);
					}
				}
				constraints.push_back(c);
			}
		}
	}

	for(uint32_t i = 0; i < size2; i++) {
		for(uint32_t j = 0; j < size2; j++) {
			if(puzzle[size2 * i + j] != 0) {
				MPConstraint *const c = solver->MakeRowConstraint(1, 1);
				c->SetCoefficient(variables[size4 * i + size2 * j + puzzle[size2 * i + j] - 1], 1);
				constraints.push_back(c);
			}
		}
	}

	LOG(INFO) << "Number of constraints = " << solver->NumConstraints();
	
	MPObjective *const objective = solver->MutableObjective();
	for(auto const variable : variables) {
		objective->SetCoefficient(variable, 1);
	}
	objective->SetMaximization();

	auto const result_status = solver->Solve();
	if(result_status != MPSolver::OPTIMAL) {
		LOG(WARNING) << "The problem does not have an optimal solution!";
		return false;
	}

	LOG(INFO) << "Solution:";
	LOG(INFO) << "Objective value = " << objective->Value();

	LOG(INFO) << "\nAdvanced usage:";
	LOG(INFO) << "Problem solved in " << solver->wall_time() << " milliseconds";
	LOG(INFO) << "Problem solved in " << solver->iterations() << " iterations";
	LOG(INFO) << "Problem solved in " << solver->nodes() << " branch-and-bound nodes";

	for(uint32_t i = 0; i < size2; i++) {
		for(uint32_t j = 0; j < size2; j++) {
			for(uint32_t k = 0; k < size2; k++) {
				if(variables[size4 * i + size2 * j + k]->solution_value() == 1) {
					assert(puzzle[size2 * i + j] == 0 || puzzle[size2 * i + j] == k + 1);
					puzzle[size2 * i + j] = k + 1;
				}
			}
		}
	}
	return true;
}

bool SudokuSheet::isValid() const {
	return true;
}

bool SudokuSheet::hasUniqueSolution() const {
	return true;
}

void SudokuSheet::print(std::ostream &os) const {
	uint32_t const len = std::to_string(size * size).size() + 1;
	for(uint32_t i = 0; i < size * size; i++) {
		for(uint32_t j = 0; j < size * size; j++) {
			auto const str = std::to_string(puzzle[size * size * i + j]);

			os << std::string(len - str.size(), ' ') << str;
		} os << '\n';
	}
}

class KakuroSheet : public Sheet {
public:
	enum class direction { right, down };
	struct position { uint32_t x; uint32_t y; };
	struct hint {
		position pos; // position of the triangle
		uint32_t len;
		uint32_t sum;
		direction dir;
	};

	KakuroSheet(uint32_t, uint32_t, std::vector<hint>);
	KakuroSheet(uint32_t, uint32_t, uint32_t, std::vector<hint>);
	bool isValid() const override;
	bool hasUniqueSolution() const override;

private:
	uint32_t row;
	uint32_t col;
	uint32_t limit;
	std::vector<uint32_t> puzzle;
	std::vector<hint> hints;

	bool solveSheet() override;
	void print(std::ostream &os) const override;
};

KakuroSheet::KakuroSheet(uint32_t row, uint32_t col, std::vector<hint> hints)
	: row(row), col(col), limit(10), puzzle(row * col), hints(std::move(hints)), Sheet() {
	assert(puzzle.size() == row * col);
}

KakuroSheet::KakuroSheet(uint32_t row, uint32_t col, uint32_t limit, std::vector<hint> hints)
	: row(row), col(col), limit(limit), puzzle(row * col), hints(std::move(hints)), Sheet() {
	assert(puzzle.size() == row * col);
}

bool KakuroSheet::solveSheet() {
	using namespace operations_research;

	std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
	if(!solver) {
		LOG(WARNING) << "SCIP solver unavailable.";
		return false;
	}
	
	double const infinity = solver->infinity();
	std::vector<MPVariable*> variables(row * col * limit);
	for(auto const &h : hints) {
		switch(h.dir) {
			case direction::right:
				for(uint32_t j = h.pos.y + 1; j <= h.pos.y + h.len; j++) {
					for(uint32_t k = 1; k < limit; k++) {
						variables[row * limit * h.pos.x + limit * j + k] = solver->MakeIntVar(0, 1, "");
					}
				}
				break;
			case direction::down: break;
			default: __builtin_unreachable(); // std::unreachable();
		}
	}

	std::vector<MPConstraint*> constraints;

	for(auto const &h : hints) {
		switch(h.dir) {
			case direction::right:
				for(uint32_t j = h.pos.y + 1; j <= h.pos.y + h.len; j++) {
					MPConstraint *const c = solver->MakeRowConstraint(1, 1);
					for(uint32_t k = 1; k < limit; k++) {
						c->SetCoefficient(variables[row * limit * h.pos.x + limit * j + k], 1);
					}
					constraints.push_back(c);
				}
				break;
			case direction::down: break;
			default: __builtin_unreachable(); // std::unreachable();
		}
	}

	for(auto const &h : hints) {
		switch(h.dir) {
			case direction::right:
				for(uint32_t k = 1; k < limit; k++) {
					MPConstraint *const c = solver->MakeRowConstraint(1, 1);
					for(uint32_t j = h.pos.y + 1; j <= h.pos.y + h.len; j++) {
						c->SetCoefficient(variables[row * limit * h.pos.x + limit * j + k], 1);
					}
					constraints.push_back(c);
				}
				break;
			case direction::down:
				for(uint32_t k = 1; k < limit; k++) {
					MPConstraint *const c = solver->MakeRowConstraint(1, 1);
					for(uint32_t i = h.pos.x + 1; i <= h.pos.x + h.len; i++) {
						c->SetCoefficient(variables[row * limit * i + limit * h.pos.y + k], 1);
					}
					constraints.push_back(c);
				}
				break;
			default: __builtin_unreachable(); // std::unreachable();
		}
	}

	for(auto const &h : hints) {
		MPConstraint *const c = solver->MakeRowConstraint(h.sum, h.sum);
		switch(h.dir) {
			case direction::right:
				for(uint32_t j = h.pos.y + 1; j <= h.pos.y + h.len; j++) {
					for(uint32_t k = 1; k < limit; k++) {
						c->SetCoefficient(variables[row * limit * h.pos.x + limit * j + k], k);
					}
				}
				break;
			case direction::down:
				for(uint32_t i = h.pos.x + 1; i <= h.pos.x + h.len; i++) {
					for(uint32_t k = 1; k < limit; k++) {
						c->SetCoefficient(variables[row * limit * i + limit * h.pos.y + k], k);
					}
				}
				break;
			default: __builtin_unreachable(); // std::unreachable();
		}
		constraints.push_back(c);
	}

	LOG(INFO) << "Number of constraints = " << solver->NumConstraints();

	MPObjective *const objective = solver->MutableObjective();
	for(auto const &h : hints) {
		switch(h.dir) {
			case direction::right:
				for(uint32_t j = h.pos.y + 1; j <= h.pos.y + h.len; j++) {
					for(uint32_t k = 1; k < limit; k++) {
						objective->SetCoefficient(variables[row * limit * h.pos.x + limit * j + k], 1);
					}
				}
				break;
			case direction::down: break;
			default: __builtin_unreachable(); // std::unreachable();
		}
	}
	objective->SetMaximization();

	auto const result_status = solver->Solve();
	if(result_status != MPSolver::OPTIMAL) {
		LOG(WARNING) << "The problem does not have an optimal solution!";
		return false;
	}
	
	LOG(INFO) << "Solution:";
	LOG(INFO) << "Objective value = " << objective->Value();

	LOG(INFO) << "\nAdvanced usage:";
	LOG(INFO) << "Problem solved in " << solver->wall_time() << " milliseconds";
	LOG(INFO) << "Problem solved in " << solver->iterations() << " iterations";
	LOG(INFO) << "Problem solved in " << solver->nodes() << " branch-and-bound nodes";

	for(auto const &h : hints) {
		switch(h.dir) {
			case direction::right:
				for(uint32_t j = h.pos.y + 1; j <= h.pos.y + h.len; j++) {
					for(uint32_t k = 1; k < limit; k++) {
						if(variables[row * limit * h.pos.x + limit * j + k]->solution_value() == 1) {
							assert(puzzle[row * h.pos.x + j] == 0);
							puzzle[row * h.pos.x + j] = k;
						}
					}
				}
				break;
			case direction::down: break;
			default: __builtin_unreachable(); // std::unreachable();
		}
	}

	return true;
}

bool KakuroSheet::isValid() const {
	return true;
}

bool KakuroSheet::hasUniqueSolution() const {
	return true;
}

void KakuroSheet::print(std::ostream &os) const {
	for(uint32_t i = 0; i < row; i++) {
		for(uint32_t j = 0; j < col; j++) {
			os << ' ' << puzzle[row * i + j];
		} os << '\n';
	}
}

namespace operations_research {
void SimpleMipProgram() {
  // Create the mip solver with the SCIP backend.
  std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("SCIP"));
  if (!solver) {
    LOG(WARNING) << "SCIP solver unavailable.";
    return;
  }

  const double infinity = solver->infinity();
  // x and y are integer non-negative variables.
  MPVariable* const x = solver->MakeIntVar(0.0, infinity, "x");
  MPVariable* const y = solver->MakeIntVar(0.0, infinity, "y");

  LOG(INFO) << "Number of variables = " << solver->NumVariables();

  // x + 7 * y <= 17.5.
  MPConstraint* const c0 = solver->MakeRowConstraint(-infinity, 17.5, "c0");
  c0->SetCoefficient(x, 1);
  c0->SetCoefficient(y, 7);

  // x <= 3.5.
  MPConstraint* const c1 = solver->MakeRowConstraint(-infinity, 3.5, "c1");
  c1->SetCoefficient(x, 1);
  c1->SetCoefficient(y, 0);

  LOG(INFO) << "Number of constraints = " << solver->NumConstraints();

  // Maximize x + 10 * y.
  MPObjective* const objective = solver->MutableObjective();
  objective->SetCoefficient(x, 1);
  objective->SetCoefficient(y, 10);
  objective->SetMaximization();

  const MPSolver::ResultStatus result_status = solver->Solve();
  // Check that the problem has an optimal solution.
  if (result_status != MPSolver::OPTIMAL) {
    LOG(FATAL) << "The problem does not have an optimal solution!";
  }

  LOG(INFO) << "Solution:";
  LOG(INFO) << "Objective value = " << objective->Value();
  LOG(INFO) << "x = " << x->solution_value();
  LOG(INFO) << "y = " << y->solution_value();

  LOG(INFO) << "\nAdvanced usage:";
  LOG(INFO) << "Problem solved in " << solver->wall_time() << " milliseconds";
  LOG(INFO) << "Problem solved in " << solver->iterations() << " iterations";
  LOG(INFO) << "Problem solved in " << solver->nodes()
            << " branch-and-bound nodes";
}
}  // namespace operations_research

int main(int argc, char *argv[]) {
	assert(argc == 1);
	std::vector<uint32_t> test_sudoku_puzzle {
		5, 3, 0, 0, 7, 0, 0, 0, 0,
		6, 0, 0, 1, 9, 5, 0, 0, 0,
		0, 9, 8, 0, 0, 0, 0, 6, 0,
		8, 0, 0, 0, 6, 0, 0, 0, 3,
		4, 0, 0, 8, 0, 3, 0, 0, 1,
		7, 0, 0, 0, 2, 0, 0, 0, 6,
		0, 6, 0, 0, 0, 0, 2, 8, 0,
		0, 0, 0, 4, 1, 9, 0, 0, 5,
		0, 0, 0, 0, 8, 0, 0, 7, 9
	};
	SudokuSheet sudokuSheet(3, std::move(test_sudoku_puzzle));

	std::cout << sudokuSheet << "\n\n";

	sudokuSheet.solve();

	std::cout << sudokuSheet << "\n\n";

	std::vector<KakuroSheet::hint> hints {
		{ { 1, 0 }, 2, 4, KakuroSheet::direction::right },
		{ { 2, 0 }, 2, 16, KakuroSheet::direction::right },
		{ { 0, 1 }, 2, 8, KakuroSheet::direction::down },
		{ { 0, 2 }, 2, 12, KakuroSheet::direction::down }
	};

	KakuroSheet kakuroSheet(3, 3, std::move(hints));

	std::cout << kakuroSheet << "\n\n";

	kakuroSheet.solve();

	std::cout << kakuroSheet << "\n\n";

}
