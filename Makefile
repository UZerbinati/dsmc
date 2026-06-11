# DSMC test suite — make-based runner.
#
# Replaces the prior bash scripts (run_needle.sh, run_disc.sh,
# tests/boltzmann/run_tests.sh) with a Makefile that splits the test
# suite into logically-grouped targets:
#
#   make boltzmann       - the homogeneous Boltzmann tests in
#                          tests/boltzmann/test_<N>.py.
#   make disc            - all discotic tests (CFMZDiscDSMCHomo + CFMZDiscDSMC).
#                          Boltzmann kernel; 3-D-disc-in-2-D nematic
#                          interpretation as the default.
#   make needle          - the non-smectic needle tests:
#                          test_needle_<N>.py for the homogeneous
#                          calamitic suite (Boltzmann), plus
#                          test_needle_inhomo_0.py for the existing
#                          1-D inhomogeneous Sod tube.
#   make needle-smectic  - the Enskog-kernel needle tests demonstrating
#                          the smectic-A phase: test_needle_smectic_2d,
#                          test_needle_smectic_2d_sweep, and the two
#                          Enskog-vs-Boltzmann Sod tubes
#                          (test_needle_sod_dense, test_needle_sod_orient).
#   make phase-diagram   - (re)generate the non-smectic isotropic→nematic
#                          phase diagram from the test_needle_27 sweep
#                          outputs.  Run after `make needle`.  The smectic
#                          phase diagram is written inline by
#                          test_needle_smectic_2d_sweep itself.
#   make all             - all of the above (phase-diagram runs last, once
#                          every experiment has finished).
#
# Each test runs as `mpirun -n NPROCS python TEST -nlocal NLOCAL`.
# Failures stop the loop (set -e); to keep going through failures
# pass `KEEPGOING=1`.
#
# Tunable variables (override on the command line, e.g. `make disc NPROCS=8`):
#   NPROCS        number of MPI ranks (default 4)
#   NLOCAL        particles per rank (default 250 000)
#   PYTHON        python interpreter (default `python`)
#   MPIRUN        mpirun executable (default `mpirun`)
#   MPIRUN_FLAGS  extra flags (default `--use-hwthread-cpus`)
#   KEEPGOING     1 to continue through individual test failures (default 0)

NPROCS       ?= 4
NLOCAL       ?= 2500000
PYTHON       ?= python
MPIRUN       ?= mpirun
MPIRUN_FLAGS ?= --use-hwthread-cpus
KEEPGOING    ?= 0

TEST_DIR  := tests/cfmz
BOLTZ_DIR := tests/boltzmann

# Homogeneous Boltzmann tests in tests/boltzmann — sorted numerically.
BOLTZMANN_TESTS := $(shell ls $(BOLTZ_DIR)/test_[0-9]*.py 2>/dev/null | sort -t_ -k2 -n)

# Discotic tests — sorted numerically by the trailing index so test_disc_2
# runs before test_disc_10 etc.
DISC_TESTS := $(shell ls $(TEST_DIR)/test_disc_*.py 2>/dev/null | sort -t_ -k3 -n)

# Needle smectic / Enskog tests (the new Phase-3 work).
NEEDLE_SMECTIC_TESTS := $(sort \
    $(wildcard $(TEST_DIR)/test_needle_smectic_*.py) \
    $(wildcard $(TEST_DIR)/test_needle_sod_*.py))

# Non-smectic needle tests:
#   - test_needle_<N>.py for the homogeneous calamitic suite, sorted
#     numerically by the trailing index.
#   - test_needle_inhomo_<N>.py for the inhomogeneous Sod-tube test.
NEEDLE_TESTS_NUMERIC := $(shell ls $(TEST_DIR)/test_needle_[0-9]*.py 2>/dev/null | sort -t_ -k3 -n)
NEEDLE_TESTS_INHOMO  := $(shell ls $(TEST_DIR)/test_needle_inhomo_*.py 2>/dev/null | sort -t_ -k4 -n)
NEEDLE_TESTS         := $(NEEDLE_TESTS_NUMERIC) $(NEEDLE_TESTS_INHOMO)

# Shell prefix: stop on first failure unless KEEPGOING=1.
SHELL_PREFIX := $(if $(filter 1,$(KEEPGOING)),set +e,set -e)

# A single test invocation, used by every target.  Each test is run
# under mpirun with the configured rank count and particle budget.
define run_one
	echo "--- $(1) ---"; \
	$(MPIRUN) $(MPIRUN_FLAGS) -n $(NPROCS) $(PYTHON) $(1) -nlocal $(NLOCAL); \
	echo
endef

.PHONY: all boltzmann disc needle needle-smectic phase-diagram clean help

# Simulation output lives in these directories — wiped by `make clean`.
OUTPUT_DIRS := output $(BOLTZ_DIR)/output $(TEST_DIR)/output

help:
	@echo "DSMC test runner"
	@echo
	@echo "Targets:"
	@echo "  make boltzmann       run $(words $(BOLTZMANN_TESTS)) homogeneous Boltzmann tests"
	@echo "  make disc            run $(words $(DISC_TESTS)) discotic tests"
	@echo "  make needle          run $(words $(NEEDLE_TESTS)) non-smectic needle tests"
	@echo "  make needle-smectic  run $(words $(NEEDLE_SMECTIC_TESTS)) Enskog / smectic needle tests"
	@echo "  make phase-diagram   (re)plot the non-smectic I->N phase diagram (test_needle_27)"
	@echo "  make all             all of the above (phase-diagram last)"
	@echo "  make clean           remove all simulation outputs ($(OUTPUT_DIRS))"
	@echo
	@echo "Variables (override on the command line):"
	@echo "  NPROCS=$(NPROCS)  NLOCAL=$(NLOCAL)  KEEPGOING=$(KEEPGOING)"
	@echo "  PYTHON=$(PYTHON)  MPIRUN=$(MPIRUN)"
	@echo "  MPIRUN_FLAGS=$(MPIRUN_FLAGS)"

boltzmann:
	@$(SHELL_PREFIX); for t in $(BOLTZMANN_TESTS); do \
	  $(call run_one,$$t); \
	done

disc:
	@$(SHELL_PREFIX); for t in $(DISC_TESTS); do \
	  $(call run_one,$$t); \
	done

needle:
	@$(SHELL_PREFIX); for t in $(NEEDLE_TESTS); do \
	  $(call run_one,$$t); \
	done

needle-smectic:
	@$(SHELL_PREFIX); for t in $(NEEDLE_SMECTIC_TESTS); do \
	  $(call run_one,$$t); \
	done

# Regenerate the non-smectic isotropic->nematic phase diagram from the
# per-temperature history.pickle files written by test_needle_27.  Pure
# post-processing — single process, no mpirun — so it can be re-run
# cheaply after the sweep to restyle the plot.  (The smectic sweep
# plots itself inline, so there is no separate target for it.)
phase-diagram:
	@echo "--- phase-diagram: non-smectic I->N (test_needle_27) ---"
	@$(PYTHON) $(TEST_DIR)/plot_phase_diagram_27.py
	@echo

# phase-diagram is listed last so it runs only after every experiment
# (including the test_needle_27 sweep inside `needle`) has completed.
all: boltzmann disc needle needle-smectic phase-diagram

clean:
	@for d in $(OUTPUT_DIRS); do \
	  if [ -d "$$d" ]; then \
	    echo "removing $$d"; \
	    rm -rf "$$d"; \
	  fi; \
	done
