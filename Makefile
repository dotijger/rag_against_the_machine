src_dir = src
mypy_flags = --warn-return-any \
			 --warn-unused-ignores \
			 --ignore-missing-imports \
			 --disallow-untyped-defs \
			 --check-untyped-defs
args ?=

all: install run

install:
	uv sync

run:
	uv run python3 -m $(src_dir) $(args)

debug:
	uv run python3 -m pdb -m $(src_dir) $(args)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -f data/log

lint:
	uv run flake8 .
	uv run mypy . $(mypy_flags)

lint-strict:
	uv run flake8 .
	uv run mypy . --strict



