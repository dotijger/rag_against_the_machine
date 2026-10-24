src_dir = src
mypy_flags = --warn-return-any \
			 --warn-unused-ignores \
			 --ignore-missing-reports \
			 --disallow-untyped-defs \
			 --check-untyped-defs

all: install run

install:
	uv sync

run:
	uv run python3 -m $(src_dir)

debug:
	uv run python3 -m pdb -m $(src_dir)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	uv run flake8 $(src_dir)
	uv run mypy $(src_dir) $(mypy_flags)

lint-strict:
	uv run flake8 $(src_dir)
	uv run mypy $(src_dir) --strict



