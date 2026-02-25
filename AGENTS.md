# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

ToyAI is a C++ educational project (with Python tooling) for learning transformers from first principles. There are no web services, databases, or external dependencies.

### Building C++ examples

The default C++ compiler symlink (`/usr/bin/c++`) points to Clang, which may fail to link due to missing `-lstdc++`. Always specify g++ explicitly when running CMake:

```bash
mkdir -p build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ .. && make -j$(nproc)
```

All 7 example executables are output under `build/examples/example<N>_<name>/example<N>`.

### Python tooling

- **Diagram generator**: `python3 -m diagram_generator generate <json> -o <output.svg>` (stdlib only, no pip deps)
- **Book builder**: `python3 scripts/build_book.py [--latex|--pdf|--all]` (PDF requires `pandoc` and `texlive-latex-base texlive-latex-extra`, which are optional)

### Standard commands

See `README.md` for build/run instructions. No lint or test frameworks are configured in this repo.
