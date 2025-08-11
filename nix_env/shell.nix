# shell.nix
with import <nixpkgs> {};

mkShell rec {
  # Only needed if you're dealing with compiled libraries (C extensions)
  NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
    stdenv.cc.cc
    zlib
  ];

  LD_LIBRARY_PATH = NIX_LD_LIBRARY_PATH;
  NIX_LD = lib.fileContents "${stdenv.cc}/nix-support/dynamic-linker";

  buildInputs = [
    # Python with pip, setuptools, wheel, virtualenv
    (python311.withPackages (ps: with ps; [
      pip
      setuptools
      wheel
      virtualenv
    ]))

    cmake
  ];

  shellHook = ''
    if [ -d ".venv" ]; then
      echo "Activating existing virtual environment..."
      source .venv/bin/activate
    else
      echo "No virtual environment found."
      echo "Run: python -m venv .venv && source .venv/bin/activate"
    fi
  '';
}

