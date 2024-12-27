{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.virtualenv
    pkgs.python3Packages.pyserial
    pkgs.python3Packages.pandas
    pkgs.python3Packages.numpy
    pkgs.python3Packages.tkinter
    pkgs.python3Packages.joblib
    pkgs.python3Packages.scikit-learn
  ];

  shellHook = ''
    # Create a virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
      virtualenv .venv
    fi

    # Activate the virtual environment
    source .venv/bin/activate

    # Install Python dependencies
    pip install -r requirements.txt || true
  '';
}