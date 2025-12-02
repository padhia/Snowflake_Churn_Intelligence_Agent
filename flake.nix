{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
  let
    eachSystem = system:
    let
      pkgs = import nixpkgs { inherit system; };

    in {
      devShells.default = pkgs.mkShell {
        name = "aiml-dashboard";
        venvDir = "./.venv";
        buildInputs = with pkgs.python312Packages; [
          pkgs.ruff
          pkgs.uv
          python
          venvShellHook
        ];
      };
    };

  in {
    inherit (flake-utils.lib.eachDefaultSystem eachSystem) devShells;
  };
}
