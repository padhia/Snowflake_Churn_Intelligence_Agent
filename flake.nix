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
        name = "churn";
        venvDir = "./.venv";
        buildInputs = with pkgs.python312Packages; [
          pkgs.ruff
          pkgs.uv
          python
          venvShellHook
        ];
        postVenvCreation = ''
          unset SOURCE_DATE_EPOCH
          pip install -r requirements.txt
        '';
      };
    };

  in {
    inherit (flake-utils.lib.eachDefaultSystem eachSystem) devShells;
  };
}
