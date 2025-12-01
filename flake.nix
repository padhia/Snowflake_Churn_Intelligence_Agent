{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    snowflake.url = "github:padhia/snowflake";

    snowflake.inputs.nixpkgs.follows = "nixpkgs";
    snowflake.inputs.flake-utils.follows = "flake-utils";
  };

  outputs = { nixpkgs, flake-utils, snowflake, ... }:
  let
    eachSystem = system:
    let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [
          snowflake.overlays.default
        ];
      };

    in {
      devShells.default = pkgs.mkShell {
        name = "aiml-dashboard";
        venvDir = "./.venv";
        buildInputs = with pkgs.python3Packages; [
          pkgs.ruff
          pkgs.uv
          python
          venvShellHook
          snowflake-snowpark-python
          pandas-stubs
          streamlit
          watchdog
        ];
      };
    };

  in {
    inherit (flake-utils.lib.eachDefaultSystem eachSystem) devShells;
  };
}
