# !!! FLAKE TEMPLATE FROM https://github.com/cjdell/node-nix/blob/master/flake.nix !!!

{
  description = "Node.js TypeScript Project Flake";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShell = pkgs.mkShell {
          # Shell environment setup for development
          buildInputs = with pkgs; [
            git git-lfs

            nodejs_20 nodePackages.npm
            nodePackages.typescript
            cloudflared

            python312
            python312Packages.pip
            python312Packages.virtualenv
            python312Packages.coloredlogs

            python312Packages.polars # We are migrating to polars!
            python312Packages.numpy
            python312Packages.pyarrow

            python312Packages.httpx
            python312Packages.tenacity

            python312Packages.torch
            python312Packages."sentence-transformers"
            python312Packages.spacy
            python312Packages.faiss # not faiss-cpu, nix has a different naming scheme
            python312Packages.tqdm
            python312Packages.scikit-learn

            python312Packages.matplotlib
            python312Packages.plotly
            python312Packages.seaborn

            python312Packages.grpcio
            python312Packages.grpcio-tools
          ];

          # Optional: Set some environment variables if needed
          shellHook = ''
            export GOOGLE_CLOUD_PROJECT="reelraider"
            echo "Welcome to the Node.js TypeScript development shell!"
          '';
        };

        # Build process: You can customize this if you have specific build steps
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "node-nix";
          version = "1.0.0";

          # Use the project directory (might be good to limit this to just source at some point)
          src = ./.;

          # Build dependencies for building the production package
          buildInputs = [
            pkgs.nodejs
            pkgs.nodePackages.typescript
            pkgs.cacert
            pkgs.python312
          ];

          # NPM install and TypeScript compile
          buildPhase = ''
            # Need for NPM to work
            mkdir -p tmp-npm
            HOME=tmp-npm

            npm install
            npm run build
          '';

          # Out the things we need to the output
          installPhase = ''
            mkdir -p            $out/lib/
            cp -r out           $out/lib/
            cp    package.json  $out/lib/
            cp -r node_modules  $out/lib/
          '';

          # Set metadata
          meta = with pkgs.lib; {
            description = "A Node.js TypeScript project built with Nix";
            license = licenses.mit;
            platforms = platforms.all;
          };

          # Fixed output derivation. Means we can do impure things like access the internet (for NPM to work) as long as we lock down the output hash
          outputHashAlgo = "sha256";
          outputHashMode = "recursive";
          outputHash = "sha256-Zah7U1zkOQdeIduKrWN3/Yryxa4wS+MK20FQIgUvjSA=";
        };
      });
}