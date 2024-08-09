let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell{
 packages = [
  (pkgs.python3.withPackages (python-pkgs: [
    python-pkgs.pip
    python-pkgs.setuptools
    python-pkgs.build
    python-pkgs.sphinx
  ]))
 ];
 shellHook = ''
 zsh
 '';
}
