#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIXI="$ROOT/.tools/bin/pixi"
VERSION=0.81.0
export PIXI_CACHE_DIR="$ROOT/.cache/pixi"
ACTION="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$ACTION" = bootstrap ]; then
  if [ ! -x "$PIXI" ] || [ "$("$PIXI" --version)" != "pixi $VERSION" ]; then
    installer="$(mktemp)"
    trap 'rm -f "$installer"' EXIT
    curl --fail --silent --show-error --location "https://raw.githubusercontent.com/prefix-dev/pixi/v$VERSION/install/install.sh" -o "$installer"
    PIXI_VERSION="$VERSION" PIXI_HOME="$ROOT/.tools" PIXI_BIN_DIR="$ROOT/.tools/bin" PIXI_NO_PATH_UPDATE=1 bash "$installer"
  fi
  "$PIXI" --version
  echo "Pixi is ready. Next: bash cosmos.sh setup"
  exit 0
fi
if [ "$ACTION" = help ]; then
  echo 'Usage: bash cosmos.sh {bootstrap|setup|add|update|upgrade|run COMMAND [ARGS...]}'
  exit 0
fi
if [ ! -x "$PIXI" ]; then echo 'Run bash bootstrap.sh first.' >&2; exit 1; fi
if [ "$("$PIXI" --version)" != "pixi $VERSION" ]; then echo 'Pixi version mismatch. Run bash bootstrap.sh.' >&2; exit 1; fi
case "$ACTION" in
  setup) "$PIXI" install --locked --manifest-path "$ROOT/pixi.toml" ;;
  update) "$PIXI" install --manifest-path "$ROOT/pixi.toml" ;;
  upgrade) "$PIXI" update --manifest-path "$ROOT/pixi.toml" ;;
  add)
    read -r -p 'Package name (e.g. numpy): ' package
    read -r -p 'Exact version (e.g. 1.26.4): ' version
    read -r -p 'Source: conda or pypi [conda]: ' source
    [[ "$package" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid package name.' >&2; exit 1; }
    [[ "$version" =~ ^[0-9][A-Za-z0-9.!+_-]*$ ]] || { echo 'Enter an exact version, without comparison signs.' >&2; exit 1; }
    case "${source:-conda}" in
      conda) "$PIXI" add --manifest-path "$ROOT/pixi.toml" "$package==$version" ;;
      pypi) "$PIXI" add --manifest-path "$ROOT/pixi.toml" --pypi "$package==$version" ;;
      *) echo 'Source must be conda or pypi.' >&2; exit 1 ;;
    esac ;;
  run)
    [ "$#" -gt 0 ] || { echo 'Example: bash cosmos.sh run python your_script.py' >&2; exit 1; }
    "$PIXI" run --locked --manifest-path "$ROOT/pixi.toml" -- "$@" ;;
  *) echo "Unknown action: $ACTION" >&2; exit 1 ;;
esac
