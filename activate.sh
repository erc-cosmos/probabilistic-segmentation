# Source this file: source ./activate.sh
# Works in bash and zsh; changes only this terminal session.
if [ -n "${ZSH_VERSION:-}" ]; then
  _cosmos_file="${(%):-%x}"
else
  _cosmos_file="${BASH_SOURCE[0]}"
fi
_cosmos_root="$(cd -- "$(dirname -- "$_cosmos_file")" && pwd)"
if [ ! -x "$_cosmos_root/.tools/bin/pixi" ]; then
  echo 'Run bash bootstrap.sh first.' >&2
  unset _cosmos_file _cosmos_root
  return 1
fi
export PATH="$_cosmos_root/.tools/bin:$PATH"
export PIXI_CACHE_DIR="$_cosmos_root/.cache/pixi"
unset _cosmos_file _cosmos_root
