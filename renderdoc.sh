#!/bin/sh

#alias renderdoccmd="$HOME/Documents/gpu/renderdoc/build/bin/renderdoccmd"

set -xe

clear
cargo b
mkdir -p /tmp/renderdoc/
find /tmp/renderdoc/ -type f -exec rm {} +
renderdoccmd capture -c /tmp/renderdoc/ "$1"
