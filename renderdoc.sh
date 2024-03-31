#!/bin/sh

set -xe

clear
cargo b
mkdir -p /tmp/renderdoc/
find /tmp/renderdoc/ -type f -exec rm {} +
renderdoccmd capture -c /tmp/renderdoc/ ../target/debug/game
