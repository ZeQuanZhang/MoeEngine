#!/bin/zsh
exec /Library/Developer/CommandLineTools/usr/bin/swiftc \
  -interface-compiler-version 6.3.2 "$@"
