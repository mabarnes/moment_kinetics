#!/usr/bin/env bash

# Detect operating system, following
# https://stackoverflow.com/questions/3466166/how-to-check-if-running-in-cygwin-mac-or-linux
case "$(uname -s)" in

   Darwin*)
     machines/shared/get-julia-macos.sh $@
     ;;

   Linux*)
     machines/shared/get-julia-linux-x86_64.sh $@
     ;;

   CYGWIN*|MINGW*|MINGW32*|MSYS*)
     echo "Setting up on Windows is unsupported"
     exit 1
     OS=windows
     ;;

   # Add here more strings to compare
   # See correspondence table at the bottom of this answer

   *)
     echo "Unrecognised operaing system, 'uname -s'= $(uname -s)"
     exit 1
     ;;
esac

exit 0
