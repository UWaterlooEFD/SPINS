#!/bin/bash
find . -regextype egrep -iregex ".*\.[ch]p?p?" | xargs -I % clang-format -i  %
