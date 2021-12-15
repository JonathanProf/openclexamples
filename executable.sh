clear
FILE=./$1.out
if test -f "$FILE"; then
    rm $1.out
fi
g++ -Wall $1.cpp -o  $1.out -lOpenCL -std=c++11
./$1.out
