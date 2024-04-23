#!/bin/bash
SUBDIRS="../src ../src/* ../src/*/* ../src/*/*/* ../include ../include/* ../include/*/* ../include/*/*/*"
FILETYPES="*.hpp *.h *.inl *.cc *.cpp *.c *.cu *.cuh"
CLANG_FORMAT="clang-format"
echo "start format..."
for d in ${SUBDIRS}
do
    echo "clang-format format subdir: $d "
    for t in ${FILETYPES}
    do
        for file in $d/$t
        do
            echo ">>>>>>>>>> format file: $file "
            if test -f $file
            then
               ${CLANG_FORMAT} -style=file:../.clang-format -i $file
            fi
        done
    done
done
echo "format end."
