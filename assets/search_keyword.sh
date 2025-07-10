#!/bin/bash


keyword=$1

if [ -z "${keyword}" ]; then

        echo "Usage: bash $0 [KEYWORD]"

else

        mapfile -t flist < <(find . -type f \( -name "*.py" -o -name "*.yaml" \) -a \! -name "*.npy" -a \! -name "*__*")

        for fn in "${flist[@]}"; do

                to_list=$(grep -l "${keyword}" "${fn}" | wc -l)

                if [ "$to_list" -ne "0" ]; then

                        echo "=========================================
                        "

                        grep -n "${keyword}" $fn

                        echo "
                        === === ===
                        "

                fi

        done

fi
