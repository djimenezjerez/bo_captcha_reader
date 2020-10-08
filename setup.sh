#!/usr/bin/env bash

source /etc/os-release
commands=(python3 firefox geckodriver)

function verifyCommand {
    search=${NAME// /+}
    if ! [ -x "$(command -v $1)" ];
    then
        search="${search}+$1"
        echo -e "$(tput setaf 1)\xF0\x9F\x97\x99$(tput sgr0) Debe instalar la herramienta $1: $(tput setaf 4)http://www.google.com/search?q=$search$(tput sgr0)"
        return 0
    else
        echo -e "$(tput setaf 2)\xE2\x9C\x93$(tput sgr0) La herramienta $1 se ha instalado correctamente"
        return 1
    fi
}

function main() {
    for command in "${commands[@]}"
    do
        verifyCommand $command $i
    done
}

main