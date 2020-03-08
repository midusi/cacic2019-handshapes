# DONT EXECUTE THIS CODE!!
# Just include to load tests cases
# by defining ROOT dir

has_param() {
    local term="$1"
    shift
    for arg; do
        if [[ $arg == "$term" ]]; then
            return 0
        fi
    done
    return 1
}

if has_param '--show' "$@"; then
    echo "Tests list:"
    echo
    for i in "${!tests[@]}"; do 
        printf "\t%s\t%s\n" "$i" "${tests[$i]}"
    done
    exit 0
fi

for test in "${tests[@]}"; do
    printf "Running test \t%s\n" $test
    $test && echo "Test finished successfuly" || {
        echo "Test $test failed"
        exit 1
    }
done
