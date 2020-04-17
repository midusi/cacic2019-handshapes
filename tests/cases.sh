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

test_cases="$@"
if has_param '--all' "$@"; then
    test_cases=$(find $ROOT -name "*.conf")
fi
