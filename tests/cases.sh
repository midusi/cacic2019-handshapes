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

. $ROOT/rwth_tests.sh
. $ROOT/lsa16_tests.sh
. $ROOT/ciarp_tests.sh

rwth_flags=""
lsa16_flags=""
ciarp_flags=""

for key in ${!rwth_config[*]}; do rwth_flags+=" --$key ${rwth_config[$key]}"; done
for key in ${!lsa16_config[*]}; do lsa16_flags+=" --$key ${lsa16_config[$key]}"; done
for key in ${!ciarp_config[*]}; do ciarp_flags+=" --$key ${ciarp_config[$key]}"; done

declare -A tests_cases

if has_param '--rwth' "$@"; then tests_cases["rwth"]="$rwth_flags"; fi
if has_param '--lsa16' "$@"; then tests_cases["lsa16"]="$lsa16_flags"; fi
if has_param '--ciarp' "$@"; then tests_cases["ciarp"]="$ciarp_flags"; fi
